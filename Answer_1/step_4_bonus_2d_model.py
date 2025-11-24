# --- Bonus Question: 2D Fuel Flow Model (Speed + Altitude) ---

"""
Goal
-----
Build a fuel-flow model using both speed and altitude, and report what is
actually learnable given the data (coverage, variance structure, CV metrics).

Why this is non-trivial
-----------------------
- Large between-flight differences (aircraft/weight/ISA/engine settings).
- Cruise-heavy data reduces within-flight variation (altitude and speed
  can be nearly constant).
- We must separate aircraft offsets (between flights) from true physics
  effects (within flights).
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV, HuberRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
BASE = Path("../../question_1")
DATA_FILES = {
    "fuel_flow": BASE / "signals_fuel_flow.pkl",
    "altitude":  BASE / "signals_altitude.pkl",
    "wind":      BASE / "signals_wind.pkl",
    "speed":     BASE / "signals_vitesse.pkl",
}

# Generous initial ranges; we’ll report the effective range from the data
ALTITUDE_RANGE = (0, 60000)      # ft
SPEED_RANGE = (100, 1000)        # km/h
FUEL_SCALE = 1000.0              # multiply fuel_flow by this factor (unit correction)

MIN_DATA_POINTS = 100
MIN_FLIGHTS = 5

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def make_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns to numeric, coercing errors to NaN."""
    return df.apply(pd.to_numeric, errors="coerce")

def clip_outliers(values, lower_pct=0.5, upper_pct=99.5):
    """Percentile-based clipping with guards for small samples."""
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if arr.size < 20 or finite.sum() < 20:
        return values
    lo = np.nanpercentile(arr[finite], lower_pct)
    hi = np.nanpercentile(arr[finite], upper_pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return values
    return np.clip(arr, lo, hi)

def air_density_kg_m3(alt_ft: float) -> float:
    """ISA density up to stratosphere (sufficient for 0–60 kft)."""
    alt_m = float(alt_ft) * 0.3048
    T0, p0, L, R, g0 = 288.15, 101325.0, -0.0065, 287.058, 9.80665
    if alt_m <= 11000.0:
        T = T0 + L * alt_m
        p = p0 * (T / T0) ** (-g0 / (L * R))
    else:
        T11 = T0 + L * 11000.0
        p11 = p0 * (T11 / T0) ** (-g0 / (L * R))
        p = p11 * np.exp(-g0 * (alt_m - 11000.0) / (R * T11))
        T = T11
    return p / (R * T)

def dynamic_pressure_pa(alt_ft, speed_kmh):
    """q = 0.5 * rho * V^2, with V in m/s."""
    V_ms = np.asarray(speed_kmh, dtype=float) * (1000.0 / 3600.0)
    rho = np.vectorize(air_density_kg_m3)(np.asarray(alt_ft, dtype=float))
    return 0.5 * rho * (V_ms ** 2)

def print_cv(model_name, cv):
    r2 = cv["test_r2"]
    rmse = -cv["test_neg_root_mean_squared_error"]
    mae = -cv["test_neg_mean_absolute_error"]
    print(f"  {model_name:32s} | R²={np.mean(r2):+7.3f}±{np.std(r2):.3f} | "
          f"RMSE={np.mean(rmse):.4f} | MAE={np.mean(mae):.4f}")

# ---------------------------------------------------------------------
# Step 1: Load and merge
# ---------------------------------------------------------------------
print("=" * 80)
print("BONUS: 2D Fuel Flow Model (Speed + Altitude)")
print("=" * 80)

print("\n[1/6] Loading data...")
raw = {name: make_numeric(pd.read_pickle(path)) for name, path in DATA_FILES.items()}
all_ids = sorted(set.intersection(*[set(df.columns) for df in raw.values()]))
print(f"  Flights with complete signals: {len(all_ids)}")

def merge_flight(fid):
    frames = []
    for name, df in raw.items():
        s = df[fid].dropna().rename(name).to_frame()
        frames.append(s)
    out = frames[0].join(frames[1:], how="inner")
    out["flight_id"] = fid
    return out.reset_index(drop=True)

full = pd.concat([merge_flight(fid) for fid in all_ids], ignore_index=True)
for c in ["fuel_flow", "altitude", "speed", "wind"]:
    full[c] = pd.to_numeric(full[c], errors="coerce")

full["fuel_flow"] = full["fuel_flow"] * FUEL_SCALE

clean = full.dropna(subset=["fuel_flow", "altitude", "speed"]).copy()
clean = clean[
    (clean["fuel_flow"] > 0) &
    (clean["altitude"].between(*ALTITUDE_RANGE)) &
    (clean["speed"].between(*SPEED_RANGE))
]

for col in ["fuel_flow", "altitude", "speed"]:
    clean[col] = clip_outliers(clean[col].values)

n_obs = len(clean)
n_flights = clean["flight_id"].nunique()
print(f"  After cleaning: {n_obs:,} observations from {n_flights} flights")

if n_obs < MIN_DATA_POINTS or n_flights < MIN_FLIGHTS:
    print("  Not enough data after cleaning. Stop.")
    sys.exit(0)

# ---------------------------------------------------------------------
# Step 2: Data exploration
# ---------------------------------------------------------------------
print("\n[2/6] Determining effective range and 2D coverage…")
alt_p5, alt_p95 = clean["altitude"].quantile([0.05, 0.95])
spd_p5, spd_p95 = clean["speed"].quantile([0.05, 0.95])
ff_p5, ff_p95 = clean["fuel_flow"].quantile([0.05, 0.95])

print("  Effective range (P5–P95):")
print(f"    Altitude: {alt_p5:8.0f} – {alt_p95:8.0f} ft")
print(f"    Speed:       {spd_p5:5.0f} –     {spd_p95:5.0f} km/h")
print(f"    Fuel flow:    {ff_p5:6.2f} –      {ff_p95:6.2f} lb/s")

corr_alt_spd = clean[["altitude", "speed"]].corr().iloc[0, 1]
print(f"\n  Altitude–Speed correlation: {corr_alt_spd:+.3f} "
      f"({'high' if abs(corr_alt_spd) > 0.85 else 'moderate'})")

var_by_flight = clean.groupby("flight_id").agg(
    altitude_cv=("altitude", lambda x: x.std() / (x.mean() + 1e-9)),
    speed_cv=("speed", lambda x: x.std() / (x.mean() + 1e-9)),
    n=("altitude", "count"),
)
print(f"  Within-flight median CV — altitude: {var_by_flight['altitude_cv'].median():.4f}, "
      f"speed: {var_by_flight['speed_cv'].median():.4f}")

# ---------------------------------------------------------------------
# Step 3: Variance decomposition (log1p target)
# ---------------------------------------------------------------------
print("\n[3/6] Variance decomposition (log1p fuel)…")
clean["log_fuel"] = np.log1p(clean["fuel_flow"])
grand_mean = clean["log_fuel"].mean()

# Sum of squares
flight_mean = clean.groupby("flight_id")["log_fuel"].transform("mean")
ss_between = ((flight_mean - grand_mean) ** 2).sum()
ss_total = ((clean["log_fuel"] - grand_mean) ** 2).sum()
ss_within = ss_total - ss_between

# Report as proportions via sums of squares
icc = ss_between / ss_total if ss_total > 0 else np.nan
print(f"  SS total:   {ss_total:.4f}")
print(f"  SS between: {ss_between:.4f} ({100*icc:.1f}% of total)")
print(f"  SS within:  {ss_within:.4f} ({100*(1-icc):.1f}% of total)")
print(f"  ICC (between / total): {icc:.3f}")

# ---------------------------------------------------------------------
# Step 4: Models (between vs within)
# ---------------------------------------------------------------------
print("\n[4/6] Building models…")

# Physics features
clean["q"] = dynamic_pressure_pa(clean["altitude"], clean["speed"])
clean["inv_q"] = 1.0 / np.maximum(clean["q"], 1e-9)

# Between-flight aggregation (medians)
summary = clean.groupby("flight_id").agg(
    fuel_flow=("fuel_flow", "median"),
    log_fuel=("log_fuel", "median"),
    altitude=("altitude", "median"),
    speed=("speed", "median"),
    q=("q", "median"),
    inv_q=("inv_q", "median"),
).reset_index()

y_between = summary["log_fuel"].values
g_between = summary["flight_id"].values
k_between = min(5, len(summary))

print(f"  Cross-validation folds — between: {k_between}")

# Between: Linear [alt, speed]
X_b_lin = summary[["altitude", "speed"]].values
pipe_b_lin = Pipeline([
    ("scale", StandardScaler()),
    ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 25)))
])
cv_b_lin = cross_validate(
    pipe_b_lin, X_b_lin, y_between, groups=g_between,
    cv=GroupKFold(n_splits=k_between),
    scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
    return_train_score=False, error_score="raise"
)
pipe_b_lin.fit(X_b_lin, y_between)
print_cv("Between Linear [alt, speed]", cv_b_lin)

# Between: Interactions
X_b_int = summary[["altitude", "speed"]].values
pipe_b_int = Pipeline([
    ("scale", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("ridge", RidgeCV(alphas=np.logspace(-3, 5, 25)))
])
cv_b_int = cross_validate(
    pipe_b_int, X_b_int, y_between, groups=g_between,
    cv=GroupKFold(n_splits=k_between),
    scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
    return_train_score=False, error_score="raise"
)
pipe_b_int.fit(X_b_int, y_between)
print_cv("Between Interactions [alt, speed, alt*speed]", cv_b_int)

# Between: Physics [q, 1/q]
X_b_phy = summary[["q", "inv_q"]].values
pipe_b_phy = Pipeline([
    ("scale", StandardScaler()),
    ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 25)))
])
cv_b_phy = cross_validate(
    pipe_b_phy, X_b_phy, y_between, groups=g_between,
    cv=GroupKFold(n_splits=k_between),
    scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
    return_train_score=False, error_score="raise"
)
pipe_b_phy.fit(X_b_phy, y_between)
print_cv("Between Physics [q, 1/q]", cv_b_phy)

# Within-flight demeaned targets and features
clean["log_fuel_d"] = clean.groupby("flight_id")["log_fuel"].transform(lambda s: s - s.mean())
clean["altitude_d"] = clean.groupby("flight_id")["altitude"].transform(lambda s: s - s.mean())
clean["speed_d"] = clean.groupby("flight_id")["speed"].transform(lambda s: s - s.mean())
clean["q_d"] = clean.groupby("flight_id")["q"].transform(lambda s: s - s.mean())
clean["inv_q_d"] = clean.groupby("flight_id")["inv_q"].transform(lambda s: s - s.mean())

y_within = clean["log_fuel_d"].values
g_within = clean["flight_id"].values
k_within = min(5, clean["flight_id"].nunique())

print(f"  Cross-validation folds — within:  {k_within}")

# Within: Linear [Δalt, Δspeed]
X_w_lin = clean[["altitude_d", "speed_d"]].values
pipe_w_lin = Pipeline([
    ("scale", StandardScaler()),
    ("huber", HuberRegressor()),
])
cv_w_lin = cross_validate(
    pipe_w_lin, X_w_lin, y_within, groups=g_within,
    cv=GroupKFold(n_splits=k_within),
    scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
    return_train_score=False, error_score="raise"
)
pipe_w_lin.fit(X_w_lin, y_within)
print_cv("Within Linear [Δalt, Δspeed]", cv_w_lin)

# Within: Physics [Δq, Δ(1/q)]
X_w_phy = clean[["q_d", "inv_q_d"]].values
pipe_w_phy = Pipeline([
    ("scale", StandardScaler()),
    ("huber", HuberRegressor()),
])
cv_w_phy = cross_validate(
    pipe_w_phy, X_w_phy, y_within, groups=g_within,
    cv=GroupKFold(n_splits=k_within),
    scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
    return_train_score=False, error_score="raise"
)
pipe_w_phy.fit(X_w_phy, y_within)
print_cv("Within Physics [Δq, Δ(1/q)]", cv_w_phy)

# ---------------------------------------------------------------------
# Step 5: Diagnose performance
# ---------------------------------------------------------------------
print("\n[5/6] Model performance summary…")
best_between_r2 = max(np.mean(cv_b_lin["test_r2"]),
                      np.mean(cv_b_int["test_r2"]),
                      np.mean(cv_b_phy["test_r2"]))
best_within_r2 = max(np.mean(cv_w_lin["test_r2"]),
                     np.mean(cv_w_phy["test_r2"]))

print(f"  Best between-flight R²: {best_between_r2:+.3f}")
print(f"  Best within-flight  R²: {best_within_r2:+.3f}")

# Outlier inspection for between-flight physics model
y_between_pred = pipe_b_phy.predict(X_b_phy)
errors = np.abs(y_between - y_between_pred)
worst_idx = np.argmax(errors)
worst_fid = summary.iloc[worst_idx]["flight_id"]
worst_err = errors[worst_idx]
print("  Between-flight physics model largest absolute error "
      f"(log1p units): {worst_err:.2f} on flight_id={worst_fid}")

# ---------------------------------------------------------------------
# Step 6: Visualization
# ---------------------------------------------------------------------
print("\n[6/6] Creating diagnostic figure…")
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# Coverage
ax1 = fig.add_subplot(gs[0, 0])
sc = ax1.scatter(clean["speed"], clean["altitude"],
                 c=clean["log_fuel"], s=8, alpha=0.4, cmap="viridis")
ax1.set_xlabel("Speed (km/h)")
ax1.set_ylabel("Altitude (ft)")
ax1.set_title(f"2D coverage (n={n_obs:,}) | corr(alt,spd)={corr_alt_spd:+.2f}")
plt.colorbar(sc, ax=ax1, label="log1p(fuel)")

# Sample trajectories
ax2 = fig.add_subplot(gs[0, 1])
for i, fid in enumerate(clean["flight_id"].unique()[:5]):
    fdf = clean[clean["flight_id"] == fid]
    ax2.plot(fdf["speed"], fdf["altitude"], alpha=0.7, linewidth=2, label=f"Flight {i}")
ax2.set_xlabel("Speed (km/h)")
ax2.set_ylabel("Altitude (ft)")
ax2.set_title("Sample flight trajectories")
ax2.legend(fontsize=8)

# Variance decomposition (using SS proportions)
ax3 = fig.add_subplot(gs[0, 2])
bars = ax3.bar(["Between\nflights", "Within\nflights"],
               [ss_between, ss_within],
               color=["#e74c3c", "#3498db"], alpha=0.8)
ax3.set_ylabel("Sum of squares (log1p fuel)")
ax3.set_title(f"Variance decomposition (ICC={icc:.3f})")
for bar, ss, frac in zip(bars, [ss_between, ss_within], [icc, 1-icc]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
             f"{100*frac:.1f}%", ha="center", va="center", color="white", fontweight="bold")

# Between-flight scatter (physics model)
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(y_between, y_between_pred, s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
low = min(y_between.min(), y_between_pred.min())
high = max(y_between.max(), y_between_pred.max())
ax4.plot([low, high], [low, high], 'r--', lw=2, alpha=0.7)
ax4.set_xlabel("Actual log1p(fuel)")
ax4.set_ylabel("Predicted log1p(fuel)")
ax4.set_title("Between-flight model (physics)")

# Within-flight scatter (linear)
ax5 = fig.add_subplot(gs[1, 1])
y_within_pred = pipe_w_lin.predict(X_w_lin)
nplot = min(5000, len(y_within))
idx = np.random.choice(len(y_within), nplot, replace=False)
ax5.scatter(y_within[idx], y_within_pred[idx], s=5, alpha=0.3)
ax5.plot([y_within.min(), y_within.max()],
         [y_within.min(), y_within.max()], 'r--', lw=2, alpha=0.7)
ax5.set_xlabel("Actual Δlog1p(fuel)")
ax5.set_ylabel("Predicted Δlog1p(fuel)")
ax5.set_title("Within-flight model (linear)")

# Within residuals
ax6 = fig.add_subplot(gs[1, 2])
res = y_within[idx] - y_within_pred[idx]
ax6.scatter(y_within_pred[idx], res, s=5, alpha=0.3)
ax6.axhline(0, color='r', lw=2, linestyle='--', alpha=0.7)
ax6.set_xlabel("Predicted Δlog1p(fuel)")
ax6.set_ylabel("Residual")
ax6.set_title("Within-flight residuals")

# CV comparison
ax7 = fig.add_subplot(gs[2, :])
names = ["Between\nLinear", "Between\nInteractions", "Between\nPhysics",
         "Within\nLinear", "Within\nPhysics"]
means = [np.mean(cv_b_lin["test_r2"]),
         np.mean(cv_b_int["test_r2"]),
         np.mean(cv_b_phy["test_r2"]),
         np.mean(cv_w_lin["test_r2"]),
         np.mean(cv_w_phy["test_r2"])]
stds = [np.std(cv_b_lin["test_r2"]),
        np.std(cv_b_int["test_r2"]),
        np.std(cv_b_phy["test_r2"]),
        np.std(cv_w_lin["test_r2"]),
        np.std(cv_w_phy["test_r2"])]
colors = ["#e74c3c"] * 3 + ["#3498db"] * 2
bars = ax7.bar(names, means, yerr=stds, color=colors, alpha=0.7,
               capsize=5, edgecolor="black", linewidth=1.0)
ax7.axhline(0, color='k', lw=1, linestyle='--', alpha=0.5, label="Baseline (mean)")
ax7.axhline(icc, color='orange', lw=2, linestyle=':',
            label=f"ICC = {icc:.2f} (between-flight share)")
ax7.set_ylabel("Cross-validated R²")
ax7.set_title("Model comparison (GroupKFold)")
ax7.legend(fontsize=9, loc="upper left")
ax7.set_ylim([-1.5, 1.0])

for bar, r2 in zip(bars, means):
    y = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2,
             y + (0.05 if y > 0 else -0.1),
             f"{r2:+.2f}", ha="center", va="bottom" if y > 0 else "top", fontweight="bold")

plt.tight_layout()
outpath = BASE / "bonus_2d_comprehensive_diagnostics.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"\n  Saved diagnostic figure: {outpath}")

# ---------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"Data overview:")
print(f"  Observations: {n_obs:,}")
print(f"  Flights:      {n_flights}")
print(f"  Effective altitude range: {alt_p5:.0f}–{alt_p95:.0f} ft")
print(f"  Effective speed range:    {spd_p5:.0f}–{spd_p95:.0f} km/h")

print("\nKey findings:")
print(f"  ICC (between/total SS): {icc:.3f}")
print(f"  Between-flight best R²: {best_between_r2:+.3f}")
print(f"  Within-flight  best R²: {best_within_r2:+.3f}")

print("\nInterpretation:")
if best_between_r2 > 0.3:
    print("  Between flights: Some cross-aircraft predictability.")
else:
    print("  Between flights: No reliable cross-aircraft predictability; offsets dominate.")
if best_within_r2 > 0.2:
    print("  Within flights: Altitude/speed effects are detectable.")
else:
    print("  Within flights: Effects are weak or not generalizable on this dataset.")

print("\nCaveats:")
print(f"  Results assume FUEL_SCALE={FUEL_SCALE} is appropriate.")
print("  Speed may be ground speed if not explicitly TAS/Mach.")
print("  ISA density is assumed; real atmosphere may differ.")
print("  Models are valid within the effective range only.")

print("\n" + "=" * 80)
print("Analysis complete.")
print("=" * 80)
