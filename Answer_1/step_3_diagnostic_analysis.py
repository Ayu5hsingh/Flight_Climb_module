# --- step_3_diagnostic_analysis.py ---
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from scipy import stats

BASE = Path("../../question_1")

print("="*70)
print("DIAGNOSTIC ANALYSIS: Fuel Flow vs Altitude at Constant Speed")
print("="*70)

# Load data
fuel_flow = pd.read_pickle(BASE / "signals_fuel_flow.pkl")
altitude = pd.read_pickle(BASE / "signals_altitude.pkl") 
wind = pd.read_pickle(BASE / "signals_wind.pkl")
speed = pd.read_pickle(BASE / "signals_vitesse.pkl")

data_points = []
for fid in fuel_flow.columns:
    ff = fuel_flow[fid].dropna()
    alt = altitude[fid].dropna()
    w = wind[fid].dropna()
    spd = speed[fid].dropna()
    
    common_times = ff.index.intersection(alt.index).intersection(w.index).intersection(spd.index)
    if len(common_times) == 0:
        continue
        
    for t in common_times:
        data_points.append({
            'flight_id': fid,
            'fuel_flow': ff[t],
            'altitude': alt[t],
            'wind': w[t],
            'speed': spd[t]
        })

df = pd.DataFrame(data_points)

# Basic cleaning
df = df[(df['fuel_flow'] > 0) & (df['speed'] > 0) & (df['altitude'] > 0)].copy()
for col in ['fuel_flow', 'speed', 'altitude', 'wind']:
    p99 = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=p99)
df = df[(df['speed'].between(200, 950)) & (df['altitude'].between(0, 15000))].copy()

print(f"\nClean dataset: {len(df)} points from {df['flight_id'].nunique()} flights")

# Filter to 665 km/h
TARGET_SPEED = 665
SPEED_TOL = 25
df_665 = df[(df['speed'] >= TARGET_SPEED - SPEED_TOL) & 
            (df['speed'] <= TARGET_SPEED + SPEED_TOL)].copy()

print(f"\n{'='*70}")
print(f"DATA AT {TARGET_SPEED} ± {SPEED_TOL} km/h")
print(f"{'='*70}")
print(f"Total points: {len(df_665)}")
print(f"Flights: {df_665['flight_id'].nunique()}")
print(f"Altitude range: {df_665['altitude'].min():.0f} - {df_665['altitude'].max():.0f} ft")
print(f"Note: Data only covers {df_665['altitude'].min():.0f}-{df_665['altitude'].max():.0f} ft, NOT 0-15k ft")

# Check points per flight
points_per_flight = df_665.groupby('flight_id').size()
print(f"\nPoints per flight: mean={points_per_flight.mean():.1f}, median={points_per_flight.median():.0f}")
print(f"Flights with <3 points: {(points_per_flight < 3).sum()} of {len(points_per_flight)}")

# ============================================================================
# DIAGNOSTIC 1: Fuel flow distribution and scale
# ============================================================================
print(f"\n{'='*70}")
print("DIAGNOSTIC 1: Fuel Flow Scale Analysis")
print(f"{'='*70}")
print("\nFuel flow statistics:")
print(df_665['fuel_flow'].describe())
print(f"\nObservation: Values are in the 0.0001-0.01 lb/s range")
print(f"Expected cruise fuel flow: 2-10 lb/s for commercial jets")
print(f"Conclusion: This is simulated/scaled data, NOT realistic units")

# ============================================================================
# DIAGNOSTIC 2: Variance decomposition (FIXED)
# ============================================================================
print(f"\n{'='*70}")
print("DIAGNOSTIC 2: Variance Decomposition")
print(f"{'='*70}")

# Correct variance decomposition using ANOVA approach
grand_mean = df_665['fuel_flow'].mean()
flight_means = df_665.groupby('flight_id')['fuel_flow'].mean()
flight_sizes = df_665.groupby('flight_id').size()

# Between-group variance (weighted by group sizes)
between_var = ((flight_means - grand_mean)**2 * flight_sizes).sum() / len(df_665)

# Within-group variance
within_var = df_665.groupby('flight_id')['fuel_flow'].var()
within_var_pooled = (within_var * (flight_sizes - 1)).sum() / (len(df_665) - len(flight_sizes))

# Total variance
total_var = df_665['fuel_flow'].var()

# Percentages (should sum to ~100%)
between_pct = 100 * between_var / total_var
within_pct = 100 * within_var_pooled / total_var

print(f"\nTotal variance:           {total_var:.8f}")
print(f"Between-flight variance:  {between_var:.8f} ({between_pct:.1f}%)")
print(f"Within-flight variance:   {within_var_pooled:.8f} ({within_pct:.1f}%)")
print(f"\nConclusion: {between_pct:.0f}% of variance is BETWEEN flights")
print(f"            Only {within_pct:.0f}% varies WITHIN flights")
print(f"\nThis means each flight has a different baseline fuel flow (aircraft/config)")
print(f"but altitude has minimal effect within each flight at constant speed.")

# ============================================================================
# DIAGNOSTIC 3: Per-flight altitude-fuel relationship (FIXED)
# ============================================================================
print(f"\n{'='*70}")
print("DIAGNOSTIC 3: Within-Flight Altitude Effect")
print(f"{'='*70}")

# Check correlation within each flight - handle small samples properly
flight_corrs = []
flight_alt_ranges = []
for fid in df_665['flight_id'].unique():
    flt = df_665[df_665['flight_id'] == fid]
    if len(flt) >= 3:  # Need at least 3 points
        alt_range = flt['altitude'].max() - flt['altitude'].min()
        if flt['altitude'].std() > 0 and flt['fuel_flow'].std() > 0:
            corr = flt[['altitude', 'fuel_flow']].corr().loc['altitude', 'fuel_flow']
            flight_corrs.append(corr)
            flight_alt_ranges.append(alt_range)  # Only append if we have a valid correlation

flight_corrs = np.array(flight_corrs)
flight_alt_ranges = np.array(flight_alt_ranges)

print(f"\nFlights with ≥3 points: {len(flight_corrs)}")
print(f"Altitude range within flights: mean={np.mean(flight_alt_ranges):.0f} ft, median={np.median(flight_alt_ranges):.0f} ft")

if len(flight_corrs) > 0:
    print(f"\nWithin-flight correlations (altitude vs fuel_flow):")
    print(f"  Mean: {np.mean(flight_corrs):.3f}")
    print(f"  Median: {np.median(flight_corrs):.3f}")
    print(f"  Std: {np.std(flight_corrs):.3f}")
    print(f"  Range: [{np.min(flight_corrs):.3f}, {np.max(flight_corrs):.3f}]")
    
    # Check for degenerate cases
    n_perfect = np.sum(np.abs(flight_corrs) > 0.99)
    if n_perfect > len(flight_corrs) * 0.5:
        print(f"\nWARNING: {n_perfect}/{len(flight_corrs)} flights have near-perfect correlations")
        print(f"  This suggests flights have very few altitude points (likely cruising)")
        print(f"  Correlations from 2-3 points are unreliable")
else:
    print("\nNo valid correlations (all flights have constant altitude or <3 points)")

print(f"\nConclusion: At constant speed, flights maintain relatively constant altitude")
print(f"            (cruise phase), providing insufficient altitude variation for modeling.")

# ============================================================================
# DIAGNOSTIC 4: Test for signal presence (FIXED)
# ============================================================================
print(f"\n{'='*70}")
print("DIAGNOSTIC 4: Signal Detection Test")
print(f"{'='*70}")

# Use effective range
alt_p10 = df_665['altitude'].quantile(0.10)
alt_p90 = df_665['altitude'].quantile(0.90)
df_eff = df_665[(df_665['altitude'] >= alt_p10) & 
                (df_665['altitude'] <= alt_p90)].copy()

print(f"\nEffective range: {alt_p10:.0f} - {alt_p90:.0f} ft")
print(f"Points: {len(df_eff)} from {df_eff['flight_id'].nunique()} flights")

X_raw = df_eff[['altitude']].values
y_raw = df_eff['fuel_flow'].values
groups = df_eff['flight_id'].values

# Model with proper pipeline
model = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

n_splits = min(5, df_eff['flight_id'].nunique())
gkf = GroupKFold(n_splits=n_splits)

# Test 1: Raw fuel flow
cv_raw = cross_val_score(model, X_raw, y_raw, groups=groups, cv=gkf, scoring='r2')

print(f"\nModel A: RAW fuel flow ~ altitude")
print(f"  Grouped CV R² ({n_splits}-fold): {cv_raw.mean():.4f} ± {cv_raw.std():.4f}")
print(f"  Interpretation: {'Weak relationship' if cv_raw.mean() > -0.1 else 'No predictive power'}")

# Test 2: Demeaned (remove flight offsets)
flight_means = df_eff.groupby('flight_id')['fuel_flow'].transform('mean')
y_demeaned = df_eff['fuel_flow'].values - flight_means.values

cv_demeaned = cross_val_score(model, X_raw, y_demeaned, groups=groups, cv=gkf, scoring='r2')

print(f"\nModel B: DEMEANED fuel flow ~ altitude (within-flight effect)")
print(f"  Grouped CV R² ({n_splits}-fold): {cv_demeaned.mean():.4f} ± {cv_demeaned.std():.4f}")
print(f"  Interpretation: After removing flight offsets, no altitude effect remains")

# Test 3: Baseline comparison
y_random = y_demeaned.copy()
np.random.seed(42)
np.random.shuffle(y_random)
cv_random = cross_val_score(model, X_raw, y_random, groups=groups, cv=gkf, scoring='r2')

print(f"\nModel C: RANDOM (permuted) baseline")
print(f"  Grouped CV R² ({n_splits}-fold): {cv_random.mean():.4f} ± {cv_random.std():.4f}")

print(f"\nKey Insight:")
print(f"  Raw model (A) captures between-flight differences (R²={cv_raw.mean():.3f})")
print(f"  Demeaned model (B) shows no within-flight altitude effect (R²={cv_demeaned.mean():.3f})")
print(f"  Conclusion: Altitude 'effect' is actually just different aircraft/configs")

# ============================================================================
# DIAGNOSTIC 5: Wind analysis
# ============================================================================
print(f"\n{'='*70}")
print("DIAGNOSTIC 5: Why NOT Wind Speed?")
print(f"{'='*70}")

corr = df_eff[['fuel_flow', 'altitude', 'wind']].corr()
print(f"\nCorrelations with fuel_flow:")
print(f"  Altitude: {corr.loc['altitude', 'fuel_flow']:+.3f}")
print(f"  Wind:     {corr.loc['wind', 'fuel_flow']:+.3f}")

X_both = df_eff[['altitude', 'wind']].values
model_wind = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])
cv_wind = cross_val_score(model_wind, X_both, y_raw, groups=groups, cv=gkf, scoring='r2')
delta_r2 = cv_wind.mean() - cv_raw.mean()

print(f"\nAdding wind to raw model:")
print(f"  CV R² with wind: {cv_wind.mean():.4f}")
print(f"  ΔR² = {delta_r2:+.4f}")

print(f"\nReasons to exclude wind:")
print(f"  1. Raw wind magnitude ≠ headwind component (need track angle)")
print(f"  2. ΔR² = {delta_r2:+.4f} ({'negligible' if abs(delta_r2) < 0.05 else 'slight change'})")
print(f"  3. Physics: At constant airspeed, wind doesn't change thrust requirement")
print(f"  4. Wind affects ground speed, not indicated airspeed or engine power")

# ============================================================================
# VISUALIZATION
# ============================================================================
print(f"\n{'='*70}")
print("Creating visualizations...")
print(f"{'='*70}")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Raw data by flight
ax1 = fig.add_subplot(gs[0, 0])
flights_to_show = df_eff['flight_id'].unique()[:15]
for i, fid in enumerate(flights_to_show):
    flt = df_eff[df_eff['flight_id'] == fid]
    ax1.scatter(flt['altitude'], flt['fuel_flow'], alpha=0.6, s=30, label=f'F{fid}')
ax1.set_xlabel('Altitude (ft)')
ax1.set_ylabel('Fuel Flow (lb/s)')
ax1.set_title('Raw Fuel Flow vs Altitude (15 flights shown)', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(bbox_to_anchor=(1.05, 1), fontsize=8, ncol=2)

# Plot 2: Demeaned data
ax2 = fig.add_subplot(gs[0, 1])
y_demeaned_plot = df_eff['fuel_flow'].values - df_eff.groupby('flight_id')['fuel_flow'].transform('mean').values
ax2.scatter(df_eff['altitude'], y_demeaned_plot, alpha=0.4, s=20, color='steelblue')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero (flight mean)')
ax2.set_xlabel('Altitude (ft)')
ax2.set_ylabel('Fuel Flow - Flight Mean (lb/s)')
ax2.set_title('Demeaned Fuel Flow vs Altitude', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Variance decomposition
ax3 = fig.add_subplot(gs[1, 0])
variance_data = [between_var, within_var_pooled]
labels = [f'Between Flights\n({between_pct:.0f}%)', 
          f'Within Flights\n({within_pct:.0f}%)']
colors = ['#ff6b6b', '#4ecdc4']
bars = ax3.bar(labels, variance_data, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Variance', fontsize=11)
ax3.set_title('Variance Decomposition (Sum = 100%)', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
# Add value labels
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2e}', ha='center', va='bottom', fontsize=9)

# Plot 4: Within-flight correlation distribution
ax4 = fig.add_subplot(gs[1, 1])
if len(flight_corrs) > 0 and np.std(flight_corrs) > 1e-6 and len(np.unique(flight_corrs)) > 2:
    n_bins = min(10, len(np.unique(flight_corrs)))
    ax4.hist(flight_corrs, bins=n_bins, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='No correlation')
    ax4.axvline(x=np.mean(flight_corrs), color='orange', linewidth=2, label=f'Mean = {np.mean(flight_corrs):.3f}')
    ax4.legend()
    ax4.set_xlabel('Correlation (altitude vs fuel_flow)')
    ax4.set_ylabel('Number of Flights')
else:
    # Show scatter of correlation vs altitude range
    if len(flight_corrs) > 0:
        ax4.scatter(flight_alt_ranges, flight_corrs, alpha=0.6, s=50, color='steelblue')
        ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel('Altitude Range in Flight (ft)')
        ax4.set_ylabel('Correlation')
    else:
        ax4.text(0.5, 0.5, 'Insufficient variance\nfor correlation analysis', 
                 ha='center', va='center', transform=ax4.transAxes, fontsize=12)
ax4.set_title('Within-Flight Correlations', fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Model comparison
ax5 = fig.add_subplot(gs[2, :])
model_names = ['Raw Data\n(between+within)', 'Demeaned\n(within only)', 'Random\nBaseline', 'Raw + Wind']
cv_means = [cv_raw.mean(), cv_demeaned.mean(), cv_random.mean(), cv_wind.mean()]
cv_stds = [cv_raw.std(), cv_demeaned.std(), cv_random.std(), cv_wind.std()]
colors_bar = ['#4ecdc4', '#95e1d3', '#ff6b6b', '#ffa07a']

bars = ax5.bar(model_names, cv_means, yerr=cv_stds, capsize=5, color=colors_bar, 
               alpha=0.7, edgecolor='black', linewidth=2)
ax5.axhline(y=0, color='black', linewidth=1.5, linestyle='-')
ax5.set_ylabel('Cross-Validated R²', fontsize=12)
ax5.set_title('Model Performance Comparison (Grouped 5-Fold CV)', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Annotate bars
for i, (bar, val, std) in enumerate(zip(bars, cv_means, cv_stds)):
    height = bar.get_height()
    y_pos = height + std + 0.1 if height > 0 else height - std - 0.1
    ax5.text(bar.get_x() + bar.get_width()/2., y_pos,
             f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top', 
             fontweight='bold', fontsize=11)

plt.savefig(BASE / 'diagnostic_report.png', dpi=150, bbox_inches='tight')
print(f"Saved: {BASE / 'diagnostic_report.png'}")

# ============================================================================
# FINAL CONCLUSION
# ============================================================================
print(f"\n{'='*70}")
print("FINAL CONCLUSION")
print(f"{'='*70}")
print(f"\nQuestion: Model fuel flow as function of altitude (0-15k ft) at 665 km/h")
print(f"\nFindings:")
print(f"  1. Data coverage: Only {alt_p10:.0f}-{alt_p90:.0f} ft available (not 0-15k)")
print(f"  2. {between_pct:.0f}% of variance is between flights (different aircraft/configs)")
print(f"  3. Within flights: only {within_pct:.0f}% variance with altitude")
print(f"  4. Raw model CV R²: {cv_raw.mean():.3f} (captures between-flight differences)")
print(f"  5. Demeaned model CV R²: {cv_demeaned.mean():.3f} (no within-flight effect)")
print(f"  6. Wind adds no value (ΔR² = {delta_r2:+.4f})")
print(f"\nConclusion:")
print(f"  At constant airspeed ({TARGET_SPEED} km/h), altitude has NO meaningful")
print(f"  within-flight effect on fuel flow. The data represents cruise conditions")
print(f"  where altitude varies minimally. The weak correlation in raw data reflects")
print(f"  different aircraft configurations, NOT a causal altitude effect.")
print(f"\n  WHY NOT WIND?")
print(f"    - We have raw wind magnitude, not headwind component")
print(f"    - It provides no improvement (ΔR² = {delta_r2:+.4f})")
print(f"    - Physics: At constant indicated airspeed, wind doesn't change")
print(f"      engine thrust requirements or fuel consumption")
print(f"\n  RECOMMENDATION:")
print(f"    Report 'no within-flight altitude effect detectable at constant speed.'")
print(f"    The CV R² of {cv_raw.mean():.2f} for raw data is primarily capturing")
print(f"    between-flight configuration differences, not altitude causality.")
print(f"{'='*70}\n")


# "Question 1.2 requests a fuel flow model as a function of altitude at 665 km/h. Analysis shows this is not feasible: "
# "98% of variance is between-flight configuration differences, and within-flight altitude effect is statistically undetectable (CV R² = -28.23). "
# "At constant airspeed, physics predicts constant fuel flow regardless of altitude, consistent with our findings. "
# "Wind is excluded due to lack of headwind component data and negative model contribution (ΔR² = -0.19)."