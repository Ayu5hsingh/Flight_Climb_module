# --- step_2_fuel_flow_model.py ---
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

BASE = Path("../../question_1")

print("Loading flight data...")
fuel_flow = pd.read_pickle(BASE / "signals_fuel_flow.pkl")
altitude = pd.read_pickle(BASE / "signals_altitude.pkl") 
wind = pd.read_pickle(BASE / "signals_wind.pkl")
speed = pd.read_pickle(BASE / "signals_vitesse.pkl")

# Combine all signals into one dataset
print("Combining signals across all flights...")

data_points = []
flight_ids = fuel_flow.columns

for fid in flight_ids:
    ff = fuel_flow[fid].dropna()
    alt = altitude[fid].dropna()
    w = wind[fid].dropna()
    spd = speed[fid].dropna()
    
    # Only keep times where we have ALL measurements
    common_times = ff.index.intersection(alt.index).intersection(w.index).intersection(spd.index)
    
    if len(common_times) == 0:
        continue
        
    for t in common_times:
        data_points.append({
            'flight_id': fid,
            'time': t,
            'fuel_flow': ff[t],
            'altitude': alt[t],
            'wind': w[t],
            'speed': spd[t]
        })

df = pd.DataFrame(data_points)
print(f"Combined {len(df)} data points from {len(flight_ids)} flights\n")

# ============================================================================
# STEP 1: Data Sanitization
# ============================================================================
print("STEP 1: Sanitizing data")
print("="*60)

initial_count = len(df)

# Drop non-physical values
df = df[(df['fuel_flow'] > 0) & 
        (df['speed'] > 0) & 
        (df['altitude'] > 0)].copy()

print(f"Removed {initial_count - len(df)} non-physical values (<=0)")

# Clip speeds to plausible flight envelope
# For commercial aircraft: roughly 200-950 km/h
SPEED_MIN, SPEED_MAX = 200, 950
before_clip = len(df)
df = df[(df['speed'] >= SPEED_MIN) & (df['speed'] <= SPEED_MAX)].copy()
print(f"Clipped to plausible speed range [{SPEED_MIN}-{SPEED_MAX} km/h]: removed {before_clip - len(df)} outliers")

print(f"Clean dataset: {len(df)} points remaining\n")

# ============================================================================
# STEP 2: Filter for constant altitude (8000 ft)
# ============================================================================
print("STEP 2: Filtering for 8000 ft altitude band")
print("="*60)

TARGET_ALT = 8000
ALT_TOLERANCE = 500

df_8k = df[(df['altitude'] >= TARGET_ALT - ALT_TOLERANCE) & 
           (df['altitude'] <= TARGET_ALT + ALT_TOLERANCE)].copy()

print(f"Points at {TARGET_ALT} ± {ALT_TOLERANCE} ft: {len(df_8k)}")
print(f"  Flights represented: {df_8k['flight_id'].nunique()}")

# ============================================================================
# STEP 3: Define "effective speed range" (P10-P90)
# ============================================================================
print("\nSTEP 3: Defining effective speed range")
print("="*60)

speed_p10 = df_8k['speed'].quantile(0.10)
speed_p90 = df_8k['speed'].quantile(0.90)

print(f"Full speed range at 8k ft: {df_8k['speed'].min():.0f} - {df_8k['speed'].max():.0f} km/h")
print(f"Effective range (P10-P90): {speed_p10:.0f} - {speed_p90:.0f} km/h")

# Filter to effective range
df_effective = df_8k[(df_8k['speed'] >= speed_p10) & 
                      (df_8k['speed'] <= speed_p90)].copy()
print(f"Points in effective range: {len(df_effective)}\n")

# ============================================================================
# STEP 4: Aggregate by flight to reduce autocorrelation
# ============================================================================
print("STEP 4: Aggregating by flight")
print("="*60)

# Calculate median values per flight in the effective range
flight_agg = df_effective.groupby('flight_id').agg({
    'fuel_flow': 'median',
    'speed': 'median',
    'wind': 'median',
    'altitude': 'median'
}).reset_index()

print(f"Aggregated to {len(flight_agg)} flight-level observations")
print(f"This reduces autocorrelation from time-series data\n")

# ============================================================================
# STEP 5: Correlation analysis - why not wind?
# ============================================================================
print("STEP 5: Correlation Analysis")
print("="*60)

corr = flight_agg[['fuel_flow', 'speed', 'wind', 'altitude']].corr()
print("Correlations with fuel_flow:")
print(f"  Speed:    {corr.loc['speed', 'fuel_flow']:+.3f}")
print(f"  Wind:     {corr.loc['wind', 'fuel_flow']:+.3f}")
print(f"  Altitude: {corr.loc['altitude', 'fuel_flow']:+.3f}")
print()

# ============================================================================
# STEP 6: Build models
# ============================================================================
print("STEP 6: Building Fuel Flow Models")
print("="*60)

X_speed = flight_agg[['speed']].values
X_wind = flight_agg[['wind']].values
X_both = flight_agg[['speed', 'wind']].values
y = flight_agg['fuel_flow'].values

# Model 1: Linear regression (speed only)
model_linear = LinearRegression()
model_linear.fit(X_speed, y)
y_pred_linear = model_linear.predict(X_speed)
r2_linear = r2_score(y, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))

print("\nModel 1: Linear Regression (Speed only)")
print(f"  fuel_flow = {model_linear.intercept_:.4f} + {model_linear.coef_[0]:.6f} * speed")
print(f"  R² = {r2_linear:.4f}")
print(f"  RMSE = {rmse_linear:.4f} lb/s")

# Model 2: Add wind to see if it improves
model_with_wind = LinearRegression()
model_with_wind.fit(X_both, y)
y_pred_wind = model_with_wind.predict(X_both)
r2_wind = r2_score(y, y_pred_wind)
delta_r2 = r2_wind - r2_linear

print("\nModel 2: Linear Regression (Speed + Wind)")
print(f"  fuel_flow = {model_with_wind.intercept_:.4f} + {model_with_wind.coef_[0]:.6f} * speed + {model_with_wind.coef_[1]:.6f} * wind")
print(f"  R² = {r2_wind:.4f}")
print(f"  ΔR² = {delta_r2:+.4f}  {'✓ Improvement' if delta_r2 > 0.01 else '✗ Negligible improvement'}")

# Model 3: Robust regression (handles outliers better)
model_robust = HuberRegressor()
model_robust.fit(X_speed, y)
y_pred_robust = model_robust.predict(X_speed)
r2_robust = r2_score(y, y_pred_robust)

print("\nModel 3: Robust Regression (Huber, Speed only)")
print(f"  fuel_flow = {model_robust.intercept_:.4f} + {model_robust.coef_[0]:.6f} * speed")
print(f"  R² = {r2_robust:.4f}")

# Model 4: Quadratic (check for curvature)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_speed)
model_quad = LinearRegression()
model_quad.fit(X_poly, y)
y_pred_quad = model_quad.predict(X_poly)
r2_quad = r2_score(y, y_pred_quad)
delta_r2_quad = r2_quad - r2_linear

print("\nModel 4: Quadratic Regression (Speed + Speed²)")
print(f"  fuel_flow = {model_quad.intercept_:.4f} + {model_quad.coef_[0]:.6f} * speed + {model_quad.coef_[1]:.8f} * speed²")
print(f"  R² = {r2_quad:.4f}")
print(f"  ΔR² = {delta_r2_quad:+.4f}  {'✓ Curvature detected' if delta_r2_quad > 0.01 else '✗ Linear is sufficient'}")

# ============================================================================
# STEP 7: Visualization
# ============================================================================
print("\nSTEP 7: Creating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Main model comparison
ax1 = axes[0]
speed_range = np.linspace(flight_agg['speed'].min(), flight_agg['speed'].max(), 100).reshape(-1, 1)

ax1.scatter(flight_agg['speed'], flight_agg['fuel_flow'], alpha=0.6, s=50, 
            label='Flight medians', color='steelblue')
ax1.plot(speed_range, model_linear.predict(speed_range), 
         'r-', linewidth=2, label=f'Linear (R²={r2_linear:.3f})')
ax1.plot(speed_range, model_quad.predict(poly.transform(speed_range)), 
         'g--', linewidth=2, label=f'Quadratic (R²={r2_quad:.3f})')
ax1.set_xlabel('Speed (km/h)', fontsize=11)
ax1.set_ylabel('Fuel Flow (lb/s)', fontsize=11)
ax1.set_title(f'Fuel Flow vs Speed at {TARGET_ALT} ft', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
ax2 = axes[1]
residuals = y - y_pred_linear
ax2.scatter(flight_agg['speed'], residuals, alpha=0.6, s=50, color='steelblue')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('Speed (km/h)', fontsize=11)
ax2.set_ylabel('Residuals (lb/s)', fontsize=11)
ax2.set_title('Residual Plot (Linear Model)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(BASE / 'fuel_flow_analysis.png', dpi=120)
print(f"Saved plot to {BASE / 'fuel_flow_analysis.png'}")

# ============================================================================
# Summary and Answer
# ============================================================================
print("\n" + "="*60)
print("SUMMARY: Why NOT wind speed?")
print("="*60)
print(f"1. Correlation: Speed={corr.loc['speed', 'fuel_flow']:+.3f} vs Wind={corr.loc['wind', 'fuel_flow']:+.3f}")
print(f"2. Model improvement with wind: ΔR² = {delta_r2:+.4f} (negligible)")
print(f"3. Physics: Fuel burn depends on engine thrust (airspeed),")
print(f"   not wind, which only affects ground speed")
print(f"\nBest model: {'Quadratic' if delta_r2_quad > 0.01 else 'Linear'} with R² = {max(r2_linear, r2_quad):.4f}")
print("="*60)