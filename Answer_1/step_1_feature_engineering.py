# --- step_1_feature_engineering.py ---
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("../../question_1")

# Load the signals
print("Loading flight signals...")
fuel_flow = pd.read_pickle(BASE / "signals_fuel_flow.pkl")
altitude = pd.read_pickle(BASE / "signals_altitude.pkl")
wind = pd.read_pickle(BASE / "signals_wind.pkl")
speed = pd.read_pickle(BASE / "signals_vitesse.pkl")

print(f"Loaded {fuel_flow.shape[1]} flights\n")

# Alright, so each flight has different time lengths. We need to extract 
# consistent features from these time series that we can actually use for ML.
# Let's think about what makes sense for flight data...

def extract_features_from_flight(flight_id):
    """
    Pull out meaningful stats from each signal for a single flight.
    Returns a dict of features we can use for training.
    """
    features = {}
    
    # Fuel flow features - these tell us about engine performance
    ff = fuel_flow[flight_id].dropna()
    if len(ff) > 0:
        features['ff_mean'] = ff.mean()
        features['ff_std'] = ff.std()
        features['ff_max'] = ff.max()
        features['ff_min'] = ff.min()
        # Rate of change might be interesting too
        features['ff_rate_change'] = (ff.iloc[-1] - ff.iloc[0]) / len(ff) if len(ff) > 1 else 0
    
    # Altitude features - flight profile matters
    alt = altitude[flight_id].dropna()
    if len(alt) > 0:
        features['alt_mean'] = alt.mean()
        features['alt_std'] = alt.std()
        features['alt_max'] = alt.max()
        features['alt_min'] = alt.min()
        features['alt_range'] = alt.max() - alt.min()
        # Climb rate
        features['alt_climb_rate'] = (alt.iloc[-1] - alt.iloc[0]) / len(alt) if len(alt) > 1 else 0
    
    # Wind features - external conditions
    w = wind[flight_id].dropna()
    if len(w) > 0:
        features['wind_mean'] = w.mean()
        features['wind_std'] = w.std()
        features['wind_max'] = w.max()
        features['wind_min'] = w.min()
    
    # Speed features - aircraft performance
    spd = speed[flight_id].dropna()
    if len(spd) > 0:
        features['speed_mean'] = spd.mean()
        features['speed_std'] = spd.std()
        features['speed_max'] = spd.max()
        features['speed_min'] = spd.min()
        features['speed_change'] = (spd.iloc[-1] - spd.iloc[0]) / len(spd) if len(spd) > 1 else 0
    
    # Flight duration is also useful
    features['duration'] = len(ff)
    
    return features

# Extract features for all flights
print("Extracting features from time series...")
all_features = []
flight_ids = fuel_flow.columns

for i, fid in enumerate(flight_ids):
    if i % 20 == 0:  # progress check
        print(f"  Processing flight {i}/{len(flight_ids)}...")
    
    feat = extract_features_from_flight(fid)
    feat['flight_id'] = fid
    all_features.append(feat)

# Convert to dataframe
df_features = pd.DataFrame(all_features)
df_features.set_index('flight_id', inplace=True)

print(f"\nExtracted {df_features.shape[1]} features for {df_features.shape[0]} flights")
print("\nFeature columns:")
print(df_features.columns.tolist())

# Quick sanity check - look for any NaN values
nan_count = df_features.isna().sum()
if nan_count.sum() > 0:
    print("\nWarning: Found some NaN values:")
    print(nan_count[nan_count > 0])
else:
    print("\nNo missing values - good to go!")

# Basic stats to understand our features
print("\nFeature statistics:")
print(df_features.describe())

# Save the processed features
output_path = BASE / "processed_features.pkl"
df_features.to_pickle(output_path)
print(f"\nSaved features to {output_path}")

# Now let's normalize the features for ML
# Most models work better with normalized data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_normalized = pd.DataFrame(
    scaler.fit_transform(df_features),
    index=df_features.index,
    columns=df_features.columns
)

# Save normalized version too
output_norm = BASE / "processed_features_normalized.pkl"
features_normalized.to_pickle(output_norm)
print(f"Saved normalized features to {output_norm}")

print("\n" + "="*60)
print("Feature engineering complete!")
print("Ready for model training.")
print("="*60)