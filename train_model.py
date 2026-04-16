"""
RoadZen - AI Model Training Script
Trains an XGBoost classifier on accident data for risk prediction.
Uses both AccidentsBig1.csv (for geo data) and combined_accident_data.csv (for features).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import os
import json

print("=" * 60)
print("🚀 RoadZen - Model Training Pipeline")
print("=" * 60)

# =============================================
# 1. Load and prepare the geospatial dataset
# =============================================
print("\n📂 Loading AccidentsBig1.csv for geospatial heatmap data...")
geo_df = pd.read_csv("../data set/AccidentsBig1.csv", nrows=50000)  # Use first 50k for speed
geo_df = geo_df.dropna(subset=['latitude', 'longitude', 'Accident_Severity'])

# Save heatmap data (lat, lng, severity) for frontend
heatmap_data = geo_df[['latitude', 'longitude', 'Accident_Severity']].copy()
heatmap_data.columns = ['lat', 'lng', 'severity']
# Normalize severity to 0-1 range for heatmap intensity
heatmap_data['intensity'] = heatmap_data['severity'].map({1: 1.0, 2: 0.6, 3: 0.3})
heatmap_sample = heatmap_data.sample(min(2000, len(heatmap_data)), random_state=42)
heatmap_sample.to_json("backend/heatmap_data.json", orient="records")
print(f"   ✅ Saved {len(heatmap_sample)} heatmap points")

# =============================================
# 2. Load combined accident data for ML model
# =============================================
print("\n📂 Loading combined_accident_data.csv for ML training...")
df = pd.read_csv("../data set/combined_accident_data.csv")
df = df.dropna()

print(f"   📊 Dataset shape: {df.shape}")
print(f"   📋 Columns: {list(df.columns)}")

# =============================================
# 3. Feature Engineering
# =============================================
print("\n⚙️  Feature Engineering...")

# Extract hour from hrmn field (format: HHMM)
df['hour'] = df['hrmn'].apply(lambda x: int(str(x).zfill(4)[:2]))

# Encode categorical features
label_encoders = {}
categorical_cols = ['weather', 'lum', 'vehicle_type', 'driver_sex', 'week_day', 'state']

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"   ✅ Encoded '{col}': {len(le.classes_)} categories")

# Map severity to numeric (0=Minor, 1=Moderate, 2=Severe)
severity_map = {'Minor': 0, 'Moderate': 1, 'Severe': 2}
df['severity_num'] = df['severity'].map(severity_map)
df = df.dropna(subset=['severity_num'])

# =============================================
# 4. Prepare features and target
# =============================================
feature_cols = [
    'hour', 'driver_age', 'engine_size', 'car_age',
    'weather_encoded', 'lum_encoded', 'vehicle_type_encoded',
    'driver_sex_encoded', 'week_day_encoded', 'state_encoded'
]

X = df[feature_cols]
y = df['severity_num'].astype(int)

print(f"\n📊 Feature matrix shape: {X.shape}")
print(f"📊 Target distribution:")
for sev, count in y.value_counts().sort_index().items():
    label = {0: 'Minor', 1: 'Moderate', 2: 'Severe'}[sev]
    print(f"   {label} ({sev}): {count} ({count/len(y)*100:.1f}%)")

# =============================================
# 5. Train/Test Split
# =============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

# =============================================
# 6. Train XGBoost Model
# =============================================
print("\n🤖 Training XGBoost Classifier...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softmax',
    num_class=3,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"   ✅ Train Accuracy: {train_acc:.4f}")
print(f"   ✅ Test Accuracy:  {test_acc:.4f}")

# =============================================
# 7. Feature Importance
# =============================================
importances = model.feature_importances_
feature_importance = sorted(
    zip(feature_cols, importances),
    key=lambda x: x[1],
    reverse=True
)
print("\n📊 Feature Importance:")
for feat, imp in feature_importance:
    bar = "█" * int(imp * 50)
    print(f"   {feat:30s} {imp:.4f} {bar}")

# Save feature importance for frontend charts
importance_data = [{"feature": f, "importance": float(i)} for f, i in feature_importance]
with open("backend/feature_importance.json", "w") as f:
    json.dump(importance_data, f)

# =============================================
# 8. Generate analytics data for frontend
# =============================================
print("\n📊 Generating analytics data...")

# Severity by hour
hourly_stats = df.groupby('hour')['severity_num'].agg(['mean', 'count']).reset_index()
hourly_stats.columns = ['hour', 'avg_severity', 'count']
hourly_stats.to_json("backend/hourly_stats.json", orient="records")

# Severity by weather
weather_stats = df.groupby('weather')['severity_num'].agg(['mean', 'count']).reset_index()
weather_stats.columns = ['weather', 'avg_severity', 'count']
weather_stats.to_json("backend/weather_stats.json", orient="records")

# Severity by vehicle type
vehicle_stats = df.groupby('vehicle_type')['severity_num'].agg(['mean', 'count']).reset_index()
vehicle_stats.columns = ['vehicle_type', 'avg_severity', 'count']
vehicle_stats.to_json("backend/vehicle_stats.json", orient="records")

# Severity by day of week
day_stats = df.groupby('week_day')['severity_num'].agg(['mean', 'count']).reset_index()
day_stats.columns = ['day', 'avg_severity', 'count']
day_stats.to_json("backend/day_stats.json", orient="records")

# Severity by state
state_stats = df.groupby('state')['severity_num'].agg(['mean', 'count']).reset_index()
state_stats.columns = ['state', 'avg_severity', 'count']
state_stats.to_json("backend/state_stats.json", orient="records")

# Casualty type distribution
casualty_stats = df.groupby('casualty_type')['severity_num'].agg(['mean', 'count']).reset_index()
casualty_stats.columns = ['casualty_type', 'avg_severity', 'count']
casualty_stats.to_json("backend/casualty_stats.json", orient="records")

# =============================================
# 9. Save model and encoders
# =============================================
print("\n💾 Saving model and encoders...")
joblib.dump(model, "backend/model.pkl")
joblib.dump(label_encoders, "backend/label_encoders.pkl")

# Save model metadata
metadata = {
    "features": feature_cols,
    "categorical_features": categorical_cols,
    "severity_map": severity_map,
    "train_accuracy": float(train_acc),
    "test_accuracy": float(test_acc),
    "n_samples": len(df),
    "feature_importance": importance_data
}
with open("backend/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"   ✅ Model saved to backend/model.pkl")
print(f"   ✅ Encoders saved to backend/label_encoders.pkl")
print(f"   ✅ Metadata saved to backend/model_metadata.json")

print("\n" + "=" * 60)
print("✅ RoadZen Model Training Complete!")
print("=" * 60)
