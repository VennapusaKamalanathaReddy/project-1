import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
data = pd.read_csv("air_quality_health.csv")

# Features (8)
X = data[['PM2.5','PM10','NO2','SO2','O3','Temperature','Humidity','pH']]

# Target
y = data['HealthImpactClass']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y_encoded)

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(le, "label.pkl")

print("Model trained successfully")