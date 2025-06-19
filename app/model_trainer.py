import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_fetcher import get_price_data
from feature_engineer import add_features, label_data

# === Fetch & Prepare Data ===
df = get_price_data("AAPL", start="2022-01-01")
df = add_features(df)
df = label_data(df)

# === Define features and target ===
features = ["return_1d", "return_5d", "sma_10", "sma_50", "rsi_14", "volatility_10", "volume_change"]
X = df[features]
y = df["target"]

# === Train model ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Save model ===
dump(model, "trained_model.joblib")
print("✅ Model trained and saved as 'trained_model.joblib'")
