# src/evaluate.py

import os
import joblib
import pandas as pd

from src.utils import load_data
from pipelines.feature_pipeline import feature_engineering

# =========================
# Paths
# =========================
DATA_PATH = "data/churn.csv"
MODEL_PATH = "models/model.pkl"

TARGET_COL = "Churn"

# =========================
# Load model
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# =========================
# Load data
# =========================
df = load_data(DATA_PATH)

if df.empty:
    raise ValueError("Dataset is empty")

# =========================
# Apply SAME feature engineering as training
# =========================
df = feature_engineering(df)

# =========================
# Split features/target
# =========================
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# =========================
# Predict
# =========================
preds = model.predict(X)

# =========================
# Simple evaluation
# =========================
accuracy = (preds == y).mean()

print(f"Accuracy: {accuracy:.4f}")

# =========================
# Save metrics for CI
# =========================
with open("metrics.txt", "w") as f:
    f.write(f"accuracy:{accuracy}\n")