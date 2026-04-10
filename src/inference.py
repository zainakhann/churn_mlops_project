# src/inference.py

import os
import joblib
import pandas as pd
import logging

from pipelines.feature_pipeline import feature_engineering

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ----------------------------
# Paths
# ----------------------------
MODEL_PATH = "models"


# ----------------------------
# Load latest PIPELINE
# ----------------------------
def load_latest_pipeline():
    files = [f for f in os.listdir(MODEL_PATH) if f.startswith("pipeline_")]

    if not files:
        raise FileNotFoundError("No trained pipeline found")

    latest_file = sorted(files)[-1]
    pipeline_path = os.path.join(MODEL_PATH, latest_file)

    logging.info(f"Loading pipeline: {pipeline_path}")

    return joblib.load(pipeline_path)


# ----------------------------
# Validate RAW input (CHURN DATASET)
# ----------------------------
def validate_input(data: pd.DataFrame):
    required_cols = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges"
    ]

    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    return data[required_cols]


# ----------------------------
# Predict function
# ----------------------------
def predict(data: pd.DataFrame):
    try:
        pipeline = load_latest_pipeline()

        # Step 1: validate RAW input
        data = validate_input(data)

        # Step 2: feature engineering (same as training)
        data = feature_engineering(data)

        # Step 3: prediction
        preds = pipeline.predict(data)

        logging.info(f"Predictions generated: {preds}")

        return preds

    except Exception as e:
        logging.exception("Prediction failed")
        raise e


# ----------------------------
# Example run
# ----------------------------
if __name__ == "__main__":
    sample = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 845.5
    }])

    result = predict(sample)
    print("Prediction:", result)