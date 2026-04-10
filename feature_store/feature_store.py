import joblib
import os
import logging

def save_features(df, name="features", mlflow_run_id=None, required_columns=None):
    os.makedirs("feature_store", exist_ok=True)

    path = f"feature_store/{name}.pkl"
    joblib.dump(df, path)

    logging.info(f"Features saved at {path}")