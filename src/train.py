# ===========================
# src/train.py
# ===========================

# ===========================
# Imports
# ===========================
import os
import sys
import logging
import joblib
import subprocess
import hashlib
import yaml
from datetime import datetime

# 🔥 FIX: ensure imports work in CI + local
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from feature_store.feature_store import save_features
from src.utils import load_data, split_data
from pipelines.feature_pipeline import feature_engineering
from pipelines.model_pipeline import build_model_pipeline


# ===========================
# Config Loading
# ===========================
CONFIG_PATH = "config/config.yaml"

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"{CONFIG_PATH} not found")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# ===========================
# Paths / Config
# ===========================
LOG_PATH = config["paths"].get("logs", "logs/")
MODEL_PATH = config["paths"].get("models", "models/")
FEATURE_STORE_PATH = config["features"].get("feature_store_path", "feature_store/")

TARGET_COL = config["data"].get("target_column", "Churn")
DATA_PATH = config["data"].get("raw_path", "data/churn.csv")

TEST_SIZE = config["data"].get("test_size", 0.2)
RANDOM_STATE = config["project"].get("random_state", 42)

MLFLOW_URI = config["mlflow"].get("tracking_uri")
EXPERIMENT_NAME = config["mlflow"].get("experiment_name", "churn_prediction_model")


# ===========================
# Ensure directories exist
# ===========================
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(FEATURE_STORE_PATH, exist_ok=True)


# ===========================
# Logging Setup
# ===========================
logging.basicConfig(
    filename=os.path.join(LOG_PATH, "training.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)


# ===========================
# Helpers
# ===========================
def get_git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "N/A"


def get_dvc_checksum(path: str):
    if not os.path.exists(path):
        return "N/A"
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


# ===========================
# Training Pipeline
# ===========================
def main():
    logging.info("===== TRAINING STARTED =====")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    IS_CI = os.getenv("GITHUB_ACTIONS") == "true"

    # MLflow setup
    if IS_CI:
        mlflow.set_tracking_uri("file:./mlruns")
    else:
        mlflow.set_tracking_uri(MLFLOW_URI)

    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"train_{timestamp}"):

        # ===========================
        # Load Data
        # ===========================
        df = load_data(DATA_PATH)
        if df.empty:
            raise ValueError("Dataset is empty")

        logging.info(f"Data Loaded: {df.shape}")

        # ===========================
        # Feature Engineering
        # ===========================
        df = feature_engineering(df)

        run_id = mlflow.active_run().info.run_id

        save_features(
            df,
            name="reference_features",
            mlflow_run_id=run_id,
            required_columns=df.columns.tolist()
        )

        feature_path = os.path.join(FEATURE_STORE_PATH, f"features_{timestamp}.pkl")
        joblib.dump(df, feature_path)

        logging.info(f"Features saved: {df.shape}")

        # ===========================
        # Split
        # ===========================
        X_train, X_test, y_train, y_test = split_data(
            df,
            target=TARGET_COL,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        logging.info(f"Split done: {X_train.shape}, {X_test.shape}")

        # ===========================
        # Model
        # ===========================
        pipeline: Pipeline = build_model_pipeline(X_train)

        pipeline.fit(X_train, y_train)

        # ===========================
        # Evaluation
        # ===========================
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        logging.info(f"Accuracy: {acc:.4f}")

        # ===========================
        # Save Models
        # ===========================
        timestamped_model = os.path.join(MODEL_PATH, f"model_{timestamp}.pkl")
        latest_model = os.path.join(MODEL_PATH, "model.pkl")  # 🔥 FIX FOR CI

        joblib.dump(pipeline, timestamped_model)
        joblib.dump(pipeline, latest_model)  # critical for CI evaluation

        joblib.dump(pipeline, os.path.join(MODEL_PATH, "model.pkl"))

        logging.info(f"Model saved: {latest_model}")

        # ===========================
        # MLflow logging
        # ===========================
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        mlflow.log_param("git_commit", get_git_commit_hash())

        mlflow.sklearn.log_model(pipeline, "model")

        mlflow.log_artifact(latest_model, artifact_path="model")
        mlflow.log_artifact(feature_path, artifact_path="features")

        # ===========================
        # Register Model
        # ===========================
        client = MlflowClient()

        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            "ChurnPredictionModel"
        )

        logging.info("Model registered successfully")

        # ===========================
        # Metadata
        # ===========================
        metadata = {
            "timestamp": timestamp,
            "model_path": latest_model,
            "metrics": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1
            },
            "git_commit": get_git_commit_hash(),
        }

        metadata_path = os.path.join(MODEL_PATH, f"metadata_{timestamp}.pkl")
        joblib.dump(metadata, metadata_path)

        mlflow.log_artifact(metadata_path, artifact_path="metadata")

        logging.info("===== TRAINING COMPLETED =====")


if __name__ == "__main__":
    main()