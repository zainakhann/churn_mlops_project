import os
import joblib
import pandas as pd
from datetime import datetime

from sklearn.metrics import accuracy_score

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping

from prometheus_client import start_http_server

from pipelines.feature_pipeline import feature_engineering
from monitoring.logging import get_logger
from monitoring.metrics import MODEL_ACCURACY, DRIFT_SCORE
from feature_store.feature_store import load_latest_features

logger = get_logger(__name__)

# ==============================
# PATHS
# ==============================
REFERENCE_FEATURES = "feature_store/reference_features"
CURRENT_FEATURES = "feature_store/current_features"

REPORT_DIR = "monitoring/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

LAST_DRIFT_STATUS = False


# ==============================
# LOAD DATA (FEATURE STORE ONLY - CLEAN)
# ==============================
def load_data(feature_name):
    logger.info(f"📦 Loading features: {feature_name}")
    return load_latest_features(feature_name)


# ==============================
# SCHEMA VALIDATION
# ==============================
def validate_schema(ref, curr):
    if set(ref.columns) != set(curr.columns):
        missing_curr = set(ref.columns) - set(curr.columns)
        missing_ref = set(curr.columns) - set(ref.columns)

        logger.error(f"Missing in current: {missing_curr}")
        logger.error(f"Missing in reference: {missing_ref}")
        raise ValueError("Schema mismatch detected")

    logger.info("✅ Schema validation passed")


# ==============================
# LOAD MODEL
# ==============================
def load_latest_pipeline():
    files = [f for f in os.listdir("models") if f.startswith("pipeline_")]

    if not files:
        raise FileNotFoundError("No pipeline found")

    latest = sorted(files)[-1]
    path = os.path.join("models", latest)

    logger.info(f"📦 Loading model: {path}")
    return joblib.load(path)


# ==============================
# PREDICTIONS (SAFE + PROBABILITIES)
# ==============================
def add_predictions(df, model):
    df = df.copy()

    df = feature_engineering(df)

    X = df.drop(columns=["Churn"], errors="ignore")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        preds = (probs > 0.5).astype(int)
    else:
        preds = model.predict(X)

    df["prediction"] = preds.astype(int)

    if "Churn" in df.columns:
        df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce").fillna(0).astype(int)

    return df


# ==============================
# REPORT GENERATION
# ==============================
def generate_report(reference, current):
    logger.info("📊 Generating Evidently report...")

    reference = reference.copy()
    current = current.copy()

    for col in ["Churn", "prediction"]:
        if col in reference.columns:
            reference[col] = pd.to_numeric(reference[col], errors="coerce").fillna(0).astype(int)
        if col in current.columns:
            current[col] = pd.to_numeric(current[col], errors="coerce").fillna(0).astype(int)

    column_mapping = ColumnMapping(
        target="Churn",
        prediction="prediction"
    )

    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ])

    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    html_path = os.path.join(REPORT_DIR, f"drift_{timestamp}.html")
    json_path = os.path.join(REPORT_DIR, f"drift_{timestamp}.json")

    report.save_html(html_path)
    report.save_json(json_path)

    logger.info(f"✅ Report saved: {html_path}")

    return report


# ==============================
# METRICS EXTRACTION (FINAL FIX)
# ==============================
def extract_metrics(report, reference, current):
    try:
        data = report.as_dict()

        drift_flag = None

        # drift from Evidently
        for metric in data.get("metrics", []):
            result = metric.get("result", {})

            if "dataset_drift" in result:
                drift_flag = result["dataset_drift"]

        # accuracy computed manually (stable)
        accuracy = None

        if "Churn" in current.columns and "prediction" in current.columns:
            accuracy = accuracy_score(
                current["Churn"],
                current["prediction"]
            )

        return drift_flag, accuracy

    except Exception as e:
        logger.error(f"❌ Metric extraction failed: {e}")
        return None, None


# ==============================
# PROMETHEUS
# ==============================
def update_metrics(drift_flag, accuracy):
    if drift_flag is not None:
        DRIFT_SCORE.set(1 if drift_flag else 0)

    if accuracy is not None:
        MODEL_ACCURACY.set(float(accuracy))

    logger.info("📡 Prometheus metrics updated")


# ==============================
# DRIFT CHECK
# ==============================
def check_drift(drift_flag):
    if drift_flag:
        logger.warning("🚨 DRIFT DETECTED!")
        return True

    logger.info("✅ No drift detected")
    return False


# ==============================
# MAIN PIPELINE
# ==============================
def run_monitoring():
    logger.info("🚀 Monitoring Started")

    global LAST_DRIFT_STATUS

    reference = load_data("reference_features")
    current = load_data("current_features")

    logger.info(f"Reference shape: {reference.shape}")
    logger.info(f"Current shape: {current.shape}")

    validate_schema(reference, current)

    model = load_latest_pipeline()

    reference = add_predictions(reference, model)
    current = add_predictions(current, model)

    report = generate_report(reference, current)

    drift_flag, accuracy = extract_metrics(report, reference, current)

    logger.info(f"📉 Drift Detected: {drift_flag}")
    logger.info(f"🎯 Accuracy: {accuracy}")

    update_metrics(drift_flag, accuracy)

    LAST_DRIFT_STATUS = check_drift(drift_flag)

    if LAST_DRIFT_STATUS:
        logger.warning("⚠️ Trigger alert system")

    logger.info("✅ Monitoring Completed")

    return LAST_DRIFT_STATUS


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    start_http_server(8001)
    logger.info("📡 Prometheus running at http://localhost:8001")
    run_monitoring()