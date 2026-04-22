import joblib
import os
import logging
from datetime import datetime
import json
import pandas as pd
import hashlib

# ==============================
# CONFIG
# ==============================

FEATURE_DIR = "feature_store"
METADATA_FILE = os.path.join(FEATURE_DIR, "metadata.json")
MAX_VERSIONS = 10

os.makedirs(FEATURE_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------------
# SCHEMA HASH (FIXED)
# ------------------------------

def get_schema_hash(columns):
    return hashlib.md5(",".join(columns).encode()).hexdigest()

# ------------------------------
# METADATA HELPERS
# ------------------------------

def load_metadata():
    if not os.path.exists(METADATA_FILE):
        return {}
    with open(METADATA_FILE, "r") as f:
        return json.load(f)


def save_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

# ------------------------------
# VALIDATION
# ------------------------------

def validate_features(df, required_columns=None):
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"❌ Missing columns: {missing}")

    if df.isnull().sum().sum() > 0:
        logger.warning("⚠️ Null values found")

    if df.empty:
        raise ValueError("❌ Data is empty")

# ------------------------------
# SAVE FEATURES
# ------------------------------

def save_features(df, name="churn_features", mlflow_run_id=None, required_columns=None):
    validate_features(df, required_columns)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.pkl"
    path = os.path.join(FEATURE_DIR, filename)

    joblib.dump(df, path)

    metadata = load_metadata()
    metadata[filename] = {
        "timestamp": timestamp,
        "mlflow_run_id": mlflow_run_id,
        "columns": df.columns.tolist(),
        "num_rows": len(df),
        "schema_hash": get_schema_hash(df.columns.tolist())  # ✅ FIXED
    }

    save_metadata(metadata)

    logger.info(f"📦 Saved: {path}")
    return path

# ------------------------------
# LOAD LATEST
# ------------------------------

def load_latest_features(name="churn_features"):
    metadata = load_metadata()
    files = [f for f in metadata if f.startswith(name)]

    if not files:
        raise FileNotFoundError("❌ No feature files found")

    latest = sorted(files)[-1]
    path = os.path.join(FEATURE_DIR, latest)

    df = joblib.load(path)

    # Schema check
    saved_hash = metadata[latest]["schema_hash"]
    current_hash = get_schema_hash(df.columns.tolist())  # ✅ FIXED

    if saved_hash != current_hash:
        saved_columns = metadata[latest]["columns"]
        current_columns = df.columns.tolist()

        missing = list(set(saved_columns) - set(current_columns))
        extra = list(set(current_columns) - set(saved_columns))

        raise ValueError(
            f"❌ Schema mismatch!\nMissing: {missing}\nExtra: {extra}"
        )

    logger.info(f"📥 Loaded: {path}")
    return df

# ------------------------------
# LIST FEATURE VERSIONS
# ------------------------------

def list_feature_versions(name="churn_features"):
    metadata = load_metadata()
    versions = [f for f in metadata.keys() if f.startswith(name)]

    if not versions:
        raise FileNotFoundError(f"❌ No versions found for {name}")

    return sorted(versions)

# ------------------------------
# LOAD BY SPECIFIC FILE
# ------------------------------

def load_features_by_name(filename):
    path = os.path.join(FEATURE_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {filename}")

    df = joblib.load(path)
    logger.info(f"📥 Loaded specific version: {path}")
    return df

# ------------------------------
# COMMAND-LINE TEST
# ------------------------------

if __name__ == "__main__":
    print("🚀 Running Feature Store self-test...")

    df = pd.DataFrame({
        "gender": ["Male", "Female"],
        "tenure": [12, 24],
        "MonthlyCharges": [70.5, 99.9],
        "TotalCharges": [845.0, 2399.0],
        "Contract": ["Month-to-month", "Two year"],
        "PaymentMethod": ["Electronic check", "Credit card (automatic)"],
        "Churn": [0, 1]
    })

    # Save features
    path = save_features(
        df,
        name="test_features",
        required_columns=df.columns.tolist()
    )
    print(f"✅ Saved features at: {path}")

    # Load latest
    loaded = load_latest_features("test_features")
    print("✅ Loaded latest features:\n", loaded)

    # List versions
    versions = list_feature_versions("test_features")
    print("📄 All versions:", versions)

    # Load specific version
    specific = load_features_by_name(versions[-1])
    print("✅ Loaded specific version:\n", specific)

    print("🎯 Feature Store self-test completed successfully!")