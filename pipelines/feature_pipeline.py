import pandas as pd
import numpy as np
import logging

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ----------------------------
# SAFE QCUT (robust binning)
# ----------------------------
def safe_qcut(series, q, labels):
    """
    Safe version of qcut:
    - Handles small datasets (inference-safe)
    - Prevents bin/label mismatch
    """
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates="drop")
    except Exception:
        bins = min(len(series.unique()), q)
        if bins < 2:
            return pd.Series([labels[0]] * len(series), index=series.index)
        return pd.cut(series, bins=bins, labels=labels[:bins])

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting feature engineering...")

    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")

    df = df.copy()

    # ----------------------------
    # Convert numeric columns to proper types
    # ----------------------------
    numeric_cols = ["TotalCharges", "MonthlyCharges", "tenure", "SeniorCitizen"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ----------------------------
    # BASIC CLEANING (safe)
    # ----------------------------
    if "Partner" in df.columns:
        df["Partner"] = df["Partner"].fillna("No")
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].fillna("No")

    # ----------------------------
    # FEATURE CREATION
    # ----------------------------
    # Safe monthly charge per tenure
    if "TotalCharges" in df.columns and "tenure" in df.columns:
        df["AvgMonthlyCharge"] = np.where(
            df["tenure"] == 0, 0, df["TotalCharges"] / df["tenure"]
        )

    # Ratios (safe divisions)
    if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
        df["ChargesRatio"] = np.where(
            df["TotalCharges"] == 0, 0, df["MonthlyCharges"] / df["TotalCharges"]
        )

    # ----------------------------
    # CATEGORICAL FEATURES (robust)
    # ----------------------------
    if "MonthlyCharges" in df.columns:
        df["MonthlyChargeGroup"] = safe_qcut(
            df["MonthlyCharges"], q=3, labels=["Low", "Medium", "High"]
        )

    if "tenure" in df.columns:
        df["TenureGroup"] = safe_qcut(
            df["tenure"], q=3, labels=["ShortTerm", "MidTerm", "LongTerm"]
        )

    # ----------------------------
    # FINAL LOG
    # ----------------------------
    logging.info(
        "Feature engineering completed. Added features: "
        "['AvgMonthlyCharge', 'ChargesRatio', 'MonthlyChargeGroup', 'TenureGroup']"
    )
    logging.info(f"New DataFrame shape: {df.shape}")

    return df

# ----------------------------
# SAVE FEATURES (optional utility)
# ----------------------------
def save_features(df: pd.DataFrame, path: str):
    import joblib
    joblib.dump(df, path)
    logging.info(f"Features saved at {path}")

# ----------------------------
# TEST RUN
# ----------------------------
if __name__ == "__main__":
    logging.info("Running feature_engineering test...")

    try:
        df_test = pd.read_csv("data/churn.csv")
        df_fe = feature_engineering(df_test)
        logging.info(df_fe.head())
        logging.info("Feature engineering test completed successfully!")
    except Exception as e:
        logging.error(f"Error during feature engineering test: {e}")