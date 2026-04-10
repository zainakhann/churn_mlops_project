# src/data_validation.py

import pandas as pd
import pandera.pandas as pa

# ----------------------------
# Load CSV
# ----------------------------
df = pd.read_csv("data/churn.csv")

# ----------------------------
# Convert columns to proper types
# ----------------------------
# Numeric columns
for col in ["TotalCharges", "MonthlyCharges"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0)

# Convert Churn to int (assuming original is "Yes"/"No" or "0"/"1")
if df["Churn"].dtype == object:
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

# ----------------------------
# Define schema
# ----------------------------
schema = pa.DataFrameSchema({
    "SeniorCitizen": pa.Column(int, checks=pa.Check.ge(0)),
    "tenure": pa.Column(int, checks=pa.Check.ge(0)),
    "MonthlyCharges": pa.Column(float, checks=pa.Check.ge(0)),
    "TotalCharges": pa.Column(float, checks=pa.Check.ge(0)),
    "Churn": pa.Column(int, checks=pa.Check.isin([0, 1]))
})

# ----------------------------
# Validate
# ----------------------------
validated_df = schema.validate(df)

print("✅ Data validation passed!")