"""
Personal Expense Forecasting and Budget Optimization
Feature Engineering Script (Windows Compatible)
-------------------------------------------------
This script performs:
✔ Aggregation of spending patterns by time & category
✔ Advanced time-series feature creation
✔ Normalization and encoding for model input
✔ Outputs: model-ready feature datasets
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ---------------------------------------------------------------------
# ✅ 1. PATH CONFIGURATION
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "project_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_PATH = os.path.join(PROCESSED_DIR, "processed_features_transactions.csv")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "model_ready_features.csv")

# ---------------------------------------------------------------------
# ✅ 2. LOAD PROCESSED DATA
# ---------------------------------------------------------------------
print("🔹 Loading processed dataset...")
df = pd.read_csv(INPUT_PATH)

print(f"📊 Input shape: {df.shape}")
print(f"📑 Columns: {list(df.columns)}")

# ---------------------------------------------------------------------
# ✅ 3. HANDLE MISSING VALUES & CLEANUP
# ---------------------------------------------------------------------
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# ---------------------------------------------------------------------
# ✅ 4. FEATURE ENGINEERING
# ---------------------------------------------------------------------
# Convert year_month to datetime for sorting
df["year_month"] = pd.to_datetime(df["year_month"], format="%Y-%m")

# Sort by category & date
df.sort_values(["category", "year_month"], inplace=True)

# Spending trend feature (difference from previous month)
df["monthly_change"] = df.groupby("category")["amount"].diff()

# Percentage change
df["monthly_pct_change"] = df.groupby("category")["amount"].pct_change().replace([np.inf, -np.inf], 0)

# Rolling variance (3 months)
df["rolling_var_3m"] = df.groupby("category")["amount"].transform(lambda x: x.rolling(3).var())

# Cumulative spend per category
df["cumulative_spend"] = df.groupby("category")["amount"].cumsum()

# Relative ratio of category spend to overall month spend
monthly_total = df.groupby("year_month")["amount"].transform("sum")
df["category_ratio"] = df["amount"] / monthly_total

# ---------------------------------------------------------------------
# ✅ 5. ENCODING CATEGORICAL FEATURES
# ---------------------------------------------------------------------
label_enc = LabelEncoder()
df["category_encoded"] = label_enc.fit_transform(df["category"])

# ---------------------------------------------------------------------
# ✅ 6. NORMALIZATION
# ---------------------------------------------------------------------
scaler = MinMaxScaler()
num_cols = ["amount", "monthly_amount_lag1", "rolling_mean_3m",
            "monthly_change", "monthly_pct_change", "rolling_var_3m",
            "cumulative_spend", "category_ratio"]

df[num_cols] = scaler.fit_transform(df[num_cols])

# ---------------------------------------------------------------------
# ✅ 7. FINAL CLEANUP
# ---------------------------------------------------------------------
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# ---------------------------------------------------------------------
# ✅ 8. SAVE MODEL-READY FEATURES
# ---------------------------------------------------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("\n✅ Feature Engineering Completed Successfully!")
print(f"✅ Output saved → {OUTPUT_PATH}")
print(f"📊 Final feature rows: {len(df)} | Columns: {len(df.columns)}")
print("🚀 Ready for model training!")
