"""
Personal Expense Forecasting and Budget Optimization
Data Preprocessing Script (Windows Compatible)
-------------------------------------------------
This script performs:
✔ Date parsing and feature extraction
✔ Merchant categorization (NLP + rules)
✔ Handling missing values, duplicates, and outliers
✔ Currency normalization
✔ Feature engineering (lag, rolling stats)
✔ Output: cleaned + feature-engineered CSVs
"""

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------
# ✅ 1. PATH CONFIGURATION
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

DATA_PATH_1 = os.path.join(RAW_DIR, "budgetwise_synthetic_dirty.csv")
DATA_PATH_2 = os.path.join(RAW_DIR, "budgetwise_finance_dataset.csv")

print("🚀 Starting data preprocessing...")

# ---------------------------------------------------------------------
# ✅ 2. LOAD DATA
# ---------------------------------------------------------------------
print("🔹 Loading datasets...")
df1 = pd.read_csv(DATA_PATH_1)
df2 = pd.read_csv(DATA_PATH_2)

print(f"📊 Dataset 1 shape: {df1.shape}")
print(f"📊 Dataset 2 shape: {df2.shape}")

df = pd.concat([df1, df2], ignore_index=True)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# ---------------------------------------------------------------------
# ✅ 3. HANDLE DUPLICATES & BASIC CLEANING
# ---------------------------------------------------------------------
df.drop_duplicates(inplace=True)
df.replace(["na", "NaN", "null", "NULL", " "], np.nan, inplace=True)

# ---------------------------------------------------------------------
# ✅ 4. DATE PARSING & FEATURE EXTRACTION
# ---------------------------------------------------------------------
def parse_date(x):
    """Try to parse multiple date formats safely."""
    if pd.isna(x):
        return pd.NaT
    try:
        return pd.to_datetime(x, errors="coerce", dayfirst=True)
    except Exception:
        return pd.NaT

date_cols = [c for c in df.columns if "date" in c.lower()]
if len(date_cols) == 0:
    raise ValueError("❌ No date column found in dataset!")

df["date"] = df[date_cols[0]].apply(parse_date)
df.dropna(subset=["date"], inplace=True)

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["weekday"] = df["date"].dt.day_name()
df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"])
df["year_month"] = df["date"].dt.to_period("M").astype(str)

# ---------------------------------------------------------------------
# ✅ 5. CURRENCY NORMALIZATION
# ---------------------------------------------------------------------
def normalize_currency(val):
    """Removes ₹/$ symbols, commas, spaces; converts to float."""
    if pd.isna(val):
        return np.nan
    val = str(val).replace(",", "").replace("₹", "").replace("$", "").strip()
    val = re.sub(r"[^0-9.\-]", "", val)  # keep only numeric chars
    try:
        return float(val)
    except ValueError:
        return np.nan

amount_col = [c for c in df.columns if "amount" in c.lower()]
if len(amount_col) == 0:
    raise ValueError("❌ No amount column found in dataset!")

df["amount"] = df[amount_col[0]].apply(normalize_currency)

# Remove negative or zero amounts unless marked as refunds
df = df[df["amount"] > 0]

# ---------------------------------------------------------------------
# ✅ 6. HANDLE MISSING VALUES
# ---------------------------------------------------------------------
# Fill category & merchant
if "category" not in df.columns:
    df["category"] = np.nan
if "merchant" not in df.columns:
    df["merchant"] = np.nan

df["category"] = df["category"].fillna("unknown")
df["merchant"] = df["merchant"].fillna("unknown")

# Fill missing amount by category median
df["amount"] = df.groupby("category")["amount"].transform(
    lambda x: x.fillna(x.median())
)
df["amount"].fillna(df["amount"].median(), inplace=True)

# ---------------------------------------------------------------------
# ✅ 7. CATEGORIZE MERCHANTS → CATEGORIES
# ---------------------------------------------------------------------
def infer_category(row):
    text = str(row["merchant"]).lower()
    if any(k in text for k in ["restaurant", "food", "hotel", "eat"]):
        return "food"
    elif any(k in text for k in ["uber", "ola", "fuel", "bus", "taxi", "travel"]):
        return "transport"
    elif any(k in text for k in ["rent", "pg", "apartment", "flat"]):
        return "housing"
    elif any(k in text for k in ["amazon", "flipkart", "myntra", "shop"]):
        return "shopping"
    elif any(k in text for k in ["netflix", "prime", "cinema", "movie"]):
        return "entertainment"
    elif any(k in text for k in ["electricity", "water", "bill", "gas"]):
        return "utilities"
    elif any(k in text for k in ["hospital", "pharmacy", "doctor"]):
        return "healthcare"
    else:
        return row["category"]

df["category"] = df.apply(infer_category, axis=1)

# ---------------------------------------------------------------------
# ✅ 8. OUTLIER REMOVAL
# ---------------------------------------------------------------------
q1 = df["amount"].quantile(0.01)
q3 = df["amount"].quantile(0.99)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df = df[(df["amount"] >= lower) & (df["amount"] <= upper)]

# ---------------------------------------------------------------------
# ✅ 9. FEATURE ENGINEERING
# ---------------------------------------------------------------------
df["amount_log1p"] = np.log1p(df["amount"])

# Monthly aggregated spend by category
monthly = (
    df.groupby(["year_month", "category"])["amount"]
    .sum()
    .reset_index()
    .sort_values("year_month")
)

monthly["monthly_amount_lag1"] = monthly.groupby("category")["amount"].shift(1)
monthly["rolling_mean_3m"] = (
    monthly.groupby("category")["amount"].transform(lambda x: x.rolling(3).mean())
)

# Ensure consistent columns
monthly = monthly[["year_month", "category", "amount", "monthly_amount_lag1", "rolling_mean_3m"]]

# ---------------------------------------------------------------------
# ✅ 10. SAVE OUTPUTS
# ---------------------------------------------------------------------
interim_path = os.path.join(INTERIM_DIR, "interim_cleaned_transactions.csv")
processed_path = os.path.join(PROCESSED_DIR, "processed_features_transactions.csv")

df.to_csv(interim_path, index=False)
monthly.to_csv(processed_path, index=False)

print("\n✅ Data Preprocessing Completed Successfully!")
print(f"✅ Cleaned file saved → {interim_path}")
print(f"✅ Feature file saved → {processed_path}")
print(f"📊 Final cleaned rows: {len(df)} | Features: {len(df.columns)}")
