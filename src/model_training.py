"""
Personal Expense Forecasting — Unified Model Training
-----------------------------------------------------
This script trains and saves:
1️⃣ Baseline Models  (Linear Regression, ARIMA, Prophet)
2️⃣ ML Models        (Random Forest, XGBoost, LightGBM)
3️⃣ Deep Learning    (LSTM, GRU, Bi-LSTM, CNN-1D)
Each model is evaluated using MAE, RMSE, R², and MAPE.
Results are logged and saved to project_output/model_results.csv
"""

# =====================================================
# 1️⃣ Imports & Config
# =====================================================
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import Sequential, layers

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_features_transactions.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "project_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 Starting Unified Model Training...")

# =====================================================
# 2️⃣ Load Data
# =====================================================
data = pd.read_csv(DATA_PATH)
print(f"✅ Data loaded → {data.shape}")

# Handle NaN
data.fillna(0, inplace=True)

# Feature-target split
X = data.select_dtypes(include=[np.number]).drop(columns=["amount"], errors="ignore")
y = data["amount"]

# =====================================================
# 3️⃣ Utility Functions
# =====================================================
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

results = []

# =====================================================
# 4️⃣ Baseline Models
# =====================================================
print("\n📈 Training Baseline Models...")

## Linear Regression
lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)
results.append(evaluate_model("Linear Regression", y, y_pred))

## ARIMA (on total monthly amount)
try:
    ts = data.groupby("year_month")["amount"].sum()
    model = ARIMA(ts, order=(1,1,1)).fit()
    forecast = model.predict(start=0, end=len(ts)-1)
    results.append(evaluate_model("ARIMA", ts, forecast))
except Exception as e:
    print(f"⚠️ ARIMA failed: {e}")

## Prophet
try:
    prophet_df = data.groupby("year_month")["amount"].sum().reset_index()
    prophet_df.columns = ["ds", "y"]
    m = Prophet()
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=3, freq="M")
    forecast = m.predict(future)
    y_pred = forecast["yhat"][:len(prophet_df)]
    results.append(evaluate_model("Prophet", prophet_df["y"], y_pred))
except Exception as e:
    print(f"⚠️ Prophet failed: {e}")

# =====================================================
# 5️⃣ Machine Learning Models
# =====================================================
print("\n🧠 Training Machine Learning Models...")

## Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X, y)
y_pred = rf.predict(X)
results.append(evaluate_model("Random Forest", y, y_pred))
tf.keras.backend.clear_session()

## XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
xgb_model.fit(X, y)
y_pred = xgb_model.predict(X)
results.append(evaluate_model("XGBoost", y, y_pred))

## LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
lgb_model.fit(X, y)
y_pred = lgb_model.predict(X)
results.append(evaluate_model("LightGBM", y, y_pred))

# =====================================================
# 6️⃣ Deep Learning Models
# =====================================================
print("\n🤖 Training Deep Learning Models...")

X_dl = np.expand_dims(X.values, axis=1)

def train_dl_model(name, model):
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_dl, y, epochs=10, batch_size=32, verbose=0)
    y_pred = model.predict(X_dl).flatten()
    results.append(evaluate_model(name, y, y_pred))

# LSTM
model_lstm = Sequential([
    layers.LSTM(64, input_shape=(1, X.shape[1])),
    layers.Dense(1)
])
train_dl_model("LSTM", model_lstm)

# GRU
model_gru = Sequential([
    layers.GRU(64, input_shape=(1, X.shape[1])),
    layers.Dense(1)
])
train_dl_model("GRU", model_gru)

# Bi-LSTM
model_bilstm = Sequential([
    layers.Bidirectional(layers.LSTM(64), input_shape=(1, X.shape[1])),
    layers.Dense(1)
])
train_dl_model("BiLSTM", model_bilstm)

# CNN-1D
model_cnn = Sequential([
    layers.Conv1D(64, kernel_size=1, activation='relu', input_shape=(1, X.shape[1])),

    layers.Flatten(),
    layers.Dense(1)
])
train_dl_model("CNN-1D", model_cnn)

# =====================================================
# 7️⃣ Save Results & Models
# =====================================================
results_df = pd.DataFrame(results)
results_path = os.path.join(OUTPUT_DIR, "model_results.csv")
results_df.to_csv(results_path, index=False)
print("\n✅ All models trained successfully!")
print(f"📊 Results saved to: {results_path}")
print(results_df)
