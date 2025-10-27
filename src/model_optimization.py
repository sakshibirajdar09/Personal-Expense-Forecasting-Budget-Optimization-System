import optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# =====================================
# Load Processed Features
# =====================================
print("\nLoading processed features...")
df = pd.read_csv("data/processed/processed_features_transactions.csv")

# Drop irrelevant columns
drop_cols = [col for col in ['transaction_id', 'user_id', 'notes'] if col in df.columns]
df = df.drop(columns=drop_cols, errors='ignore')

# Handle datetime
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df = df.drop(columns=['date'])

# Encode categorical columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes

# Remove rows with any NaN values
df = df.dropna()

# =====================================
# Split Data
# =====================================
X = df.drop(columns=['amount'])
y = df['amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# =====================================
# LightGBM Objective
# =====================================
def objective_lightgbm(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

# =====================================
# XGBoost Objective
# =====================================
def objective_xgboost(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10),
        'random_state': 42
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

# =====================================
# LSTM Objective
# =====================================
def objective_lstm(trial):
    # Normalize data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # Convert to sequences (lookback = 5 months)
    lookback = 5
    X_seq, y_seq = [], []
    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i - lookback:i])
        y_seq.append(y_scaled[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    # Split into train/test
    split = int(len(X_seq) * 0.8)
    X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
    y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]

    # Define hyperparams
    n_units = trial.suggest_int("n_units", 32, 128)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    epochs = trial.suggest_int("epochs", 20, 50)
    batch_size = trial.suggest_int("batch_size", 16, 64)

    # Build LSTM model
    model = Sequential([
        LSTM(n_units, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), return_sequences=False),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mae')

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train_seq, y_train_seq, validation_split=0.1, epochs=epochs,
              batch_size=batch_size, verbose=0, callbacks=[early_stop])

    preds = model.predict(X_test_seq)
    mae = mean_absolute_error(y_test_seq, preds)
    return mae

# =====================================
# Run Optimizations
# =====================================
print("\n🔍 Starting Optuna tuning for LightGBM...")
study_lgb = optuna.create_study(direction="minimize")
study_lgb.optimize(objective_lightgbm, n_trials=20, show_progress_bar=True)
best_lgb_params = study_lgb.best_params
print("✅ Best LightGBM Params:", best_lgb_params)
joblib.dump(lgb.LGBMRegressor(**best_lgb_params).fit(X_train, y_train), "models/lightgbm_model.pkl")

print("\n🔍 Starting Optuna tuning for XGBoost...")
study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgboost, n_trials=20, show_progress_bar=True)
best_xgb_params = study_xgb.best_params
print("✅ Best XGBoost Params:", best_xgb_params)
joblib.dump(xgb.XGBRegressor(**best_xgb_params).fit(X_train, y_train), "models/xgboost_model.pkl")

print("\n🔍 Starting Optuna tuning for LSTM (slower)...")
study_lstm = optuna.create_study(direction="minimize")
study_lstm.optimize(objective_lstm, n_trials=10, show_progress_bar=True)
best_lstm_params = study_lstm.best_params
print("✅ Best LSTM Params:", best_lstm_params)

# Train Final LSTM Model
print("\n🚀 Training final LSTM model...")
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
lookback = 5
X_seq, y_seq = [], []
for i in range(lookback, len(X_scaled)):
    X_seq.append(X_scaled[i - lookback:i])
    y_seq.append(y_scaled[i])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)
split = int(len(X_seq) * 0.8)
X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]

model_lstm = Sequential([
    LSTM(best_lstm_params["n_units"], input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dropout(best_lstm_params["dropout"]),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mae')
model_lstm.fit(X_train_seq, y_train_seq, epochs=best_lstm_params["epochs"],
               batch_size=best_lstm_params["batch_size"], verbose=0)

os.makedirs("models", exist_ok=True)
model_lstm.save("models/lstm_model.h5")

print("\n✅ All models (LightGBM, XGBoost, LSTM) optimized and saved successfully!")
