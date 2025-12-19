# ===============================
# CLEAN CI / GITHUB ACTIONS ENV
# ===============================
import os

# HAPUS ENV YANG MEMBUAT MLFLOW ERROR DI CI
os.environ.pop("MLFLOW_RUN_ID", None)
os.environ.pop("MLFLOW_EXPERIMENT_ID", None)

# ===============================
# IMPORTS
# ===============================
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# CONFIG
# ===============================
DATA_PATH = "../data/train.csv"
TARGET_COLUMN = "SalePrice"

# ===============================
# MLFLOW CONFIG (CI SAFE)
# ===============================
# Gunakan local file tracking (AMAN UNTUK CI)
mlflow.set_tracking_uri("file:///tmp/mlruns")

# Set experiment (akan dibuat otomatis jika belum ada)
mlflow.set_experiment("house-price-mlops")

# Pastikan TIDAK ADA run aktif
if mlflow.active_run():
    mlflow.end_run()

# ===============================
# LOAD DATA
# ===============================
print("Loading dataset from:", os.path.abspath(DATA_PATH))
df = pd.read_csv(DATA_PATH)

# Ambil fitur numerik saja (baseline aman)
X = df.select_dtypes(include=[np.number]).drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# ===============================
# SPLIT DATA
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# TRAINING + LOGGING
# ===============================
with mlflow.start_run(run_name="random_forest_ci", nested=False):

    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # ===============================
    # LOG TO MLFLOW
    # ===============================
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="house_price_model"
    )

    print("Training completed")
    print(f"RMSE : {rmse}")
    print(f"R2   : {r2}")
