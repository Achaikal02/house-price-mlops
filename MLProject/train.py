import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# CONFIG
# =========================
DATA_PATH = "../data/train.csv"
TARGET_COLUMN = "SalePrice"

# =========================
# LOAD DATA
# =========================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Ambil fitur numerik saja (aman untuk baseline)
X = df.select_dtypes(include=[np.number]).drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# START MLFLOW RUN
# =========================
with mlflow.start_run(run_name="RandomForest-MLProject"):
    # Parameters
    n_estimators = 100
    max_depth = 10

    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # =========================
    # TRAIN MODEL
    # =========================
    print("Training model...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # =========================
    # EVALUATION
    # =========================
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # =========================
    # LOG MODEL
    # =========================
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("Training completed")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")
