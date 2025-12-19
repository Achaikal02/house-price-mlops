import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ======================
# CONFIG
# ======================
DATA_PATH = "../data/train.csv"
TARGET_COLUMN = "SalePrice"

# ======================
# LOAD DATA
# ======================
print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

X = df.select_dtypes(include=[np.number]).drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# TRAINING WITH MLFLOW
# ======================
mlflow.set_experiment("house-price-mlops")

with mlflow.start_run(run_name="random_forest_ci"):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # METRICS (CI SAFE)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(model, "model")

    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")
