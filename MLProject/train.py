import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# =====================
# CONFIG (PATH AMAN CI)
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "train.csv")
TARGET_COLUMN = "SalePrice"


# =====================
# LOAD DATA
# =====================
print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

X = df.select_dtypes(include=[np.number]).drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]


# =====================
# SPLIT DATA
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =====================
# TRAIN MODEL + MLFLOW
# =====================
with mlflow.start_run():
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(model, "model")

print("Training finished successfully")
