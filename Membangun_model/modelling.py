import pandas as pd
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "namadataset_preprocessing/train_preprocessed.csv"
TARGET_COLUMN = "SalePrice"
MODEL_OUTPUT = "house_price_model.pkl"
EXPERIMENT_NAME = "house-price-mlops"

print("Loading preprocessed dataset...")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COLUMN])
X = X.select_dtypes(include=[np.number])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="random_forest_model"):
    print("Training model...")

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE : {rmse}")
    print(f"R2   : {r2}")

    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(model, artifact_path="model")

    joblib.dump(model, MODEL_OUTPUT)
    print(f"Model saved as {MODEL_OUTPUT}")
