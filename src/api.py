from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import time

from prometheus_client import (
    Counter,
    Histogram,
    make_asgi_app
)

predict_requests_total = Counter(
    "predict_requests_total",
    "Total number of prediction requests"
)

predict_request_latency_seconds = Histogram(
    "predict_request_latency_seconds",
    "Latency of prediction requests"
)

prediction_success_total = Counter(
    "prediction_success_total",
    "Total successful predictions"
)

app = FastAPI(
    title="House Price Prediction API",
    description="API untuk memprediksi harga rumah menggunakan model Machine Learning",
    version="1.0"
)

model = joblib.load("models/house_price_model.pkl")

class HouseFeatures(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "House Price Prediction API is running"}

@app.post("/predict")
def predict(data: HouseFeatures):
    predict_requests_total.inc() 

    start_time = time.time()

    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)[0]

    prediction_success_total.inc()  

    latency = time.time() - start_time
    predict_request_latency_seconds.observe(latency)

    return {
        "predicted_price": float(prediction)
    }

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
