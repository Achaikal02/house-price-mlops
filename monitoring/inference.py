from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import time

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from monitoring.prometheus_exporter import (
    REQUEST_COUNT,
    REQUEST_LATENCY
)

app = FastAPI()

# Load model
model = joblib.load("Membangun_model/house_price_model.pkl")

class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    REQUEST_COUNT.inc()

    start_time = time.time()
    prediction = model.predict([data.features])
    REQUEST_LATENCY.observe(time.time() - start_time)

    return {"prediction": prediction.tolist()}

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
