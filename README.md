# House Price Prediction – Final Project

## Deskripsi Proyek
Proyek ini merupakan implementasi end-to-end Machine Learning System
untuk memprediksi harga rumah menggunakan dataset House Price.
Proyek mencakup eksperimen model, MLflow tracking, model serving,
containerization, serta monitoring dengan Prometheus dan Grafana.

---

## Dataset
Dataset yang digunakan adalah `train.csv` yang berisi fitur properti
seperti luas bangunan, jumlah kamar, dan variabel lainnya.

Lokasi dataset: data/train.csv

---

## Eksperimen & Model
Eksperimen dilakukan menggunakan:
- Linear Regression
- Random Forest Regressor

Tracking eksperimen menggunakan **MLflow** dengan metrik:
- RMSE
- R² Score

Notebook eksperimen: notebooks/house_price.ipynb

Model terbaik disimpan di: models/house_price_model.pkl

---

## Model Serving
Model disajikan menggunakan **FastAPI** dengan endpoint:
- `POST /predict` → prediksi harga rumah
- `GET /metrics` → metrik Prometheus

Menjalankan API secara lokal:
```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api

Swagger UI: http://localhost:8000/docs

## Monitoring & Logging

Monitoring dilakukan menggunakan:
- Prometheus
- Grafana

Metrik yang dimonitor:
- predict_requests_total
- predict_request_latency_seconds
- python_gc_collections_total

Endpoint metrics: http://localhost:8000/metrics
Prometheus UI: http://localhost:9090
Grafana Dashboard: http://localhost:3000