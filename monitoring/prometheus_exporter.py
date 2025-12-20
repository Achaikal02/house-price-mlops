from prometheus_client import Counter, Histogram

# Total request ke endpoint predict
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

# Waktu inferensi
REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latency of prediction requests"
)
