from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter("churn_api_requests_total", "Total API requests processed")
ERROR_COUNT = Counter("churn_api_errors_total", "Total API errors encountered")
REQUEST_LATENCY = Histogram("churn_api_latency_seconds", "Request processing latency in seconds")
MODEL_ACCURACY = Gauge("churn_model_accuracy", "Current model accuracy")
DRIFT_SCORE = Gauge("churn_model_drift_score", "Current model drift score")