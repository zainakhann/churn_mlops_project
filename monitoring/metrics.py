from prometheus_client import Counter, Histogram

# Total requests
REQUEST_COUNT = Counter(
    "request_count",
    "Total number of requests"
)

# Error count
ERROR_COUNT = Counter(
    "error_count",
    "Total number of failed requests"
)

# Latency tracking
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds"
)