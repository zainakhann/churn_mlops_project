import random

def get_model_metrics(model_name: str):
    """
    Simulated metrics (replace with Prometheus later)
    """

    return {
        "accuracy": random.uniform(0.70, 0.95),
        "latency": random.uniform(50, 200),   # ms
        "error_rate": random.uniform(0.01, 0.1)
    }