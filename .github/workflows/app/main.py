import os
import time
import pandas as pd
from fastapi import FastAPI
from app.model import ChurnModelV1, ChurnModelV2
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

app = FastAPI()

# ─────────────────────────────
# MODEL SELECTION (ENV BASED)
# ─────────────────────────────
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
model = ChurnModelV1() if MODEL_VERSION == "v1" else ChurnModelV2()

# ─────────────────────────────
# METRICS (PROMETHEUS)
# ─────────────────────────────
REQUEST_COUNT = Counter("requests_total", "Total requests", ["model"])
LATENCY = Histogram("request_latency_seconds", "Latency", ["model"])

# ─────────────────────────────
# PREDICTION ENDPOINT
# ─────────────────────────────
@app.post("/predict")
def predict(data: dict):
    start = time.time()

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    latency = time.time() - start

    REQUEST_COUNT.labels(MODEL_VERSION).inc()
    LATENCY.labels(MODEL_VERSION).observe(latency)

    return {
        "prediction": int(prediction),
        "model_version": MODEL_VERSION,
        "latency": latency
    }

# ─────────────────────────────
# METRICS ENDPOINT
# ─────────────────────────────
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)