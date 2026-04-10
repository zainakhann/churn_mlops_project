# deployment/app.py

from fastapi import FastAPI
import pandas as pd
import time

# ----------------------------
# Your inference pipeline
# ----------------------------
from src.inference import predict

# ----------------------------
# Logging (Centralized)
# ----------------------------
from monitoring.logging import get_logger
logger = get_logger(__name__)

# ----------------------------
# Prometheus Metrics
# ----------------------------
from monitoring.metrics import (
    REQUEST_COUNT,
    ERROR_COUNT,
    REQUEST_LATENCY
)
from prometheus_client import start_http_server

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Churn Prediction API")

# Start Prometheus metrics server (port 8001)
start_http_server(8001)
logger.info("📡 Prometheus running at http://localhost:8001")


# ----------------------------
# API Endpoint (Instrumented)
# ----------------------------
@app.post("/predict")
async def predict_churn(data: dict):

    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        df = pd.DataFrame([data])
        logger.info(f"📥 Received input: {df.to_dict(orient='records')}")

        # Model prediction
        pred = predict(df)[0]

        logger.info(f"✅ Prediction success: {pred}")

        return {"churn_prediction": int(pred)}

    except Exception as e:
        ERROR_COUNT.inc()
        logger.exception(f"❌ Prediction failed: {str(e)}")
        return {"error": str(e)}

    finally:
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        logger.info(f"⏱ Request latency: {latency:.4f} seconds")