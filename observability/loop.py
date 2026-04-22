from .metrics_client import get_model_metrics
from .controller import decide
from .istio_updater import update_traffic

def run_loop():
    print("\n🧠 Collecting metrics...\n")

    v1_metrics = get_model_metrics("v1")
    v2_metrics = get_model_metrics("v2")

    decision = decide(v1_metrics, v2_metrics)

    print(f"\n🎯 Decision: {decision}")

    update_traffic(decision)


if __name__ == "__main__":
    run_loop()