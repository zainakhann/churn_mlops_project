from .evaluator import compute_score

def decide(v1_metrics, v2_metrics):
    score_v1 = compute_score(**v1_metrics)
    score_v2 = compute_score(**v2_metrics)

    print(f"📊 V1 Score: {score_v1:.4f}")
    print(f"📊 V2 Score: {score_v2:.4f}")

    if score_v2 > score_v1:
        return "promote_v2"
    return "keep_v1"