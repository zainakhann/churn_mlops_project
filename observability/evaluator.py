def compute_score(accuracy, latency, error_rate):
    return accuracy - (0.003 * latency) - (0.5 * error_rate)