#!/bin/bash
set -e

trap 'echo "❌ Error occurred at $(date)"' ERR

LOCKFILE="/tmp/churn_mlops.lock"

# =========================
# Prevent parallel runs
# =========================
if [ -f "$LOCKFILE" ]; then
    echo "⚠️ Churn model retraining already running. Exiting."
    exit 1
fi

touch "$LOCKFILE"
trap "rm -f $LOCKFILE" EXIT

echo "🚀 Starting churn model retraining at $(date)"

# =========================
# Environment
# =========================
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export PATH=/usr/bin:/bin:/usr/local/bin

# =========================
# Check MLflow server
# =========================
if ! curl -s http://127.0.0.1:5000 >/dev/null; then
    echo "❌ MLflow server is not running!"
    exit 1
fi

# =========================
# Go to project directory
# =========================
cd /home/zaina/churn_mlops_project || exit 1

# =========================
# Pull latest code
# =========================
/usr/bin/git pull || { echo "❌ Git pull failed"; exit 1; }

# =========================
# Activate virtual environment
# =========================
source /home/zaina/churn_mlops_project/.venv/bin/activate

# =========================
# Pull latest data (DVC)
# =========================
dvc pull || { echo "❌ DVC pull failed"; exit 1; }

# =========================
# Train model (uses data/churn.csv)
# =========================
python -m src.train --data data/churn.csv || { echo "❌ Training failed"; exit 1; }

echo "✅ Churn model retraining completed at $(date)"
