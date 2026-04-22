import subprocess

def update_traffic(decision):

    if decision == "promote_v2":
        print("🚀 Applying Istio traffic shift...")

        cmd = """
        kubectl patch virtualservice churn-ab \
        --type='json' \
        -p='[
          {"op": "replace", "path": "/spec/http/0/route/0/weight", "value": 10},
          {"op": "replace", "path": "/spec/http/0/route/1/weight", "value": 90}
        ]'
        """

    else:
        print("🧊 Resetting to balanced traffic...")

        cmd = """
        kubectl patch virtualservice churn-ab \
        --type='json' \
        -p='[
          {"op": "replace", "path": "/spec/http/0/route/0/weight", "value": 50},
          {"op": "replace", "path": "/spec/http/0/route/1/weight", "value": 50}
        ]'
        """

    subprocess.run(cmd, shell=True)