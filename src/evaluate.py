import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

model = joblib.load("models/model.pkl")
df = pd.read_csv("data/churn.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

preds = model.predict(X)
acc = accuracy_score(y, preds)

print("Accuracy:", acc)

with open("metrics.txt", "w") as f:
    f.write(str(acc))