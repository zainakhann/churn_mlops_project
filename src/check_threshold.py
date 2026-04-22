with open("metrics.txt", "r") as f:
    acc = float(f.read())

THRESHOLD = 0.80

if acc < THRESHOLD:
    raise Exception("Model failed quality gate")
else:
    print("Model passed")