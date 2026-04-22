with open("metrics.txt", "r") as f:
    line = f.readline().strip()

acc = float(line.split(":")[1])

THRESHOLD = 0.80

if acc < THRESHOLD:
    raise Exception("Model failed quality gate")
else:
    print("Model passed")