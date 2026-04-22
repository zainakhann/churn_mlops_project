import numpy as np

class ChurnModelV1:
    def predict(self, X):
        return np.random.randint(0, 2, len(X))

class ChurnModelV2:
    def predict(self, X):
        return np.random.randint(0, 2, len(X))