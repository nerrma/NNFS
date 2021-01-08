import numpy as np

# General class
class Accuracy: 
    def calculate(self, predictions, y):
        comparisions = self.compare(predictions, y)
        accuracy = np.mean(comparisions)
        
        return accuracy

# Accuracy of regression model
class Regression(Accuracy):
    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

# Accuracy of categorial model
class Categorial(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary

    def init(self, y):
        pass

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        return predictions == y
