import numpy as np


class Perceptron:
    def __init__(self, number_of_weights):
        self.weights = np.random.uniform(low=-1, high=1, size=(number_of_weights,))
        self.theta = 0

    def predict_classification(self, row):
        dot_product = 0
        for i in range(len(row) - 1):
            dot_product += self.weights[i] * row[i]
        activation = dot_product - self.theta
        return self.unit_step_function(activation)

    def unit_step_function(self, activation):
        return 1 if activation >= 0 else 0
