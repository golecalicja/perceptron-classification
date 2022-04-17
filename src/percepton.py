import numpy as np


class Perceptron:
    def __init__(self, number_of_weights):
        self.weights = np.random.uniform(low=-1, high=1, size=(number_of_weights,))
        self.theta = 0

    def predict_classification(self, row):
        dot_product = 0
        for i in range(len(row) - 1):
            dot_product += self.weights[i] * row[i]
        return self.unit_step_function(dot_product)

    def unit_step_function(self, dot_product):
        return 1 if dot_product >= self.theta else 0
