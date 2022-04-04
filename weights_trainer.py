import numpy as np
from percepton import Perceptron


class WeightsTrainer:
    def __init__(self, train, alpha, number_of_epochs):
        self.train = train
        self.alpha = alpha
        self.number_of_epochs = number_of_epochs

    def train_weights(self):
        np.random.seed(0)
        perceptron = Perceptron(len(self.train) - 1)
        theta_index = 0
        for epoch in range(self.number_of_epochs):
            for row in self.train:
                prediction = perceptron.predict_classification(row)
                actual = row[-1]
                error = actual - prediction
                perceptron.weights[theta_index] += self.alpha * error
                for i in range(len(row) - 1):
                    perceptron.weights[i + 1] += self.alpha * error * row[i]
        return perceptron
