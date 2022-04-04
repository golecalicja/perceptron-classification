import numpy as np


class UserInputPredictor:
    def __init__(self, perceptron, input_names, size):
        self.perceptron = perceptron
        self.input_names = input_names
        self.size = size

    def predict_user_input(self):
        prediction = self.perceptron.predict_classification(self.get_user_input())
        prediction = self.convert_back_to_label_name(prediction)
        print('Predicted iris: ' + prediction)

    def get_user_input(self):
        print('Enter vector values: ')
        vector = []
        for i in range(self.size):
            vector.append(float(input('Value: ')))

        vector = np.array(vector)
        return vector

    def convert_back_to_label_name(self, prediction):
        for i in range(len(self.input_names)):
            if prediction == i:
                return self.input_names[i]
            i += 1
