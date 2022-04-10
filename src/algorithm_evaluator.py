def get_correct_predictions(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct


class AlgorithmEvaluator:
    def __init__(self, perceptron, test):
        self.perceptron = perceptron
        self.test = test

    def evaluate_model(self):
        correct, accuracy = self.calculate_accuracy()
        print('Correct predictions: %d' % correct)
        print('Accuracy: {:.1%}'.format(accuracy))

    def calculate_accuracy(self):
        actual = self.test[:, -1]
        predicted = []
        for row in self.test:
            prediction = self.perceptron.predict_classification(row)
            predicted.append(prediction)
        correct = get_correct_predictions(actual, predicted)
        accuracy = correct / len(actual)
        return correct, accuracy

    def get_test_set_predictions(self):
        predictions = list()
        for row in self.test:
            prediction = self.perceptron.predict_classification(row)
            predictions.append(prediction)
        return predictions
