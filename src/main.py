from algorithm_evaluator import AlgorithmEvaluator
from data_cleaner import DataCleaner
from data_loader import DataLoader, get_user_input
from user_input_predictor import UserInputPredictor
from weights_trainer import WeightsTrainer

alpha = 0.0001
number_of_epochs = 1000

train_file = '../data/iristrain.csv'
test_file = '../data/iristest.csv'


def get_prepared_train(input_names):
    data_loader = DataLoader(train_file)
    df_train = data_loader.load_data()
    data_cleaner = DataCleaner(df_train, input_names)
    train = data_cleaner.cleaned()
    return train


def get_prepared_test(input_names):
    data_loader = DataLoader(test_file)
    df_test = data_loader.load_data()
    data_cleaner = DataCleaner(df_test, input_names)
    test = data_cleaner.cleaned()
    return test


def main():
    input_names = get_user_input()
    train = get_prepared_train(input_names)
    test = get_prepared_test(input_names)
    weights_trainer = WeightsTrainer(train, alpha, number_of_epochs)
    perceptron = weights_trainer.train_weights()
    algorithm_evaluator = AlgorithmEvaluator(perceptron, test)
    algorithm_evaluator.evaluate_model()
    user_input_predictor = UserInputPredictor(perceptron, input_names, test[0].size - 1)
    user_input_predictor.predict_user_input()


if __name__ == '__main__':
    main()
