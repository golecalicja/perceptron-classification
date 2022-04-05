from algorithm_evaluator import AlgorithmEvaluator
from data_cleaner import DataCleaner
from data_loader import DataLoader
from user_input_predictor import UserInputPredictor
from weights_trainer import WeightsTrainer

alpha = 0.0001
number_of_epochs = 1000

train_file = '../data/iristrain.csv'
test_file = '../data/iristest.csv'


def main():
    data_loader = DataLoader(train_file, test_file)
    df_train, df_test = data_loader.load_data()
    input_names = data_loader.get_user_input()
    data_cleaner = DataCleaner(df_train, df_test, input_names)
    train, test = data_cleaner.cleaned()
    weights_trainer = WeightsTrainer(train, alpha, number_of_epochs)
    perceptron = weights_trainer.train_weights()
    algorithm_evaluator = AlgorithmEvaluator(perceptron, test)
    algorithm_evaluator.evaluate_model()
    user_input_predictor = UserInputPredictor(perceptron, input_names, test[0].size - 1)
    user_input_predictor.predict_user_input()


if __name__ == '__main__':
    main()
