import argparse
from CVOA.CVOA import CVOA
from sklearn.datasets import load_iris, load_boston, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import time
import numpy as np


class Fitness():
    def __init__(self, X_train, y_train, X_evaluation, y_evaluation, model, type):
        self.model = model
        self.X_train, self.y_train = X_train, y_train
        self.X_evaluation, self.y_evaluation = X_evaluation, y_evaluation
        self.type = type

    def train(self, X_train_selected, y_train_selected):
        return self.model.fit(X_train_selected.values, y_train_selected.values)

    def __call__(self, individual, make_selection=True):
        if make_selection:
            X_train_selected = self.X_train[map(bool, individual.fixed_part)]
            X_evaluation_selected = self.X_evaluation[map(bool, individual.fixed_part)]
        else:
            X_train_selected = self.X_train
            X_evaluation_selected = self.X_evaluation

        if not any(map(bool, individual.fixed_part)):  # 0 featres selected
            individual.fitness = float('inf')
            return individual

        model = self.train(X_train_selected, self.y_train)

        predictions = model.predict(X_evaluation_selected.values)

        if self.type == 'classification':
            fitness = -accuracy_score(self.y_evaluation, predictions)
        else:
            fitness = mean_absolute_error(self.y_evaluation, predictions)
        individual.fitness = fitness

        return individual


def load_data(name):
    data = None
    if name == 'iris':
        data = load_iris(as_frame=True)
    elif name == 'boston':
        data = load_boston(as_frame=True)
    else:
        data = load_wine(as_frame=True)

    return data.data, data.target


def load_model(model_name, type):
    model = None

    if model_name == 'xgb' and type == 'classification':
        model = xgb.XGBClassifier()
    elif model_name == 'xgb' and type != 'classification':
        model = xgb.XGBRegressor()
    if model_name == 'svm' and type == 'classification':
        model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    elif model_name == 'svm' and type != 'classification':
        model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Name of dataset", choices=['iris', 'boston', 'wine'], default='iris',
                        required=True)
    parser.add_argument("-t", "--test", help="Test size", default=.3)
    parser.add_argument("-v", "--val", help="Validation size", default=.3)
    parser.add_argument("-n", "--normalize", help="Normalize data", action="store_true", default=False)
    parser.add_argument("-m", "--model", help="Type of model (xgb,  rf, svm)",
                        choices=['xgb', 'rf', 'svm'], default='xgb')
    parser.add_argument("-ty", "--type", help="Type of problem",
                        choices=['classification', 'regression'], default='classification')
    parser.add_argument("-p", "--processes", help="Number of processes (Default 2)",
                        type=int, default=2)
    parser.add_argument("-uv", "--use_variable", help="Use variable size in optimization",
                        action="store_true", default=False)
    parser.add_argument("-di", "--discrete", help="Discrete optimization problem",
                        action="store_true", default=False)
    parser.add_argument("-tm", "--time", help="Max time",
                        type=int, default=100)

    args = parser.parse_args()

    X, y = load_data(args.data)
    features = X.columns
    num_features = len(features)
    print(features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.val)

    model = load_model(args.model, args.type)

    fitness = Fitness(X_train, y_train, X_val, y_val, model, args.type)

    max_values = [1 for _ in range(num_features)]
    min_values = [0 for _ in range(num_features)]

    cvoa = CVOA(size_fixed_part=num_features, fixed_part_max_values=max_values, max_time=args.time,
                fixed_part_min_values=min_values, use_var_part=args.use_variable, discrete=args.discrete,
                fitness=fitness)

    start_time = int(round(time.time() * 1000))
    solution = cvoa.run()
    delta_time = int(round(time.time() * 1000)) - start_time

    fitness_evaluation = Fitness(X_train, y_train, X_test, y_test, model, args.type)
    fitness_evaluation(solution)

    selected_features = [feat for feat, selected in zip(features, solution.fixed_part) if selected > 0]

    log = """====================================================================
Best solution: {}
Selected features: {}
Best fitness validation: {}
Fitness test: {}
Fitness without selection: {}
Execution time: {}"""
    log = log.format(solution, selected_features, fitness(solution).fitness, fitness_evaluation(solution).fitness,
                     fitness_evaluation(solution, False).fitness, round(delta_time / 60000, 2))

    print(log)

    with open("log_fs_wine.txt", "a+") as myfile:
        myfile.write(log)


if __name__ == '__main__':
    main()
