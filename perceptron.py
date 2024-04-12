import pandas as pd
from confusion_metrics import *
from matplotlib import pyplot as plt


def split_data(data: pd.DataFrame) -> tuple:
    data = data.sample(frac=1)
    cutoff = int(data.index.max() * 0.75)
    x_train = data.iloc[:cutoff, :-1]
    y_train = data.iloc[:cutoff, [-1]]
    x_test = data.iloc[cutoff:, :-1]
    y_test = data.iloc[cutoff:, [-1]]
    return x_train, y_train, x_test, y_test


def check_progress(progress):
    average = sum(progress[-10:-1])/9
    is_change_real = True
    print(average)
    for i in range(10):
        is_change_real = False if not 0.0005 > progress[-i] - average > -0.0005 and is_change_real else True
    return is_change_real


class Perceptron:

    def __init__(self, steps: int, x_train, y_train, x_test, y_test, bias: float, alpha: float, beta: float):
        self.steps = steps
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
        self.weights = [0.1 for _ in range(x_train.shape[1])]

    def perceptron(self, looked_value=1) -> list[float]:
        progress = []
        self.transform_y_values(looked_value)

        for i in range(self.steps):
            self.weights, self.bias = self.training()
            progress.append(classification_metrics(self.test_perceptron(), self.y_test, only_f1=True))
        outcome = self.test_perceptron()
        return outcome

    def transform_y_values(self, looked_value):
        self.y_test = self.y_test.map(lambda value: 1 if value == looked_value else 0)
        self.y_train = self.y_train.map(lambda value: 1 if value == looked_value else 0)

    def write_weigths(self):
        with open("perceptron_weights.txt", "a") as file:
            file.write(f"{self.weights}\n")

    @staticmethod
    def write_progress(output, y_test):
        with open("perceptron_progress.txt", "a") as file:
            file.write(f"{classification_metrics(output, y_test)}\n")

    def test_perceptron(self) -> list[float]:
        output = [-1.0 for _ in range(self.x_test.shape[0])]
        for i in range(len(output)):
            zipped = list(zip(self.x_test.iloc[i].values.tolist(), self.weights))
            output[i] = self.z_function(zipped, self.bias)
        return output

    def training(self) -> tuple:
        for i in range(self.x_train.shape[0]):
            zipped = list(zip(self.x_train.iloc[i].values.tolist(), self.weights))
            output = self.z_function(zipped, self.bias)
            if output != self.y_train.iloc[i, 0]:
                self.weights, self.bias = self.correction(zipped, self.bias, self.y_train.iloc[i, 0], output,
                                                          self.alpha, self.beta)
        return self.weights, self.bias

    @staticmethod
    def correction(vectors, bias: float, correct_output, output, alpha, beta) -> tuple:
        new_vectors = [float(vector[1]) + (correct_output - output) * vector[0] * alpha for vector in vectors]
        bias += (output - correct_output) * beta
        return new_vectors, bias

    @staticmethod
    def z_function(x_weights: list, bias: float) -> float:
        function = -sum(x_weights[i][0] * float(x_weights[i][1]) for i in range(len(x_weights))) + float(bias)
        return 1 if function < 0 else 0

    @staticmethod
    def z_score_normalization(df) -> pd.DataFrame:
        for column in df.columns[:-1]:
            df[column] = ((df[column] - df[column].mean()) / df[column].std())
        return df


def plot_points(x_train, y_train, x_test, y_test) -> None:
    df_train_below_zero = x_train[y_train.iloc[:, -1] == 0]
    df_train_above_zero = x_train[y_train.iloc[:, -1] == 1]
    df_test_below_zero = x_test[y_test.iloc[:, -1] == 0]
    df_test_above_zero = x_test[y_test.iloc[:, -1] == 1]

    training_plot_below = plt.scatter(x=df_train_below_zero.iloc[:, 0], y=df_train_below_zero.iloc[:, 1], c="blue")
    training_plot_above = plt.scatter(x=df_train_above_zero.iloc[:, 0], y=df_train_above_zero.iloc[:, 1],  c="red")

    testing_plot_below = plt.scatter(x=df_test_below_zero.iloc[:, 0], y=df_test_below_zero.iloc[:, 1], c="cyan")
    testing_plot_above = plt.scatter(x=df_test_above_zero.iloc[:, 0], y=df_test_above_zero.iloc[:, 1], c="orange")

    plt.grid()
    plt.show()
