from svmutil import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def readdata(csvfile):
    df = pd.read_csv(csvfile)
    density = df.density.tolist()
    sugar = df.sugar.tolist()
    x = [[val] for val in sugar]
    return (density, x)


def train(x, y):
    """
    -s : epsilon-SVR
    -t : radial basis
    -c : penalty
    """
    model = svm_train(y, x, '-s 3 -t 2 -c 100')

    return model


if __name__ == '__main__':
    y, x = readdata('./data_set3.0_alpha.csv')
    train_x = x[2:15]
    train_y = y[2:15]
    test_x = x[:2] + x[15:]
    test_y = y[:2] + y[15:]

    print("Train:")
    m = train(train_x, train_y)

    print("\nPredict:")
    p_label, p_acc, p_val = svm_predict(test_y, test_x, m)

    # Figure
    t_x = np.linspace(0, 0.5)
    t_x = t_x.tolist()
    t_y = t_x
    t_x = [[v] for v in t_x]
    p_label, p_acc, p_y = svm_predict(t_y, t_x, m)
    plt.plot(t_x, p_y)
    plt.plot(x, y, 'ro')
    plt.show()