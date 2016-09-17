import numpy as np
import pandas as pd
from math import exp
from numpy.linalg import inv

def sigmoid(x):
    return 1 / (1 + exp(-x))


def logit_regression(X, y, iterations=15):
    """
    :param X: np.matrix
    :param y: np.array
    :param iterations: int
    :return: np.array
    """
    m, n = X.shape[0], X.shape[1]
    X = np.c_[X, np.ones([m,1])]
    X = np.matrix(X.T)  # convert to matrix

    beta = np.random.rand(n+1, 1)   # parameter

    for it in xrange(iterations):
        # Derivations
        d1 = np.zeros([n+1, 1])
        d2 = np.zeros([n+1, n+1])

        for i in xrange(m):
            xi = X[:, i]
            # probability P(y=1|x)
            p1 = sigmoid(beta.T.dot(xi))
            # Calculate first order and second order derivations
            d1 = d1 + (- xi * (y[i] - p1))
            d2 = d2 + xi.dot(xi.T) * p1 * (1 - p1)

        # Update beta (Newton method)
        beta = beta - inv(d2).dot(d1)

    test = ((X.T.dot(beta)).T).A1
    results = map(sigmoid, test)
    print [val > 0.5 for val in results]

    return beta

if __name__ == '__main__':
    df = pd.read_csv('data_set3.0_alpha.csv')
    label = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]).T
    beta = logit_regression(df, label, 15)

    print beta

