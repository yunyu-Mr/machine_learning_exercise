"""
Car Evaluation Data Set

| class values
unacc, acc, good, vgood

| attributes
buying:   vhigh, high, med, low.
maint:    vhigh, high, med, low.
doors:    2, 3, 4, 5more.
persons:  2, 4, more.
lug_boot: small, med, big.
safety:   low, med, high.
"""

import pandas as pd
import numpy as np
from svmutil import *
from sklearn.cross_validation import KFold


class CarEvaluation(object):
    def __init__(self):
        pass

    def loaddata(self, csvfile):
        # Read dataframe
        df = pd.read_csv(csvfile)
        # Get attribute
        attrs = list(df.columns)
        label = attrs.pop()  # The last one is the label 'y'
        # Get data
        data = df[attrs + [label]]

        for attr in attrs + [label]:
            # Coding categories.
            l = data[attr]
            category = l.unique()
            l_dict = {category[i]: i for i in xrange(len(category))}
            data[attr] = map(lambda cat: l_dict[cat], l)

            # Scaling to [-1, +1]
            col = data[attr]
            r = col.max() - col.min()
            data[attr] = ((col*2 - col.min()) - r) / r

        # Transform data format to the form of an SVM package.
        y = np.array(data[label])
        X = data[attrs]
        X = np.array(map(list, X.values))

        return y, X

    def grid_search(self, y, X):
        # cs = [2^-5, 2^-3, ..., 2^15]
        cs = 2.0 ** np.array(range(-5, 16, 2))
        # gamas = [2^-15, 2^-13, ..., 2^3]
        gamas = 2.0 ** np.array(range(-15, 4, 2))

        acc = 0
        param = (0, 0)
        for c in cs:
            for gama in gamas:
                new_acc = self.cross_validate(y, X, k=10, c=c, gama=gama)
                if new_acc > acc:
                    acc = new_acc
                    param = (c, gama)

        return acc, param

    def cross_validate(self, y, X, k=10, c=100, gama=0.01):
        """
        :param y: np.array
        :param X: np.array(list)
        :param k: int
        :return:
        """
        acc_total = 0   # total accuracy
        n = y.size  # length
        # Define k_fold.
        kf = KFold(n, n_folds=k, shuffle=True)
        # Split into k folds.
        for train_idx, test_idx in kf:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train model
            m = self.train(y_train, X_train, c=c, gama=gama)

            # Predict
            p_label, p_acc, p_val = self.predict(y_test, X_test, m)
            acc_total += p_acc[0]

        return acc_total / k

    def train(self, y, X, c=100, gama=0.01):
        """
        :param y: np.array
        :param X: np.array(list)
        :param c: int (penalty)
        :return: model
        """
        # Use RBF kernel (radial basis)
        model = svm_train(y.tolist(), X.tolist(), '-t 2 -c %f -g %f' % (c, gama))
        return model

    def predict(self, y, X, m):
        p_label, p_acc, p_val = svm_predict(y.tolist(), X.tolist(), m)
        return p_label, p_acc, p_val


if __name__ == '__main__':
    car = CarEvaluation()
    y, X = car.loaddata('car.csv')

    acc, param = car.grid_search(y, X)
    print("Accuracy: %2.3f, param: C=%f, Gama=%f" % (acc, param[0], param[1]))