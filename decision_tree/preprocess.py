import numpy as np


def cont2mid(data_set, ai):
    """
    Preprocess of continuous attributes.
    :param data_set: Array
    :param ai: int
    :return: List()
    """
    col = list(data_set[:, ai])
    col.sort()
    t = [(col[i]+col[i+1])/2 for i in xrange(len(col)-1)]
    return t


if __name__ == '__main__':
    data_set = np.array([[1,2,3], [4,5,6], [7,8,9]], np.float)
    print cont2mid(data_set, 2)
