from svmutil import *

def readdata(filename):
    f = open(filename, 'r')
    x = [map(float, line.split(', ')) for line in f.readlines()]
    y = [1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    return (y, x)


if __name__ == '__main__':
    y, x = readdata('./dataset3.0_alpha.txt')
    m = svm_train(y[2:15], x[2:15], '-t 0')
    p_label, p_acc, p_val = svm_predict(y[:2] + y[15:], x[:2] + x[15:], m)
    print p_label