import numpy as np
def accuracy (y_true,y_hat):
    accuracy = np.sum(y_true == y_hat) / len(y_true)
    return accuracy