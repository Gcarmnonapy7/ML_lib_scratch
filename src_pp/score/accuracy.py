import numpy as np
def accuracy (y_true,y_hat):
    return np.sum(y_hat == y_true) / len(y_true)