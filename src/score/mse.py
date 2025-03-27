import numpy as np

def mse(y_true,y_hat):
    '''Mean Squared Error'''
    return np.mean((y_true - y_hat) ** 2)