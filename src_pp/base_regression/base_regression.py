import numpy as np


class BaseRegression:
    def __init__(self,lr=0.001,n_inters=1000):
        self.lr = lr
        self.n_interns = n_inters
        self.weights = None
        self.bias = None
        
    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        
        for _ in range(self.n_interns):
            y_hat = np.dot(X,self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T,(y_hat - y))
            db = (1/n_samples) * np.sum(y_hat - y)
            
            self.weights -= self.lr * dw
    
    def _predicted(self):
        raise NotImplementedError("The method 'predicted' is not implemented")
    
    def approximation(self,X):
        y_hat = np.dot(self.weights,X) + self.bias
