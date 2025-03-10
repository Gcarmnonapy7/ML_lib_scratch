import numpy as np

def deviation(x) :
    """ std formula """
    return np.sqrt(np.sum(x-np.mean(x) ** 2) / len(x))

class CustomStandardScaler :
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self,X):
        ''' calculate the mean and de deviation(std) of X'''
        self.mean = np.mean(X)
        self.std = np.mean(X)
    def transform(self,X):
        """standard scaler formula"""
        return (X - self.mean) / self.std
    def fit_transform(self,X):
        """ fit and transform """
        self.fit(X)
        return self.transform(X)