from main import BasicModel
import numpy as np
from collections import Counter
#K-Nearest Neighbors Algorithm
#The basic idea is determined a class of a sample based in the nearest neighbors labels
#And predict the label by the most common label near the sample => using the Euclidean distance

def euclidean_distance(x1,x2):
    '''
    Euclidean distance formula in numpy
    '''
    return np.sqrt(np.sum(x1-x2)**2)

class KNN(BasicModel):

    def __init__(self, k=3):
        self.nearest_n = k  # Number of neighbors

    def fit(self, X, Y):
        '''Store the data'''
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        '''
        For each sample they are gonna store it individual (helper to _predict)
        '''
        predict_labels = [self._predict(x) for x in X]
        return np.array(predict_labels)

    def _predict(self, x):
        '''
        Storing distance -> Get nearest sample and labels-> majority vote(most common label)
        '''
        distance = [euclidean_distance(x,x_train) for x_train in self.X_train]
        k_index = np.argsort(distance)[:self.nearest_n]
        k_nearest_labels = [self.Y_train[i] for i in k_index]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
