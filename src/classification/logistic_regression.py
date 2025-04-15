import numpy as np
from src.base_regression.base_regression import BaseRegression
#The logistic regression is very similar to the linear regression(implementation // not the function)
class LogisticRegression(BaseRegression):
    
    def predict(self,X):
        """
        return the predict of the x_test in 0 or 1
        """
        linear_model = np.dot(X,self.weights) + self.bias
        y_hat = self.sigmoid(linear_model)
        predicted = np.where(y_hat > 0.5 ,1,0) #np form to right a condicional => if y_hat > 0.5 : return 1 else 0
        return predicted
    def sigmoid(self,z):
        '''
        :return: Implementation of the sigmoid formula / Activation function / y_hat
        '''
        return 1/(1 + np.exp(-z)) # to optimize it use expit