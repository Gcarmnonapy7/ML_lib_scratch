import numpy as np
#The logistic regression is very similar to the linear regression(implementation // not the function)
class LogisticRegression:
    def __init__(self,lr=0.001,epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
    def fit (self,x,y):
        '''
        :return: "stored" the data
        '''
        n_samples,n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.epochs):
            linear_model = np.dot(x,self.weights) + self.bias
            y_hat = self.sigmoid(linear_model)

            #derivatives of weights and bias
            dw = (1/n_samples) * np.dot(x.T,(y_hat- y))
            db = (1/n_samples) * np.sum(y_hat- y)

            #uptade the weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

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