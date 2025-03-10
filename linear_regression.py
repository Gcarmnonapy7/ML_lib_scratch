#Linear Regression's algorithm -> in lr we want to predict values there are predict and
# in classification values that can be express in 0 or 1
#In lr we have our data (usually represents in scatterplot) and we want to approximate it from a linear function
# Approximation -> y = wx + b // "Cost function" MSE -> function(x,b) = 1/len(samples) * sum(y - (wx + b)**2 )

import numpy as np
class LinearRegression() :
    def __init__(self,learning_rate=0.001,n_inters=1000):
        """
        :param learning_rate: learning rate
        :param n_inters: number of itereations
        """
        self.learning_rate = learning_rate
        self.n_inters = n_inters
        self.weight = None
        self.bias = None

    def fit(self,x,y):
        ''' implement the gradiant descent '''
        #Init values for our variables
        n_samples,n_features = x.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_inters):
            #Approximate and updating the weight and bias // need the dx/dw and dx/db (Loss function)
            y_predict = np.dot(x,self.weight) + self.bias #predictions // Line equation

            dw = (1/n_samples) * np.dot(x.T,(y_predict-y)) # gradient of self.weight // x.T is the transpose matrix
            db = (1/n_samples) * np.sum(y_predict - y) #gradient of self.bias

            #Update the weights and bias
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:
                #Print the loss after 100 n_iterations
                Loss = self.mse(y,y_predict)
                print(f"Iteration number{i} and Loss equals : {Loss}")

    def  predict(self,x):
        """predict -> line equation return """
        y_predict = np.dot(x,self.weight) + self.bias
        return y_predict

    def mse(self,y_real,y):
        """Mean Squared error"""
        return np.mean((y_real-y) ** 2)