#Linear Regression's algorithm -> in lr we want to predict values there are predict and
# in classification values that can be express in 0 or 1
#In lr we have our data (usually represents in scatterplot) and we want to approximate it from a linear function
# Approximation -> y = wx + b // "Cost function" MSE -> function(x,b) = 1/len(samples) * sum(y - (wx + b)**2 )
from src.base_regression import BaseRegression
import numpy as np
class LinearRegression(BaseRegression) :
   
    def  predict(self,x):
        """predict -> line equation return """
        y_hat = np.dot(x,self.weight) + self.bias
        return y_hat
