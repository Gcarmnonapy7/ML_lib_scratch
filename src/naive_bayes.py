import numpy as np 
#Based on Bayes Theorem that says when we have two events A/B then 
#the proba of A happen when B already happened => P(A/B) = (P(B/A) * P(A)) / P(B)
#feature vector x => (n_samples,n_features) in the naive bayes we assume that all the features are mutually independent
#IRL the features aren't independent all the time but this assumption works ok for certains problems 
#Objective => pick the class with the highest proba => y = argmax(class proba) => log the pxn/y and sum it and for least add 
class NaiveBayes:
    def __init__(self):
        self.priors = None
        self.classes = None
        self.mean = None
        self.std = None 
        
    def fit(self,x,y):
        
        n_samples,n_features = x.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        #Init the values of std(variances), mean and priors
        self.mean = np.zeros((n_classes,n_features),dtype=np.float64)
        self.std = np.zeros((n_classes,n_features),dtype=np.float64)
        self.priors = np.zeros(n_classes,dtype=np.float64)
        #Calculate the mean,std and the priors for each classes that is equals to the label(y)
        for index , classe in enumerate(self.classes):
            X_classe = x[y == classe]
            self.mean[index,:] = X_classe.mean(axis=0) #The mean of the samples with the class equals to y
            self.std[index,:] = X_classe.var(axis=0)
            self.priors[index] = X_classe.shape[0] / float(n_samples)
            
    def predict(self,X):
        '''
        Interate over all the samples in X
        '''
        return [self._predict(x) for x in X]       
    
    def _predict(self,X):
        '''
        Helper function for the predict function above
        '''
        posteriors = []
        
        for index, classe in enumerate(self.classes):
            prior = np.log(self.priors[index])
            class_conditional_y = np.sum(np.log(self._probability_density_function(index , X)))
            posterior = prior + class_conditional_y
            posteriors.append(posterior)
            
        return self.classes[np.argmax(posteriors)] # this is the y_argmax that sum the logs of the prior and the class_conditional(probability_density_function)
    
    
    def _probability_density_function(self,index,X):
        '''
        Formula of the probability density function // Gaussian PDF
        '''
        mean = self.mean[index]
        std = self.std[index]
        #Ensure the std is non-zero
        std = np.clip(std,1e-6, None) #Avoid division by zero
        
        numerator = np.exp(-(X - mean) ** 2/(2*std))
        denominator = np.sqrt(2 * np.pi * std)
        
        return numerator / denominator