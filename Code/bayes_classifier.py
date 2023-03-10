import numpy as np
from scipy.stats import multivariate_normal

class BayesClassifier:
    def __init__(self, noise) -> None:
        self.noise = noise
    def set_covariance(self,train) -> None:
        Th = np.eye(train.shape[2])
        cov = []
        for i in range(train.shape[0]):
            cov_ = np.cov(train[i].T)
            noise_ = self.noise*Th
            cov.append(cov_+noise_)
        self.covariance_Mat = np.array(cov)
    def classify(self,train_x,test_x):
        self.set_covariance(train_x)
        self.mean = np.mean(train_x,axis=1)
        prior = 1/(train_x.shape[1])
        predicted = []
        for i in range(test_x.shape[0]):
            posterior = []
            for j in range(train_x.shape[0]):
                # Calculate posteriors
                posterior.append(prior+ multivariate_normal.logpdf(test_x[i],mean=self.mean[j],cov=self.covariance_Mat[j]))
            # Get max posterior:
            predicted.append(np.argmax(posterior))
        # Preprocess : convert to labels ->
        return predicted,[('lbl' + str(x+1)) for x in predicted]
    
    