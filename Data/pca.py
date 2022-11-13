import numpy as np


class PCA:
    def __init__(self, component) -> None:
        self.component = component

    def fit(self, X):
        X_mu = X - np.mean(X, axis=0)
        cov_ = np.cov(X_mu.T, rowvar=True)

        eigen_val, eigen_vec = np.linalg.eig(cov_)

        # Set high to low:
        indx = np.argsort(eigen_val)[::-1]
        eigen_val = eigen_val[indx]
        eigen_vec = (eigen_vec.T)[indx]

        # Select top component components:
        eigen_vec = eigen_vec[0:self.component]
        self.component = eigen_vec
        self.mu = np.mean(X)

    def transform(self, X):
        mu = np.mean(X)
        X -= mu #self.mu
        X_dash = np.dot(X,self.component.T)
        return X_dash
