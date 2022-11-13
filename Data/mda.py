import numpy as np


class MDA:
    def __init__(self, noise, train_x, test_x, new_dim) -> None:
        self.noise = noise
        self.train_x = train_x
        self.test_x = test_x
        self.new_dim = new_dim
        self.mu = np.mean(np.mean(train_x, axis=1), axis=0)

    def get_matA(self):
        noise = self.noise*np.eye(self.train_x.shape[-1])
        # Calculate Prior * () [ x-mu.T * x-mu ] ) / N
        # Initialize zero matrix:
        A_class = np.zeros((self.train_x.shape[-1], self.train_x.shape[-1]))
        A_across = np.zeros((self.train_x.shape[-1], self.train_x.shape[-1]))
        for i in range(self.train_x.shape[0]):
            temp = self.train_x[i] - np.mean(self.train_x[i], axis=0)  # x-mu
            temp = np.matmul(temp.T, temp)   # [ x-mu.T * x-mu ]
            A_class += ((1/self.train_x.shape[0]) *
                        temp * (1/self.train_x.shape[1])) + noise
            A_across += (1/self.train_x.shape[0]) * np.outer((np.mean(
                self.train_x[i], axis=0) - self.mu),  (np.mean(self.train_x[i], axis=0) - self.mu))
        A = np.matmul(np.linalg.inv(A_class), A_across)

        U, _, __ = np.linalg.svd(A)
        A = U[:, :self.new_dim]
        self.A = A
        #print(A.shape)
        #return A

    def transform(self, X):
        X_dash = []
        for i in range(X.shape[0]):
            temp = np.matmul(self.A.T, X[i].T)
            X_dash.append(temp.T)
        return np.array(X_dash)