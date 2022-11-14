import numpy as np

class KNN:
    def __init__(self,k,train_x,train_y,test_x) -> None:
        self.k = k
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        # Initialize distances to zero
        self.distances = np.zeros((self.test_x.shape[0],self.train_x.shape[0]))
    def set_distance(self):
        X = self.test_x        
        # Calculate distances of each point in test to each point in train
        #self.distances = np.sqrt((X**2).sum(axis=1)[:, np.newaxis] + (self.train_x**2).sum(axis=1) - 2 * X.dot(self.train_x.T))
        for i in range(X.shape[0]):
            for j in range(self.train_x.shape[0]):
                self.distances[i, j] = np.sqrt(np.sum((X[i, :] - self.train_x[j, :]) ** 2))
    def classify(self):
        self.set_distance()
        predicted = np.zeros(self.test_x.shape[0])
        try:
            all_labels = np.array(self.train_y)
            for i in range(self.test_x.shape[0]):
                indx = np.argsort(self.distances[i,:])
                selected_k = all_labels[indx[:self.k]].astype(int)
                selected_k = np.bincount(selected_k)
                predicted[i] = np.argmax(selected_k)
            return predicted
        except Exception as ex:
            print(ex)
            return(predicted)
