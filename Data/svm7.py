import numpy as np
import math
import cvxopt
import cvxopt.solvers


def getKernel(kernel_name, gamma_or_p, X, X_dash=None,flag=0):
    num = X.shape[0]
    if flag!=0:
        x_K = np.zeros((num)).astype(float)
        match kernel_name:
            case 'rbf':
                if not gamma_or_p:
                    gamma_or_p = 1.0 / X.shape[-1]
                for i in range(num):
                    x_K[i] += math.exp(-1*(np.linalg.norm(X[i] -
                                      X_dash)**2)/(gamma_or_p**2))
            case 'polynomial':
                for i in range(num):
                    x_K[i] += math.pow(
                        (np.dot(X[i].T, X_dash) + 1), gamma_or_p)
            case _:  # 'linear'
                for i in range(num):
                    x_K[i] += np.dot(X[i].T, X_dash)
        return x_K
    else:
        x_K = np.zeros((num, num)).astype(float)
        match kernel_name:
            case 'rbf':
                if not gamma_or_p:
                    gamma_or_p = 1.0 / X.shape[-1]
                for i in range(num):
                    for j in range(num):
                        x_K[i][j] = math.exp(-1*(np.linalg.norm(X[i] -
                                                                X[j])**2)/(gamma_or_p**2))

            case 'polynomial':
                for i in range(num):
                    for j in range(num):
                        x_K[i][j] = math.pow(
                            (np.dot(X[i].T, X[j]) + 1), gamma_or_p)
            case _:  # 'linear'
                for i in range(num):
                    for j in range(num):
                        x_K[i][j] = np.dot(X[i].T, X[j])
        if abs(np.linalg.det(x_K)) <= 2:
            print('Det zero adding noise')
            noise = 0.001*np.eye(num)
            while abs(np.linalg.det(x_K)) <= 2:
                x_K += noise
        return x_K

    return None


class SVM:
    def __init__(self, kernel_name, C_vec, learning_rate=1e-3, gamma_or_p=1) -> None:
        self.kernel_name = kernel_name
        self.C_vec = C_vec
        self.lr = learning_rate
        self.gamma_or_p = gamma_or_p

    def test(self, train_x, train_y, test_x, test_y, support_vectors):
        num = train_x.shape[0]
        dis = 0.0
        accuracy_score = 0.0
        intercept = (np.argsort(support_vectors))[num - 1]
        x_K = getKernel(self.kernel_name, self.gamma_or_p,
                        train_x, train_x[intercept],1)
        for i in range(num):
            dis += x_K[i]*support_vectors[i]*train_y[i]
        for i in range(test_x.shape[0]):
            predicted_val = 0.0
            x_K = getKernel(self.kernel_name, self.gamma_or_p, train_x, test_x[i],1)
            for j in range(num):
                predicted_val += x_K[j]*train_y[j]*support_vectors[j]
            predicted_val = (
                (predicted_val + (1/train_y[intercept] - dis))) * test_y[i]
            if predicted_val >= 0:  # Check sign of predicted * actual value
                accuracy_score += 1
        return accuracy_score/(test_x.shape[0])

    def get_support_vectors(self, train_y, K, C):
        # Quadratic dual optimization:
        # minimize (1/2)*x'*P*x + q'*x subject to G*x <= h
        #        A*x = b
        num = train_y.shape[0]
        # P is a n x n dense or sparse 'd' matrix with the lower triangular part of P stored in the lower triangle. Must be positive semidefinite.
        P = cvxopt.matrix(np.outer(train_y, train_y)*K)
        # q is an n x 1 dense 'd' matrix
        q = cvxopt.matrix(np.ones(num) * -1)
        # G is an m x n dense or sparse 'd' matrix.
        G = cvxopt.matrix(np.vstack((np.eye(num)*-1, np.eye(num))))
        # h is an m x 1 dense 'd' matrix.
        h = cvxopt.matrix(np.hstack((np.zeros(num), np.ones(num) * C)))
        # A is a p x n dense or sparse 'd' matrix.
        A = cvxopt.matrix(train_y, (1, num))
        # b is a p x 1 dense 'd' matrix or None.
        b = cvxopt.matrix(0.0)
        # solver is None or 'mosek'.
        #solver = 'kkt'
        if self.kernel_name!='rbf':
            cvxopt.solvers.options['show_progress'] = False
            cvxopt.solvers.options['abstol'] = 1e-10
            cvxopt.solvers.options['reltol'] = 1e-10
            cvxopt.solvers.options['feastol'] = 1e-10
            cvxopt_output = cvxopt.solvers.qp(P, q, G, h, A, b,kktsolver='ldl', options={'kktreg':1e-9})
        else:
            cvxopt_output = cvxopt.solvers.qp(P, q, G, h, A, b) #,kktsolver='ldl', options={'kktreg':1e-9})
        # Returns a dictionary with keys 'status', 'x', 's', 'y', 'z', 'primal objective', 'dual objective', 'gap', ...etc
        # We need 'dual objective'
        support_vectors = np.ravel(cvxopt_output['x'])
        for i in range(support_vectors.shape[0]):
            if support_vectors[i] <= 1e-5:
                support_vectors[i] = 0
        return support_vectors

    def classify(self, train_x, train_y, test_x, test_y, epoch=10):
        arr_valid_c = []
        arr_accuracy = []
        for C in self.C_vec:
            try:
                K = getKernel(self.kernel_name, self.gamma_or_p, train_x,None,0)
                support_vectors = self.get_support_vectors(train_y, K, C)
                arr_accuracy.append(
                    self.test(train_x, train_y, test_x, test_y, support_vectors))
                arr_valid_c.append(C)
            except Exception as ex:
                print('Exception for C = ', C, '\n', ex)
        return arr_accuracy,arr_valid_c
