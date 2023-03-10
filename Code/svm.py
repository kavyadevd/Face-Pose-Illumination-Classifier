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

    def get_support_vectors(self, train_y, K, C=1):
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
    def classify_adaboost(self, train_x, test_x,all_labels,train_y,test_y,data_all_dash,epoch=10,step=10):
        train_acc = []
        test_acc = []
        loss_vec = []
        num_val = all_labels.shape[0] - test_x.shape[0]
        for e in range(1,epoch+1):
            print('-'*10)
            print('Adaboost Iteration ',e)

            ###### Initializations ################################
            # Step 1 Initialize weight vector:
            wn = np.zeros((e,train_x.shape[0])).astype(float)
            w0 = (train_x.shape[0])*np.ones(train_x.shape[0]).astype(float)
            wn[0] = 1/w0          # Iteration 0

            # Initialize propabilities:
            Pn = np.zeros((e,train_x.shape[0]))

            # Initialize ai s:
            ai = np.zeros((e))
            pred = np.zeros((train_x.shape[0]))

            # Initialize parameters:
            th_i = np.zeros((e))
            th = np.zeros((e, train_x.shape[1]))
            num = train_x.shape[0]
     
            ###### Update last weights and parameters ################################
            last_n = e-1
            error_points = 0
            for i in range(last_n):
                boost_yn = True                
                # datapoints to boost:
                error_points = 0
                # Init loss
                loss = 0.0
                # Normalize weights
                for n in range(num-1):
                    Pn[i][n] = wn[i][n] / (np.sum(wn[i]))

                ##### Boost data points with wrong classification till loss < 0.5
                while(boost_yn):    # while loss > 1/2
                    error_points += step
                    boosted_y = all_labels[:error_points]                    
                    boosted_x = data_all_dash[:error_points]
                    ## Classify                    
                    kernel = getKernel(self.kernel_name,0.006,boosted_x,None,0)
                    support_vectors = self.get_support_vectors(boosted_y,kernel)
                    th[i] = np.dot((support_vectors*boosted_y).T, boosted_x)
                    intercept = (np.argsort(support_vectors))[support_vectors.shape[0]-1]
                    th_i[i] = 1/boosted_y[intercept] - np.dot(th[i].T, boosted_x[intercept])
                    loss = 0.0
                    for n in range(num_val):
                        pred[n] = np.dot(th[i].T, data_all_dash[:num_val][n]) + th_i[i]
                        if (pred[n]*all_labels[:num_val][n] < 0):
                            loss += Pn[i][n]
                    loss_vec.append(loss)
                    if loss < 0.5:
                        boost_yn = False
                for n in range(num):
                    wn[i+1][n] = wn[i][n] * math.exp( (-1*all_labels[:num_val][n]*pred[n]) )
                try:
                    ai[i] = 0.5*np.log((1/loss)-1)
                except:
                    pass
                
            try: # Get train and test accuracy
                pred_correct = 0
                for i in range(num_val):
                    sign_ = 0
                    for j in range(e):
                        sign_ += ai[j] * np.dot(th[j].T, data_all_dash[:num_val][i]) + th_i[j]
                    if (sign_*all_labels[:num_val][i] > 0): # correct classification
                        pred_correct += 1
                train_acc.append((pred_correct)/num_val)                               
                pred_correct = 0
                for i in range(len(data_all_dash[num_val:])):
                    sign_ = 0
                    for j in range(e):
                        sign_ += ai[j] * np.dot(th[j].T, data_all_dash[num_val:][i]) + th_i[j]
                    if (sign_*all_labels[error_points:][i] > 0): # correct classification
                        pred_correct += 1
                test_acc.append((pred_correct)/len(data_all_dash[num_val:]))
            except Exception as ex:
                print(ex)
                train_acc.append(0.0)
                test_acc.append(0.0)
        return(train_acc, test_acc, loss_vec)