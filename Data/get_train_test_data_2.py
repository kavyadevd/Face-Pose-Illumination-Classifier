import warnings
warnings.filterwarnings('ignore')
from scipy.io import loadmat
import numpy as np
from sklearn.metrics import accuracy_score
from bayes_classifier import *


def GetAllData():
    data_all = {}
    # data.mat
    # 200 subjects
    # 3 faces per subject
    # size: 24 x 21
    print('-'*10,'Data')
    data = loadmat('data.mat')
    data = data['face']
    print(data.shape)
    data = np.moveaxis(data, -1, 0)
    print(data.shape)
    # Separate 3 faces for the 200 subjects
    data = data.reshape(200,3,(24*21))
    # LAbel each 200 classes with their index and set as label
    labels_data = []
    for i in range(data.shape[0]):
        # labels_data.append('lbl'+str(i+1))
        labels_data.append(i)

    print(data.shape)
    data_all['data'] = (data,labels_data)
    print('-'*10,'Data')

    print('-'*10,'Pose')
    # 68 subjects
    # 13 images per subject (13 different poses)
    # size: 48 x 40
    pose = loadmat('pose.mat')
    pose = pose['pose']
    labels_pose = []
    for i in range(pose.shape[3]):
        #labels_pose.append('lbl'+str(i+1))
        labels_pose.append(i)
    print(pose.shape)
    pose = np.moveaxis(np.moveaxis(pose,-1,0),-1,1)
    print(pose.shape)
    data_all['pose'] = (pose,labels_pose)
    print('-'*10,'Pose')

    print('-'*10,'Illum')
    # 68 subjects
    # 21 images per subject (21 different illuminations)
    # size: 48x40
    illum = loadmat('illumination.mat')
    illum = illum['illum']
    labels_illum = []
    print(illum.shape)
    for i in range(illum.shape[2]):
        #labels_illum.append('lbl'+str(i+1))
        labels_illum.append(i)
    illum = np.moveaxis(np.moveaxis(illum,-1,0),-1,1)
    print(illum.shape)
    data_all['illum'] = (illum,labels_illum)
    print('-'*10,'Illum')

    return data_all



def get_train_test_data():
    data_all = GetAllData()
    #  [data,pose,illum]
    n = [200,68,68]
    m = [2,7,16]
    #
    c=0
    train_test_all = {}  # Format-> Dataset: (TrainX, TrainY, TestX, TestY)
    # Can test the effect of expressions, illumination variations. 
    # Here we test the effect of illumination for data.mat
    for k in data_all:
        train_datax = data_all[k][0][:,:m[c],:]
        train_datay = data_all[k][1]
        samples = range(data_all[k][0].shape[0])
        # ndarray generated random samples for test data
        rand_indexs = np.array(np.random.choice(samples, n[c], replace = False))
        test_datax = data_all[k][0][rand_indexs,m[c],:]
        test_datay = np.array(data_all[k][1])[rand_indexs]
        train_test_all[k] = (train_datax,train_datay,test_datax,test_datay)
        c+=1
    return train_test_all
