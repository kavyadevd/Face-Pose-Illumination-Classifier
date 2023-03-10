import numpy as np
from scipy.io import loadmat

# Binary classification


def GetNeutralvsExpData():
    print('-'*10, 'Data')
    data = loadmat('data_.mat')
    data = data['face']
    print(data.shape)
    data = np.moveaxis(data, -1, 0)
    print(data.shape)
    # Separate 3 faces for the 200 subjects
    data = data.reshape(200, 3, (24*21))
    data = data[:, :2, :]
    print('Data shape with neutral and Exp data: ', data.shape)
    return data


def GetTrainTestSplit(data_all,num_train_data, validation=False, validation_start=0):
    data_all = data_all.reshape(data_all.shape[0]*data_all.shape[1] , data_all.shape[2])
    # Generate labels:
    all_labels = np.empty((data_all.shape[0],))
    # Set neutral = 1, expression = -1
    all_labels[::2] = 1
    all_labels[1::2] = -1
    print('Test Labels: ')
    print(all_labels[0],all_labels[1],all_labels[2],all_labels[3],all_labels[4],all_labels[5],all_labels[6])

    # Set start index default = 0
    if validation:
        match validation_start:
            case 1:
                validation_start = 100
            case 2:
                validation_start = 200
            case 3:
                validation_start = 300

    # Indexes of [ original - validation patch ] 
    # validation path = validation_start to validation_start+num_train_data
    samples = np.setdiff1d(np.arange(
        0, data_all.shape[0]), np.arange(validation_start, validation_start+num_train_data))
    train_x = data_all[samples]
    train_y = all_labels[samples]

    test_x = data_all[np.arange(validation_start, validation_start+num_train_data)]
    test_y = all_labels[np.arange(validation_start, validation_start+num_train_data)]

    return([train_x,train_y,test_x,test_y])
