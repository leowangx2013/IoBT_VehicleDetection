"Adapted from the code (https://github.com/leena201818/radioml) contributed by leena201818"
import os
import numpy as np
from scipy import signal
import csv

def load_X(filename, sample_len=256):
    data = []
    with open(filename, "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for line in csv_reader:
            sample = [np.float32(i) for i in line]
            sample = np.reshape(sample, [5, sample_len])
            sample = np.transpose(sample, (1, 0)) # [batch, time, feature]

            data.append(sample)
    return data

def load_Y(filename):
    data = []
    with open(filename, "r") as file:
        csv_reader = csv.reader(file, delimiter=",")
        for line in csv_reader:
            sample = [np.float32(i) for i in line]
            data.append(sample)
    return data

def load_data(filepath, sample_len=256):
    '''
    Input: 
        filename: path of dataset
        size_train_labeled: [0,1000], # of labeled training samples
        size_val_labeled: [0,1000], # of labeled validation samples
    Output:
        X_train, Y_train: all training data set
        X_val, Y_val: all validation data set
        X_test, Y_test: all testing data set
        X_train_labeled, Y_train_labeled: labeled training data
        X_val_labeled, Y_val_labeled: labeled validation data
    '''

    X_train = load_X(os.path.join(filepath, f"train_X_{sample_len}.csv"), sample_len=sample_len)
    Y_train = load_Y(os.path.join(filepath, f"train_Y_{sample_len}.csv"))

    X_val = load_X(os.path.join(filepath, f"eval_X_{sample_len}.csv"), sample_len=sample_len)
    Y_val = load_Y(os.path.join(filepath, f"eval_Y_{sample_len}.csv"))   

    X_test = load_X(os.path.join(filepath, f"test_X_{sample_len}.csv"), sample_len=sample_len)
    Y_test = load_Y(os.path.join(filepath, f"test_Y_{sample_len}.csv"))        

    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    Y_val = np.array(Y_val, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    Y_test = np.array(Y_test, dtype=np.float32)

    # Normalize data
    for i in range(len(X_train)):
        m = np.max(np.absolute(X_train[i]))
        X_train[i] = X_train[i]/m
    for i in range(len(X_val)):
        m = np.max(np.absolute(X_val[i]))
        X_val[i] = X_val[i]/m
    for i in range(len(X_test)):
        m = np.max(np.absolute(X_test[i]))
        X_test[i] = X_test[i]/m

    print("X_train.shape: ", np.array(X_train).shape)
    print("Y_train.shape: ", np.array(Y_train).shape)
    print("X_val.shape: ", np.array(X_val).shape)
    print("Y_val.shape: ", np.array(Y_val).shape)
    print("X_test.shape: ", np.array(X_test).shape)
    print("Y_test.shape: ", np.array(Y_test).shape)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test
