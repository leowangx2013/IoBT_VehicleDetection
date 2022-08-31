'''
In this file, we implement the 'rotation' data augmentation operations 
for modulation recognition mentioned in paper:
    "Data Augmentation for Deep Learning-based Radio Modulation Classification"
    https://arxiv.org/pdf/1912.03026.pdf
'''
import numpy as np
from src import data_aug_utils

def data_aug_time_rotation(X_batch):
    X_batch = data_aug_utils.rotation(X_batch)
    return X_batch

def data_aug_time_warp(X_batch):
    X_batch = data_aug_utils.time_warp(X_batch, np.random.random()*0.2)
    return X_batch

def data_aug_time_scaling(X_batch):
    X_batch = data_aug_utils.scaling(X_batch, np.random.random()*0.1)
    return X_batch

def data_aug_time_noise(X_batch):
    X_batch = data_aug_utils.jitter(X_batch, np.random.random()*0.03)
    return X_batch
