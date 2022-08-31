#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import numpy as np
# from src.dataset2016 import load_data
from src.dataset_arl import load_data
from statistics import mean
from src.utils import *
import argparse

parser = argparse.ArgumentParser()

# dataset config
parser.add_argument("--gpu", type=str, default="0", help="Visible GPU")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# ### Set Random Seed

# In[3]:

seed = 1994
random.seed(seed)  
np.random.seed(seed) 
tf.random.set_seed(seed)


SAMPLE_LEN = 1024

# ### Read Dataset

# In[4]:


'''
In dataset RML2016.10a, for each modulation type each snr, there're 1000 samples.
We use radnomly select 750 samples as training data (unlabled) adn 250 as testing.
As lableded dataset, in this experiment, we use size_train_labeled = 2 as training set
and size_val_labeled = 1 as validation set.
'''
size_train_labeled = 2 # [0,1000], # of labeled training samples
size_val_labeled = 1   # [0,1000], # of labeled validation samples

# Read dataset
# filename = '/data/dongxin3/2016.10a/RML2016.10a_dict.pkl'
# filename = './data/RML2016.10a_dict.pkl'
# (mods,snrs,lbl),(X_train,Y_train),(X_train_labeled,Y_train_labeled),(X_val_labeled,Y_val_labeled),(X_test,Y_test),    (train_idx,test_idx,train_labeled_idx,val_labeled_idx)    = load_data(filename, size_train_labeled, size_val_labeled)

filepath = "/home/tianshi/SemiAMC/data/Tank_classification/Tank_classification/Code/data/split_run"
X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(filepath, sample_len=SAMPLE_LEN)


# print("# of all training data:", X_train.shape[0])
# print("# of all testing data:", X_test.shape[0])
# print("# of labeled data:", X_train_labeled.shape[0]+X_val_labeled.shape[0],       "( Training:",X_train_labeled.shape[0], "Validation:",X_val_labeled.shape[0],")")

# ### Train Contrastive Learning Model - Simclr

# SimLCR
sim_model, epoch_losses = train_simclr(X_train, batch_size=512, Epoch=100, temperature=0.1)
plot_epoch_loss()

# ### Build a classifier on the output of the encoder and tune the parameter of the encoder with labeled data

# Tune Model
tune_model = train_tune(X_train, Y_train, X_val, Y_val)


# ### Train the encoder + classifier from the very beginning under supervised way
sup_model = train_supervised(X_train, Y_train, X_val, Y_val, sample_len=SAMPLE_LEN)

eval_supervised(X_val, Y_val, sample_len=SAMPLE_LEN)
exit()
# ### Compare the result of our model and the supervised training results

# In[17]:


weight_tune = "./saved_models/weight_tune.hdf5"
weight_sup = "./saved_models/weight_sup.hdf5"

compare_tune_and_sup(weight_tune, weight_sup, X_test, Y_test, test_idx, snrs, lbl)

