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
seed = 1994
random.seed(seed)  
np.random.seed(seed) 
tf.random.set_seed(seed)


SAMPLE_LEN = 1024
WANDB_ACTIVE = True
# WANDB_ACTIVE = True

if WANDB_ACTIVE:
    import wandb
    wandb.init(project="IoBT-vehicleclassification", entity="uiuc-dkara")

# Read dataset
filepath = "./data/Tank_classification/Tank_classification/Code/data/split_run"
# filepath = "./data/original/original"
# filepath="./data/switch_label_3/switch_label_3"

X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(filepath, sample_len=SAMPLE_LEN)

# ### Train Contrastive Learning Model - Simclr

# SimLCR
# sim_model, epoch_losses = train_simclr(X_train, batch_size=512, Epoch=100, temperature=0.1, input_shape=[SAMPLE_LEN, 5])
# plot_epoch_loss()

# ### Build a classifier on the output of the encoder and tune the parameter of the encoder with labeled data

# Tune Model
# tune_model = train_tune(X_train, Y_train, X_val, Y_val)

# ### Train the encoder + classifier from the very beginning under supervised way
sup_model = train_supervised(X_train, Y_train, X_val, Y_val, sample_len=SAMPLE_LEN)

eval_supervised(X_val, Y_val, sample_len=SAMPLE_LEN)

# ### Compare the result of our model and the supervised training results

# weight_tune = "./saved_models/weight_tune.hdf5"
# weight_sup = "./saved_models/weight_sup.hdf5"

# compare_tune_and_sup(weight_tune, weight_sup, X_test, Y_test, test_idx, snrs, lbl)

