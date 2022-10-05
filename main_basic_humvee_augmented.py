from cProfile import label
import os,random
from tabnanny import verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import numpy as np
# from src.dataset2016 import load_data
from src.dataset_arl import load_data
from statistics import mean
from src.utils import *
import argparse
import xgboost as xgb
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error,precision_score,recall_score,f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split,cross_validate,StratifiedKFold
from itertools import product
import pickle
from scipy import signal
from cgi import test
import os
import torch
import random
import getpass
import pickle as pkl
import numpy as np

from tqdm import tqdm
import glob

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

import wandb
# wandb.init(project="IoBT-vehicleclassification", entity="uiuc-dkara")

def convertLabels(Y):
    # takes a list of one hot vectors and converts to integer labels
    Y=np.argmax(Y,axis=1)
    # add 1 for index
    # Y = Y+1
    return Y

def createFeatures(X_acoustic, X_seismic,sample_len=SAMPLE_LEN):
    # takes a single second dataframe and returns basic features
    # return pse with welch method for x
    
    ## acoustic
    X = X_acoustic
    sample_len = 16000
    features_acoustic = []
    nperseg= 2000 # fft length up to 500 Hz
    for index in range(len(X)):
        x = X[index] 
        f, Pxx_den = signal.welch(x, sample_len, nperseg=nperseg)
        # take up to 1000 Hz
        len_to_take = (1*len(f)) // 8 # (3*len(f)) // 4
        # wandb.log({"len_to_take": len_to_take})
        # wandb.log({"nperseg": nperseg})
        
        pse=Pxx_den[:len_to_take]
        features_acoustic.append(np.asarray(pse).flatten())

    ## seismic
    X = X_seismic
    sample_len = 200
    features_seismic = []
    nperseg= 25 # fft length up to 500 Hz
    for index in range(len(X)):
        x = X[index] 
        f, Pxx_den = signal.welch(x, sample_len, nperseg=nperseg)
        # take up to 100 Hz
        len_to_take = len(f) # (1*len(f)) // 8
        # wandb.log({"len_to_take": len_to_take})
        # wandb.log({"nperseg": nperseg})
        
        pse=Pxx_den[:len_to_take]
        features_seismic.append(np.asarray(pse).flatten())

    # merge acoustic and seismic features
    features = []
    for i in range(len(features_acoustic)):
        features.append(np.concatenate((features_acoustic[i],features_seismic[i])))

    return np.asarray(features)
    pass

def train_supervised_basic(X_train_acoustic, X_train_seismic, Y_train, X_val_acoustic, X_val_seismic, Y_val, sample_len=SAMPLE_LEN):
    
    X_train = createFeatures(X_train_acoustic,X_train_seismic)
    X_val = createFeatures(X_val_acoustic,X_val_seismic)
    # Y_train = convertLabels(Y_train)
    # Y_val = convertLabels(Y_val)


    # model = xgb.XGBClassifier(objective='multi:softprob', n_estimators=400)
    model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=400)
    model.fit(X_train, Y_train,
            eval_set=[(X_train, Y_train), (X_val, Y_val)], 
            early_stopping_rounds=20) 
    pkl.dump(model, open("model.pkl", "wb"))
    return model
    pass

def eval_supervised_basic(model,X_val_acoustic,X_val_seismic, Y_val, sample_len=SAMPLE_LEN):
    
    if not model:
        model = pkl.load(open("model.pkl", "rb"))

    X_test = createFeatures(X_val_acoustic,X_val_seismic)
    # y_test = convertLabels(Y_val) +1
    y_test = Y_val
    y_pred = model.predict(X_test)
    from sklearn.metrics import multilabel_confusion_matrix

    print(multilabel_confusion_matrix(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %.3f' % accuracy)
    
    # find the sequence number of current window length
    precision = precision_score(y_test, y_pred, average='binary')
    print('Precision: %.3f' % precision)
    recall = recall_score(y_test, y_pred, average='binary')
    print('Recall: %.3f' % recall)
    
    f_score = f1_score(y_test,y_pred,average='binary')
    print('F1-Score: %.3f' % f_score)
    
    ## better confusion matrix
    X_val_labeled = X_test
    Y_val_labeled = y_test
    """
    runs = []
    filenames = []
    with open(f"val_file_sample_count_{sample_len}.txt", "r") as file:
        for line in file:
            segments = line.split(", ")
            filenames.append(segments[0])
            sample_n = int(segments[-1])
            runs.append(sample_n)
    runs = np.cumsum(runs)

    correctness = 0
    incorrectness = 0
    
    correctness_by_runs = np.zeros(len(runs))
    incorrectness_by_runs = np.zeros(len(runs))

    y_pred = []
    y_true = []
    
    for i in range(len(runs)):
        if i == 0:
            X_val_labeled_single_run = X_val_labeled[0: runs[i]]
            Y_val_labeled_single_run = Y_val_labeled[0: runs[i]]
        else:
            X_val_labeled_single_run = X_val_labeled[runs[i-1]: runs[i]]
            Y_val_labeled_single_run = Y_val_labeled[runs[i-1]: runs[i]]
        
        # n_sample = 1024 // sample_len
        n_sample = 1
        
        j = 0
        while j+n_sample <= len(X_val_labeled_single_run):
            prediction = model.predict(X_test[j: j+n_sample])+1
            pred = prediction# tf.math.argmax(prediction, axis=-1).numpy().tolist()

            # print("prediction: ", prediction, ", pred: ", pred, ", y_pred: ", max(set(pred), key=pred.count))
            y_pred.append(pred[0])

            # true = tf.math.argmax(Y_val[j: j+n_sample], axis=-1).numpy().tolist()
            # y_true.append(max(set(true), key=true.count))
            true = Y_val[j: j+n_sample]
            y_true.append(true[0])

            # prediction = tf.one_hot(tf.math.argmax(prediction, axis=-1), depth=9)

            sample_correctness = 0
            for p, label in zip(prediction, Y_val_labeled_single_run[j: j+n_sample]):
                if np.all(tf.math.equal(p, label).numpy()):
                    sample_correctness += 1
            if sample_correctness > n_sample // 2:
                correctness += 1
                correctness_by_runs[i] += 1
            else:
                incorrectness += 1
                incorrectness_by_runs[i] += 1
            j += n_sample
    """
    con_mat = confusion_matrix(y_test, y_pred)
    con_mat = con_mat / con_mat.sum(axis=1, keepdims=True)
    df_cm = pd.DataFrame(con_mat, range(len(set(y_test))), range(len(set(y_test))))
    plt.figure(figsize=(10,7))
    plt.title(f"Window Size = {1024}, Overall Accuracy = {accuracy}")
    s = sn.heatmap(df_cm, annot=True)
    s.set(xlabel='Prediction', ylabel='True Label')
    plt.savefig(f"./n_win={1024}.png")
    wandb.log({"Confusion Matrix": wandb.Image(f"./n_win={1024}.png")})
    wandb.log({"Accuracy": accuracy})
    """
    print(f"Correctness = {correctness}, incorrectness = {incorrectness}, accuracy = {correctness / (correctness + incorrectness)}")
    wandb.log({"Accuracy": correctness / (correctness + incorrectness),
                "Correctness": correctness,
                "Incorrectness": incorrectness})

    print("Accuracy by runs: \n")
    for n, (cor, incor, fn) in enumerate(zip(correctness_by_runs, incorrectness_by_runs, filenames)):
        print(n, fn, cor, incor, cor/(cor+incor))
        wandb.log({f"Accuracy by runs {fn}": cor/(cor+incor),
                    f"Correctness by runs {fn}": cor,
                    f"Incorrectness by runs {fn}": incor})
    """
    pass

def load_data_humvee(filepath, sample_len=256):

    def loaderHelper(index_filepath):
        train_index = []
    
        with open(index_filepath, "r") as file:
            for line in file:
                # last part of the line directory is the filename
                train_index.append(line.split("/")[-1].strip())
        # read training data from filepath
        X_train_acoustic = []
        X_train_seismic = []
        Y_train = []
        for training_case in train_index:
            # current case
            case = os.path.splitext(training_case)[0]
            # find all files in filepath including case
            files = glob.glob(os.path.join(filepath, case + "*"))
            
            for file in files:
                try:
                    # sample = torch.load(os.path.join(filepath, file))
                    sample = torch.load(file)
                    is_numpy = type(sample['data']['shake']['seismic'])==np.ndarray
                    if not is_numpy: #not "augmented" in case:
                        seismic= torch.flatten(sample['data']['shake']['seismic']).numpy()
                        acoustic = torch.flatten(sample['data']['shake']['audio']).numpy()
                    else:
                        seismic= sample['data']['shake']['seismic'].flatten()
                        acoustic = sample['data']['shake']['audio'].flatten()

                    if True: # do 1vsrest with humvee
                        if "humv" in file:
                            label = np.array(1)
                        else:
                            label = np.array(0)
                        pass
                    else:
                        label = sample['label'].numpy()
                    
                    X_train_acoustic.append(acoustic)
                    X_train_seismic.append(seismic)
                    Y_train.append(label)
                
                except:
                    print("Error reading file: ", file)
                    continue
        return X_train_acoustic, X_train_seismic, Y_train

    
    # preliminaries
    train_index_file = "time_data_partition/train_index.txt"
    val_index_file = "time_data_partition/val_index.txt"
    test_index_file = "time_data_partition/test_index.txt"
    # sample_rate_acoustic = 8000
    # sample_rate_seismic = 100 

    X_train_acoustic, X_train_seismic, Y_train = loaderHelper(train_index_file)
    X_val_acoustic, X_val_seismic, Y_val = loaderHelper(val_index_file)
    X_test_acoustic, X_test_seismic, Y_test = loaderHelper(test_index_file)

    
    X_train_acoustic = np.array(X_train_acoustic)
    X_train_seismic = np.array(X_train_seismic)
    Y_train = np.array(Y_train)
    X_val_acoustic = np.array(X_val_acoustic)
    X_val_seismic = np.array(X_val_seismic)
    Y_val = np.array(Y_val)
    X_test_acoustic = np.array(X_test_acoustic)
    X_test_seismic = np.array(X_test_seismic)
    Y_test = np.array(Y_test)
    
    # X_train_shape = (11495, 1024, 5)
    # Y_train.shape = (11495, 9)
    for i in range(len(X_val_acoustic)):
        m = np.max(np.absolute(X_val_acoustic[i]))
        X_val_acoustic[i] = X_val_acoustic[i]/m
    for i in range(len(X_val_seismic)):
        m = np.max(np.absolute(X_val_seismic[i]))
        X_val_seismic[i] = X_val_seismic[i]/m
    
    for i in range(len(X_train_acoustic)):
        m = np.max(np.absolute(X_train_acoustic[i]))
        X_train_acoustic[i] = X_train_acoustic[i]/m
    for i in range(len(X_train_seismic)):
        m = np.max(np.absolute(X_train_seismic[i]))
        X_train_seismic[i] = X_train_seismic[i]/m

    for i in range(len(X_test_acoustic)):
        m = np.max(np.absolute(X_test_acoustic[i]))
        X_test_acoustic[i] = X_test_acoustic[i]/m
    for i in range(len(X_test_seismic)):
        m = np.max(np.absolute(X_test_seismic[i]))
        X_test_seismic[i] = X_test_seismic[i]/m
    


    print("X_train_acoustic shape: ", X_train_acoustic.shape)
    print("X_train_seismic shape: ", X_train_seismic.shape)
    print("Y_train shape: ", Y_train.shape)
    print("X_val_acoustic shape: ", X_val_acoustic.shape)
    print("X_val_seismic shape: ", X_val_seismic.shape)
    print("Y_val shape: ", Y_val.shape)
    print("X_test_acoustic shape: ", X_test_acoustic.shape)
    print("X_test_seismic shape: ", X_test_seismic.shape)
    print("Y_test shape: ", Y_test.shape)

    pd.DataFrame(X_train_acoustic).to_csv("X_train_acoustic.csv")
    pd.DataFrame(X_train_seismic).to_csv("X_train_seismic.csv")
    pd.DataFrame(Y_train).to_csv("Y_train.csv")
    pd.DataFrame(X_val_acoustic).to_csv("X_val_acoustic.csv")
    pd.DataFrame(X_val_seismic).to_csv("X_val_seismic.csv")
    pd.DataFrame(Y_val).to_csv("Y_val.csv")
    pd.DataFrame(X_test_acoustic).to_csv("X_test_acoustic.csv")
    pd.DataFrame(X_test_seismic).to_csv("X_test_seismic.csv")
    pd.DataFrame(Y_test).to_csv("Y_test.csv")

    return X_train_acoustic, X_train_seismic, Y_train, X_val_acoustic, X_val_seismic, Y_val, X_test_acoustic, X_test_seismic, Y_test
    
if __name__ == "__main__":
    # filepath = "incas_data"
    filepath= "augmented_data"

    X_train_acoustic, X_train_seismic, Y_train, X_val_acoustic, X_val_seismic, Y_val, X_test_acoustic, X_test_seismic, Y_test = load_data_humvee(filepath)
    sup_model = train_supervised_basic(X_train_acoustic,X_train_seismic, Y_train, X_val_acoustic,X_val_seismic,Y_val)
    # sup_model = train_supervised_basic(X_train_acoustic,X_train_seismic, Y_train, X_test_acoustic,X_test_seismic,Y_test)
    sup_model=None # use saved model file
    eval_supervised_basic(sup_model,X_val_acoustic,X_val_seismic, Y_val, sample_len=SAMPLE_LEN)
    # eval_supervised_basic(sup_model,X_test_acoustic,X_test_seismic, Y_test, sample_len=SAMPLE_LEN)