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
wandb.init(project="IoBT-vehicleclassification", entity="uiuc-dkara")

def convertLabels(Y):
    # takes a list of one hot vectors and converts to integer labels
    Y=np.argmax(Y,axis=1)
    # add 1 for index
    # Y = Y+1
    return Y

def createFeatures(X,sample_len=SAMPLE_LEN):
    # takes a single second dataframe and returns basic features
    # return pse with welch method for x
    features = []
    nperseg= 128
    # for j,x in enumerate(X):
    for index in range(len(X)):
        x = X[index] 
        for i in range(5):
            f, Pxx_den = signal.welch(x[:,i], sample_len, nperseg=nperseg)
            # take up to 375 Hz
            len_to_take = (3*len(f)) // 4
            wandb.log({"len_to_take": len_to_take})
            wandb.log({"nperseg": nperseg})
            if i==0:
                pse=Pxx_den[:len_to_take]
            else:
                pse=np.vstack((pse,Pxx_den[:len_to_take]))
        # f, Pxx_den = signal.welch(x, 1024, nperseg=128)
        # return np.asarray(pse).flatten()
        features.append(np.asarray(pse).flatten())
    return np.asarray(features)
    pass

def train_supervised_basic(X_train, Y_train, X_val, Y_val, sample_len=SAMPLE_LEN):
    
    X_train = createFeatures(X_train)
    X_val = createFeatures(X_val)
    Y_train = convertLabels(Y_train)
    Y_val = convertLabels(Y_val)


    model = xgb.XGBClassifier(objective='multi:softprob', n_estimators=400, 
                        num_classes=9)
    model.fit(X_train, Y_train,
            eval_set=[(X_train, Y_train), (X_val, Y_val)], 
            early_stopping_rounds=5) 
    
    return model
    pass

def eval_supervised_basic(model,X_val, Y_val, sample_len=SAMPLE_LEN):
    X_test = createFeatures(X_val)
    y_test = convertLabels(Y_val) +1

    y_pred = model.predict(X_test) +1
    from sklearn.metrics import multilabel_confusion_matrix

    print(multilabel_confusion_matrix(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %.3f' % accuracy)
    
    # find the sequence number of current window length
    precision = precision_score(y_test, y_pred, average='micro')
    print('Precision: %.3f' % precision)
    recall = recall_score(y_test, y_pred, average='micro')
    print('Recall: %.3f' % recall)
    
    f_score = f1_score(y_test,y_pred,average='micro')
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
    df_cm = pd.DataFrame(con_mat, range(9), range(9))
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



# Read dataset
filepath = "./data/Tank_classification/Tank_classification/Code/data/split_run"
# filepath = "./data/original/original"
# filepath="./data/switch_label_3/switch_label_3"

X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(filepath, sample_len=SAMPLE_LEN)


# ### Train the encoder + classifier from the very beginning under supervised way
sup_model = train_supervised_basic(X_train, Y_train, X_val, Y_val, sample_len=SAMPLE_LEN)
eval_supervised_basic(sup_model,X_val, Y_val, sample_len=SAMPLE_LEN)

exit()
eval_supervised(X_val, Y_val, sample_len=SAMPLE_LEN)

# ### Compare the result of our model and the supervised training results

# weight_tune = "./saved_models/weight_tune.hdf5"
# weight_sup = "./saved_models/weight_sup.hdf5"

# compare_tune_and_sup(weight_tune, weight_sup, X_test, Y_test, test_idx, snrs, lbl)

