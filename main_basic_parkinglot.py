# from audioop import add
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
WANDB_ACTIVE=True
if WANDB_ACTIVE:
    wandb.init(project="IoBT-vehicleclassification", entity="uiuc-dkara")

def convertLabels(Y):
    # takes a list of one hot vectors and converts to integer labels
    Y=np.argmax(Y,axis=1)
    # add 1 for index
    # Y = Y+1
    return Y

def createFeatures(X_acoustic, X_seismic,sample_len=SAMPLE_LEN):
    # takes a single second dataframe and returns basic features
    # return pse with welch method for x
    from added_features import applyAndReturnAllFeatures
    feature_names = []
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
        len_to_take = (3*len(f)) // 4
        
        if WANDB_ACTIVE:
            wandb.log({"len_to_take": len_to_take})
            wandb.log({"nperseg": nperseg})
            wandb.log({"f": f[:len_to_take]})
        pse=Pxx_den[:len_to_take]
        
        additonal_features = applyAndReturnAllFeatures(x)
        additonal_feature_names = [k for k, v in sorted(additonal_features.items())] 
        additonal_features = [v for k, v in sorted(additonal_features.items())] #list(additonal_features.values())
        pse = np.concatenate((pse,additonal_features))

        features_acoustic.append(np.asarray(pse).flatten())
        pass
    # append to feature names
    for i in range(len_to_take):
        feature_names.append("pse_acoustic_f"+str(f[i]))
    for i in range(len(additonal_features)):
        feature_names.append('Acoustic'+additonal_feature_names[i])

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

        additonal_features = applyAndReturnAllFeatures(x)
        additonal_feature_names = [k for k, v in sorted(additonal_features.items())] 
        additonal_features = [v for k, v in sorted(additonal_features.items())] #list(additonal_features.values())
        pse = np.concatenate((pse,additonal_features))

        features_seismic.append(np.asarray(pse).flatten())

    # merge acoustic and seismic features
    features = []
    for i in range(len(features_acoustic)):
        features.append(np.concatenate((features_acoustic[i],features_seismic[i])))

    # append to feature names
    for i in range(len_to_take):
        feature_names.append("pse_seismic_f"+str(f[i]))
    for i in range(len(additonal_features)):
        feature_names.append('Seismic'+additonal_feature_names[i])

    return np.asarray(features),feature_names
    pass

def train_supervised_basic(X_train_acoustic, X_train_seismic, Y_train, X_val_acoustic, X_val_seismic, Y_val, sample_len=SAMPLE_LEN):
    
    X_train,feature_names = createFeatures(X_train_acoustic,X_train_seismic)
    X_val,feature_names = createFeatures(X_val_acoustic,X_val_seismic)
    # Y_train = convertLabels(Y_train)
    # Y_val = convertLabels(Y_val)


    # model = xgb.XGBClassifier(objective='multi:softprob', n_estimators=400)
    model = xgb.XGBClassifier(objective='binary:logistic')#,verbosity=3)
    model.fit(X_train, Y_train,
            eval_set=[(X_train, Y_train), (X_val, Y_val)], 
            early_stopping_rounds=20)

    model.get_booster().feature_names = feature_names
    # Interpret model
    interpretModel(model,X_val,feature_names,name="Val")
    interpretModel(model,X_train,feature_names,name="Train")
    if False:
        #model2= xgb.XGBClassifier(**model.get_params())
        print('Choosing best n_estimators as 50')
        model2= xgb.XGBClassifier(n_estimators=80,objective='binary:logistic',verbosity=3)
        full_train =np.concatenate((X_train,X_val))
        full_label = np.concatenate((Y_train,Y_val))
        model2.fit(full_train, full_label)
        pkl.dump(model2, open("model.pkl", "wb"))
        return model2
    else:
        pkl.dump(model, open("model.pkl", "wb"))
        return model
    pass

def eval_supervised_basic(model,X_val_acoustic,X_val_seismic, Y_val, sample_len=SAMPLE_LEN,files=None):
    import time
    print("Evaluating start time: ",time.time())
    if not model:
        model = pkl.load(open("model.pkl", "rb"))

    X_test,feature_names = createFeatures(X_val_acoustic,X_val_seismic)
    # y_test = convertLabels(Y_val) +1
    y_test = Y_val
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)
    interpretModel(model,X_test,feature_names,name="Test")
    print("Evaluating end time: ",time.time())
    if files:
        print(len(files))
        #print files where prediction is wrong
        for i in range(len(y_pred)):
            if y_pred[i] != y_test[i]:
                print(files[i])
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
    
    con_mat = confusion_matrix(y_test, y_pred)
    con_mat = con_mat / con_mat.sum(axis=1, keepdims=True)
    df_cm = pd.DataFrame(con_mat, range(len(set(y_test))), range(len(set(y_test))))
    plt.figure(figsize=(10,7))
    plt.title(f"Window Size = {1024}, Overall Accuracy = {accuracy}")
    s = sn.heatmap(df_cm, annot=True)
    s.set(xlabel='Prediction', ylabel='True Label')
    plt.savefig(f"./n_win={1024}.png")
    if WANDB_ACTIVE:
        wandb.log({"Confusion Matrix": wandb.Image(f"./n_win={1024}.png")})
        wandb.log({"Accuracy": accuracy})
        wandb.log({"Precision": precision})
        wandb.log({"Recall": recall})
        wandb.log({"F1-Score": f_score})
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

def load_data_sedan(filepath, sample_len=256):

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
        for file in train_index:
            try:
                sample = torch.load(os.path.join(filepath, file))
                seismic= torch.flatten(sample['data']['shake']['seismic']).numpy()
                acoustic = torch.flatten(sample['data']['shake']['audio']).numpy()
                
                if True: # do 1vsrest with humvee
                    if "mustang" in file:
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
    train_index_file = "time_data_partition_mustang/train_index.txt"
    val_index_file = "time_data_partition_mustang/val_index.txt"
    test_index_file = "time_data_partition_mustang/test_index.txt"
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
    
    '''
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
    
    '''

    print("X_train_acoustic shape: ", X_train_acoustic.shape)
    print("X_train_seismic shape: ", X_train_seismic.shape)
    print("Y_train shape: ", Y_train.shape)
    print("X_val_acoustic shape: ", X_val_acoustic.shape)
    print("X_val_seismic shape: ", X_val_seismic.shape)
    print("Y_val shape: ", Y_val.shape)
    print("X_test_acoustic shape: ", X_test_acoustic.shape)
    print("X_test_seismic shape: ", X_test_seismic.shape)
    print("Y_test shape: ", Y_test.shape)
    return X_train_acoustic, X_train_seismic, Y_train, X_val_acoustic, X_val_seismic, Y_val, X_test_acoustic, X_test_seismic, Y_test


def load_data_parkinglot(filepath, sample_len=256):

    def loaderHelper(index_filepath):
        train_index = []
    
        with open(index_filepath, "r") as file:
            for line in file:
                if 'txt' in line:
                    continue
                # last part of the line directory is the filename
                train_index.append(line.split("/")[-1].strip())
        # read training data from filepath
        X_train_acoustic = []
        X_train_seismic = []
        Y_train = []
        for file in train_index:
            try:
                sample = torch.load(os.path.join(filepath, file))
                seismic= torch.flatten(sample['data']['shake']['seismic']).numpy()
                acoustic = torch.flatten(sample['data']['shake']['audio']).numpy()
                
                if True: # do 1vsrest with humvee
                    if "driving" in file:
                        label = np.array(1)
                    elif "engine" in file:
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

    def train_test_val_split(filelist):
        random.seed(42)
        data= random.shuffle(filelist)
        train = filelist[:int(len(filelist)*0.8)]
        val = filelist[int(len(filelist)*0.8):int(len(filelist)*0.9)]
        test = filelist[int(len(filelist)*0.9):]
        return train, val, test

    def createIndexes(filepath):
        # from the files in the directory, create a list of files to read as train-test-val sets
        # create a list of all files in the directory
        files = os.listdir(filepath)
        # list of files including 'quiet'
        files_quiet = []
        # files including 'driving'
        files_driving = []
        # files including 'engine'
        files_engine = []
        for file in files:
            if "quiet" in file:
                files_quiet.append(file)
            elif "driving" in file:
                files_driving.append(file)
            elif "engine" in file:
                files_engine.append(file)
            elif 'txt' in file:
                continue
            else:
                print("Error: file not in quiet, driving, engine: ", file)
        
        training_set = []
        test_set = []
        val_set = []
        driving_train, driving_val, driving_test = train_test_val_split(files_driving)
        quiet_train, quiet_val, quiet_test = train_test_val_split(files_quiet)
        engine_train, engine_val, engine_test = train_test_val_split(files_engine)
        training_set.extend(driving_train)
        training_set.extend(quiet_train)
        training_set.extend(engine_train)
        val_set.extend(driving_val)
        val_set.extend(quiet_val)
        val_set.extend(engine_val)
        test_set.extend(driving_test)
        test_set.extend(quiet_test)
        test_set.extend(engine_test)
        
        # write the sets to files
        with open(os.path.join(filepath, "train_index.txt"), "w") as file:
            for line in training_set:
                file.write(line + "\n")
        with open(os.path.join(filepath, "test_index.txt"), "w") as file:
            for line in test_set:
                file.write(line + "\n")
        with open(os.path.join(filepath, "val_index.txt"), "w") as file:
            for line in val_set:
                file.write(line + "\n")
        # return filepaths
        return os.path.join(filepath, "train_index.txt"), os.path.join(filepath, "test_index.txt"), os.path.join(filepath, "val_index.txt")
        #return training_set, test_set, val_set
    
    # create indexes if they don't exist
    #if not os.path.exists(os.path.join(filepath, "train_index.txt")):
    train_index_file, test_index_file, val_index_file = createIndexes(filepath)

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
    
    '''
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
    
    '''

    print("X_train_acoustic shape: ", X_train_acoustic.shape)
    print("X_train_seismic shape: ", X_train_seismic.shape)
    print("Y_train shape: ", Y_train.shape)
    print("X_val_acoustic shape: ", X_val_acoustic.shape)
    print("X_val_seismic shape: ", X_val_seismic.shape)
    print("Y_val shape: ", Y_val.shape)
    print("X_test_acoustic shape: ", X_test_acoustic.shape)
    print("X_test_seismic shape: ", X_test_seismic.shape)
    print("Y_test shape: ", Y_test.shape)
    return X_train_acoustic, X_train_seismic, Y_train, X_val_acoustic, X_val_seismic, Y_val, X_test_acoustic, X_test_seismic, Y_test


def load_shake_data(filepath, sample_len=256):

    def loaderHelper(filepath):
        
        # read all file names from filepath ending with .pt
        files = []
        for file in os.listdir(filepath):
            if file.endswith(".pt"):
                files.append(file)
        
        # read training data from filepath
        X_train_acoustic = []
        X_train_seismic = []
        Y_train = []
        for file in files:
            try:
                sample = torch.load(os.path.join(filepath, file))
                seismic= torch.flatten(sample['data']['shake']['seismic']).numpy()
                acoustic = torch.flatten(sample['data']['shake']['audio']).numpy()
                
                if False: # do 1vsrest with humvee
                    if "humv" in file:
                        label = np.array(1)
                    else:
                        label = np.array(0)
                    pass
                else:
                    label = np.array(0) # 0 or 1 
                
                X_train_acoustic.append(acoustic)
                X_train_seismic.append(seismic)
                Y_train.append(label)
            
            except:
                print("Error reading file: ", file)
                continue
        return X_train_acoustic, X_train_seismic, Y_train, files

    
    # preliminaries
    # sample_rate_acoustic = 8000
    # sample_rate_seismic = 100 

    X_train_acoustic, X_train_seismic, Y_train, files = loaderHelper(filepath)
    
    
    X_train_acoustic = np.array(X_train_acoustic)
    X_train_seismic = np.array(X_train_seismic)
    Y_train = np.array(Y_train)
    
    
    '''
    for i in range(len(X_train_acoustic)):
        m = np.max(np.absolute(X_train_acoustic[i]))
        X_train_acoustic[i] = X_train_acoustic[i]/m
    for i in range(len(X_train_seismic)):
        m = np.max(np.absolute(X_train_seismic[i]))
        X_train_seismic[i] = X_train_seismic[i]/m
    '''
    
    print("X_train_acoustic shape: ", X_train_acoustic.shape)
    print("X_train_seismic shape: ", X_train_seismic.shape)
    print("Y_train shape: ", Y_train.shape)
    return X_train_acoustic, X_train_seismic, Y_train, files

def interpretModel(model,X_test,feature_names,name=None):
    import shap
    from matplotlib import pyplot

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test,feature_names= feature_names,show=False)
    if name:
        figname= name + "Shapley_summary_plot.png"
    else:
        figname= "Shapley_summary_plot.png"
    pyplot.savefig(figname,bbox_inches = "tight")
    if WANDB_ACTIVE:
        wandb.log({"Interpretation": wandb.Image(figname)})
        
    pyplot.close()

if __name__ == "__main__":

    mode = '0' # train data using pt_data
    mode = '1' # train data using both pt and sedan parkland data
    
    if mode=='0':
        filepath = "pt_data"

        X_train_acoustic, X_train_seismic, Y_train, X_val_acoustic, X_val_seismic, Y_val, X_test_acoustic, X_test_seismic, Y_test = load_data_parkinglot(filepath)
        sup_model = train_supervised_basic(X_train_acoustic,X_train_seismic, Y_train, X_val_acoustic,X_val_seismic,Y_val)
        # sup_model = train_supervised_basic(X_train_acoustic,X_train_seismic, Y_train, X_test_acoustic,X_test_seismic,Y_test)
        sup_model=None # use saved model file
        # eval_supervised_basic(sup_model,X_val_acoustic,X_val_seismic, Y_val, sample_len=SAMPLE_LEN)
        eval_supervised_basic(sup_model,X_test_acoustic,X_test_seismic, Y_test, sample_len=SAMPLE_LEN)
        
    elif mode=='1':
        filepath = "sedan_data"
        X_train_acoustic, X_train_seismic, Y_train, X_val_acoustic, X_val_seismic, Y_val, X_test_acoustic, X_test_seismic, Y_test = load_data_sedan(filepath)
        
        filepath = "pt_data"
        X_train_acoustic2, X_train_seismic2, Y_train2, X_val_acoustic2, X_val_seismic2, Y_val2, X_test_acoustic2, X_test_seismic2, Y_test2 = load_data_parkinglot(filepath)
        

        X_train_acoustic = np.concatenate((X_train_acoustic, X_train_acoustic2), axis=0)
        X_train_seismic = np.concatenate((X_train_seismic, X_train_seismic2), axis=0)
        Y_train = np.concatenate((Y_train, Y_train2), axis=0)
        X_val_acoustic = np.concatenate((X_val_acoustic, X_val_acoustic2), axis=0)
        X_val_seismic = np.concatenate((X_val_seismic, X_val_seismic2), axis=0)
        Y_val = np.concatenate((Y_val, Y_val2), axis=0)
        X_test_acoustic = np.concatenate((X_test_acoustic, X_test_acoustic2), axis=0)
        X_test_seismic = np.concatenate((X_test_seismic, X_test_seismic2), axis=0)
        Y_test = np.concatenate((Y_test, Y_test2), axis=0)
        
        #sup_model = train_supervised_basic(X_train_acoustic,X_train_seismic, Y_train, X_val_acoustic,X_val_seismic,Y_val)
        sup_model = train_supervised_basic(X_train_acoustic,X_train_seismic, Y_train, X_test_acoustic,X_test_seismic,Y_test)
        sup_model=None # use saved model file
        #eval_supervised_basic(sup_model,X_val_acoustic,X_val_seismic, Y_val, sample_len=SAMPLE_LEN)
        eval_supervised_basic(sup_model,X_test_acoustic,X_test_seismic, Y_test, sample_len=SAMPLE_LEN)

        pass
    else:
        shake_filepath = "bedroom_pt_data"        
        sup_model=None # use saved model file
        X_val_acoustic, X_val_seismic, Y_val, files = load_shake_data(shake_filepath)

        eval_supervised_basic(sup_model,X_val_acoustic,X_val_seismic, Y_val, sample_len=SAMPLE_LEN,files=files)
        # eval_supervised_basic(sup_model,X_test_acoustic,X_test_seismic, Y_test, sample_len=SAMPLE_LEN)