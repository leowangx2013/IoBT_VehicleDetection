import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from src.lstm import *
# from src.simclr_model import attach_simclr_head
from src.simclr_utility import *
from src.data_aug import *
from tensorflow.keras.layers import Dense,Dropout
from sklearn.metrics import confusion_matrix
import wandb
from wandb.keras import WandbCallback

def normalize_data(X_train, X_train_labeled, X_val_labeled, X_test):
    # Normalize the data
    for i in range(X_train.shape[0]):
        m = np.max(np.absolute(X_train[i]))
        X_train[i] = X_train[i]/m
    for i in range(X_train_labeled.shape[0]):
        m = np.max(np.absolute(X_train_labeled[i]))
        X_train_labeled[i] = X_train_labeled[i]/m
    for i in range(X_test.shape[0]):
        m = np.max(np.absolute(X_test[i]))
        X_test[i] = X_test[i]/m
    for i in range(X_val_labeled.shape[0]):
        m = np.max(np.absolute(X_val_labeled[i]))
        X_val_labeled[i] = X_val_labeled[i]/m   
    return X_train, X_train_labeled, X_val_labeled, X_test

def train_supervised(X_train_labeled, Y_train_labeled, X_val_labeled, Y_val_labeled, sample_len=256, batch_size=512, Epoch=500):

    '''
    Train the model under the supervised way.
    First build the whole model -- encoder + classifer layers.
    Then train the model from very beginning.
    '''
    encoder = model_LSTM_frequency(input_shape=[sample_len, 5])
    # encoder = model_vanilla()
    inputs = encoder.inputs

    dr=0.3
    r=1e-4
    x = encoder.output
    x=Dropout(dr)(x)
    x=Dense(256,activation="selu",name="FC1",kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)
    x=Dropout(dr)(x)

    outputs = Dense(9, activation="softmax", name="linear_Classifier")(x)
    sup_model = Model(inputs=inputs, outputs=outputs, name="Sup_Model")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    sup_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    sup_model.summary()

    wandb.config = {
        "learning_rate": r,
        "epochs": Epoch,
        "batch_size": batch_size,}

    history = sup_model.fit(X_train_labeled,
        Y_train_labeled,
        batch_size=batch_size,
        epochs=Epoch,
        verbose=2,
        validation_data=(X_val_labeled,Y_val_labeled),
        callbacks = [
                    tf.keras.callbacks.ModelCheckpoint("./saved_models/weight_sup.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto'),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.8,verbose=1,patince=5,min_lr=0.0000001),
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto'),
                    WandbCallback()           
                    ]
                        )
    
    print("Best val accuracy: ", max(history.history['val_accuracy']))

    print("=========== Supervised Training Completed! ==========")
    print("Save model to \'saved_models/weight_sup.hdf5\'")

    return sup_model

def eval_supervised(X_val_labeled, Y_val_labeled, sample_len=256):
    sup_model = tf.keras.models.load_model("./saved_models/weight_sup.hdf5")

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
            prediction = sup_model(X_val_labeled_single_run[j: j+n_sample])
            pred = tf.math.argmax(prediction, axis=-1).numpy().tolist()

            # print("prediction: ", prediction, ", pred: ", pred, ", y_pred: ", max(set(pred), key=pred.count))
            y_pred.append(max(set(pred), key=pred.count))

            true = tf.math.argmax(Y_val_labeled_single_run[j: j+n_sample], axis=-1).numpy().tolist()
            y_true.append(max(set(true), key=true.count))


            prediction = tf.one_hot(tf.math.argmax(prediction, axis=-1), depth=9)

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

    con_mat = confusion_matrix(y_true, y_pred)
    con_mat = con_mat / con_mat.sum(axis=1, keepdims=True)
    df_cm = pd.DataFrame(con_mat, range(9), range(9))
    plt.figure(figsize=(10,7))
    plt.title(f"Window Size = {n_sample}, Overall Accuracy = {correctness / (correctness + incorrectness)}")
    s = sn.heatmap(df_cm, annot=True)
    s.set(xlabel='Prediction', ylabel='True Label')
    plt.savefig(f"./n_win={n_sample}.png")
    wandb.log({"Confusion Matrix": wandb.Image(f"./n_win={n_sample}.png")})
    

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

def train_tune(X_train_labeled, Y_train_labeled, X_val_labeled, Y_val_labeled, batch_size=512, Epoch=200):
    '''
    Self-supervised contrastive learning will train the encoder.
    After that, we freeze the encoder and then build a classifier
         on the output of the encoder.
    Given the labeled training data, we will train the classifier
         as well as tune several layers of the encoder. 
    The number of layers to be tuned is related to the amount of
        labeled data we have. More labeled training data --> more tuned layers.
    '''
    # Load sim_model from file
    sim_model = tf.keras.models.load_model("./saved_models/weight_simclr.hdf5")
    
    inputs = sim_model.inputs
    x = sim_model.layers[-1].output

    dr=0.3
    r=1e-4

    x=Dropout(dr,name="DP_1")(x)
    x=Dense(128,activation="selu",name="FC1",kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)
    x=Dropout(dr,name="DP_2")(x)

    outputs = Dense(9, activation="softmax", name="linear_Classifier")(x)
    tune_model = Model(inputs=inputs, outputs=outputs, name="Tune_Model")
    
    for layer in tune_model.layers:
        layer.trainable = False

    # We can choose # of layers to be tuned according to the amount of labeled training data we have.
    # We tune less layers when we have a smaller number of labeled data.
    for layer in tune_model.layers:
        layer.trainable = True

    tune_model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    tune_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    
    
    training_history = tune_model.fit(
        x = X_train_labeled,
        y = Y_train_labeled,
        batch_size=batch_size,
        shuffle=True,
        epochs=Epoch,
        verbose=2,
        callbacks = [
                    tf.keras.callbacks.ModelCheckpoint("./saved_models/weight_tune.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto'),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.8,verbose=1,patince=5,min_lr=0.0000001),
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto'),           
                    ],
        validation_data=(X_val_labeled,Y_val_labeled)
    )

    print("Best val accuracy: ", max(training_history.history['val_accuracy']))

    print("=========== Tuning Completed! ==========")
    print("Save model to \'saved_models/weight_tune.hdf5\'")

    return tune_model

def train_simclr(X_train, batch_size=512, Epoch=100, temperature = 0.1, input_shape=[256, 5]):
    '''
    We use simclr as the self-supervised model.
    This function would train the encoder under the given unlabeled data.
    '''
    decay_steps = 15000

    # Training
    lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=0.001, decay_steps=decay_steps)
    optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)

    base_model = model_LSTM_frequency(input_shape=input_shape)
    # base_model = model_vanilla()
    sim_model = attach_simclr_head(base_model)
    sim_model.summary()
    trained_simclr_model, epoch_losses = simclr_train_model(sim_model, X_train, optimizer, batch_size, temperature=temperature, epochs=Epoch, verbose=1)

    print("=========== Contrastive Training Completed! ==========")
    print("Write model to \'saved_models/simclr/weight_simclr.hdf5\'")
    print("Write epoch loss to \'./results/simclr/epoch_loss\'")

    # Save Training Loss
    np.savetxt('./results/epoch_loss.csv', epoch_losses)
    return trained_simclr_model, epoch_losses

def plot_epoch_loss(loss_file='./results/epoch_loss.csv'):
    '''
    Plot the loss for each epoch during contrastive self-supervised training
    '''
    losses = np.loadtxt(loss_file)
    plt.figure()
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.title("Epoch Loss")
    plt.plot(losses)
    plt.savefig('./results/epoch_loss.png')

def compare_tune_and_sup(weight_tune, weight_sup, X_test, Y_test, test_idx, snrs, lbl, batch_size=512):
    '''
    In this function, we compare the performance for the tuned model (SemiAMC)
        and the supervised model.
    We study their
        1. General accuracies under the test dataset.
        2. Accuracies under different snrs.
    '''
    tune_model = tf.keras.models.load_model(weight_tune)
    sup_model = tf.keras.models.load_model(weight_sup)

    score = tune_model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print("Tuned model score:", score)
    score = sup_model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
    print("Supervised model score:", score)

    acc_tune = {}
    acc_sup = {}

    for snr in snrs:
        # Extract classes @ SNR
        test_SNRs = [lbl[x][1] for x in test_idx]
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]
        
        # # Estimate classes
        test_hat_i_tune = tune_model.predict(test_X_i)
        test_hat_i_sup = sup_model.predict(test_X_i)

        label_i= np.argmax(test_Y_i, axis=1)
        pred_i_tune = np.argmax(test_hat_i_tune, axis=1)
        pred_i_sup = np.argmax(test_hat_i_sup, axis=1)

        acc_tune[snr] = 1.0 * np.mean(label_i==pred_i_tune)
        acc_sup[snr] = 1.0 * np.mean(label_i==pred_i_sup)
        
    # Plot accuracy curve
    plt.figure()
    plt.plot(snrs, list(map(lambda x: acc_tune[x], snrs)), 'd-', label="SemiAMC")
    plt.plot(snrs, list(map(lambda x: acc_sup[x], snrs)), 'o-', label="Supervised")
    plt.xlim([-20,20])
    plt.ylim([0,0.8])
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title(" Classification Accuracy on RadioML")
    plt.grid('-.')
    plt.tight_layout()
    plt.legend()
    plt.savefig('./results/acc_under_snrs.png')