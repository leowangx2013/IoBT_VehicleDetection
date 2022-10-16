import sys, argparse, os
import pyaudio, wave
import numpy as np
import time,datetime,os,csv
from threading import Thread, current_thread
from obspy.core import UTCDateTime
from obspy import read
from obspy.clients.seedlink import *
from obspy.clients.seedlink.easyseedlink import create_client
from getmac import get_mac_address as gma
from collections import deque
# import tflite_runtime.interpreter as tflite
import torch
from scipy import signal
from scipy.io.wavfile import write
import pickle as pkl

from params.test_params import parse_test_params
from models.DeepSense_rasp import DeepSense
from general_utils.weight_utils import load_model_weight

# Parameters for Acousmic Collection
# CHUNK          = 8000  # frames to keep in buffer between reads
CHUNK = 8000
samp_rate      = 16000 # sample rate [Hz]
pyaudio_format = pyaudio.paInt16 # 16-bit device
buffer_format  = np.int16 # 16-bit for buffer
chans          = 6 # only read 1 channel
dev_index      = 1 # index of sound device  

X_SEI = list()
X_AUD = list()
START_TIME_AUD = float('inf')
START_TIME_SEI = float('inf')

LABELS = {0: 'non-humvee', 1: 'humvee'}

def get_stationID():
    ''' Get Station ID of Seismic according to the MAC Address '''
    mac_address = gma()
    StationID = mac_address[-5:-3]+mac_address[-2:]
    StationID = 'R'+StationID.upper()
    return StationID

def callback(in_data,frame_count, time_info, status):
    ''' Callback func for acoustic data collection '''
    list_data = np.frombuffer(in_data,dtype=buffer_format)[::6]
    # X_AUD.extend(list_data[::960])
    # print("list_data len: ", len(list_data)far)
    X_AUD.extend(list_data)
    # print(frame_count, time_info, status, len(QUE), len(list_data), len(X_AUD))
    return (None, pyaudio.paContinue)

def prepare_data_for_lite(x_aud, x_sei):
    ''' Format the data to as input of the lite model '''
    x_a = x_aud[np.newaxis,...]
    x_s = x_sei[np.newaxis,...]

    f, t, x_a = signal.stft(x_a, nperseg=64, noverlap=63, boundary=None)
    x_a = np.abs(x_a)                
    f, t, x_s = signal.stft(x_s, nperseg=64, noverlap=63, boundary=None)
    x_s = np.abs(x_s)

    x_a = np.swapaxes(x_a, 1,2)
    x_s = np.swapaxes(x_s, 1,2)

    x_a = x_a[..., np.newaxis]
    x_s = x_s[..., np.newaxis]
    x_a = np.float32(x_a)
    x_s = np.float32(x_s)

    return x_a, x_s

def prepare_data_for_deepsense(x_aud, x_sei):
    x_aud = signal.resample(x_aud, len(x_aud) // 2)
    x_aud = torch.tensor(x_aud).float()
    x_aud = torch.reshape(x_aud, (1, 1, 10, 1600))

    x_sei = torch.tensor(x_sei).float()
    x_sei = torch.reshape(x_sei, (1, 1, 10, 20))

    print("x_aud.shape: ", x_aud.shape)
    print("x_sei.shape: ", x_sei.shape)
    input_data = {"shake": {"audio": x_aud, "seismic": x_sei}}
    return input_data




class MyClient(EasySeedLinkClient):
    ''' Implement the on_data callback for seismic data collection'''
    def on_data(self, trace):
        global START_TIME_SEI

        data = trace.data
        time = trace.times("timestamp")
        if START_TIME_SEI > time[0]:
            START_TIME_SEI = time[0]
        
        X_SEI.extend(data)

    def terminate(self):
        self.conn.terminate()

def collect_seismic(client):
    ''' Select a stream and start receiving data '''
    StationID = get_stationID()
    client.select_stream('AM', StationID, 'EHZ')
    client.run()

def createFeatures(X_acoustic, X_seismic,sample_len=SAMPLE_LEN):
    # takes a single second dataframe and returns basic features
    # return pse with welch method for x
    from added_features import applyAndReturnAllFeatures
    
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
        # wandb.log({"len_to_take": len_to_take})
        # wandb.log({"nperseg": nperseg})
        
        pse=Pxx_den[:len_to_take]
        
        additonal_features = applyAndReturnAllFeatures(x)
        additonal_features = [v for k, v in sorted(additonal_features.items())] #list(additonal_features.values())
        pse = np.concatenate((pse,additonal_features))

        features_acoustic.append(np.asarray(pse).flatten())
        pass
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
        additonal_features = [v for k, v in sorted(additonal_features.items())] #list(additonal_features.values())
        pse = np.concatenate((pse,additonal_features))

        features_seismic.append(np.asarray(pse).flatten())

    # merge acoustic and seismic features
    features = []
    for i in range(len(features_acoustic)):
        features.append(np.concatenate((features_acoustic[i],features_seismic[i])))

    
    return np.asarray(features)
    pass


def main():
    global X_SEI, X_AUD, START_TIME_AUD, START_TIME_SEI
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model_weight",
        type=str,
        default="/home/myshake/AutoCuration/weights/Parkland_humvee_DeepSense/augmented_quiet/",
        help="Dataset to evaluate.",
    )
    args = parse_test_params()
    
    print(args)
    # Seismic
    client = MyClient('127.0.0.1:18000')
    # print(client.get_info("ID"))
    t0 = Thread(target=collect_seismic, args=(client,))
    t0.start()

    # Acoustic
    audio = pyaudio.PyAudio() # create pyaudio instantiation
    stream = audio.open(format = pyaudio_format,rate = samp_rate,channels = chans, \
                        input_device_index = dev_index,input = True, \
                        frames_per_buffer=CHUNK, \
                        stream_callback=callback)

    stream.start_stream() # start data stream
    T_0 = datetime.datetime.now() # get datetime of recording start
    T_0 = datetime.datetime.timestamp(T_0)
    START_TIME_AUD = round(T_0, 2)

    # DeepSense Model
    # logfile_lite = "/home/myshake/IOBT-OS/Sensor/liteModel/deepsense.tflite"
    weight_file = "/home/myshake/AutoCuration/weights/Parkland_humvee_DeepSense/run1/Parkland_humvee_DeepSense_pretrain_best.pt"
    # logfile_lite = "/home/myshake/IOBT-OS/Sensor/liteModel/deepsense.tflite_stale"
    # Load the TFLite model and allocate tensors.
    #classifier = DeepSense(args, self_attention=False)
    #classifier = load_model_weight(classifier, weight_file)
    #classifier.eval()
    # Get input and output tensors.
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    # simple model
    # model = pkl.load(open("/home/myshake/AutoCuration/src/models/model-original.pkl", "rb"))
    model = pkl.load(open("/home/myshake/AutoCuration/src/models/model-augmented.pkl", "rb"))
                
    try:
        target = -1
        N = 0
        sync = False

        while True:
            # Synchronize seismic and acoustic
            if START_TIME_AUD > START_TIME_SEI:
                diff_index = int((START_TIME_AUD - START_TIME_SEI) * 100)
                while len(X_SEI) < diff_index:
                    time.sleep(0.1)
                X_SEI = X_SEI[diff_index:]
                START_TIME_AUD = 0
                START_TIME_SEI = 0
                sync = True

            if sync:
                while len(X_AUD)<32000 or len(X_SEI)<200:
                    time.sleep(0.1)

                start_time = time.time()
                # Take the first 100 elements in the list
                # x_aud = np.array(X_AUD[:100])
                # x_sei = np.array(X_SEI[:100])
                x_aud = np.array(X_AUD[:32000])
                x_sei = np.array(X_SEI[:200])
                X_AUD = X_AUD[32000:]
                X_SEI = X_SEI[200:]

                write(f"./sounds/{start_time}.wav", 16000, np.array(x_aud).astype(np.int16))
                input_data = prepare_data_for_deepsense(x_aud, x_sei)
                # input_data = {"shake": {"audio": x_aud, "seismic": x_sei}}
                
                seismic= torch.flatten(input_data['shake']['seismic']).numpy()
                acoustic = torch.flatten(input_data['shake']['audio']).numpy()
                
                
                X_acoustic= acoustic
                X_seismic= seismic

                X_test = createFeatures(X_acoustic,X_seismic)
                # x_a, x_s = prepare_data_for_lite(x_aud, x_sei)

                # interpreter.set_tensor(input_details[0]['index'], x_a)
                # interpreter.set_tensor(input_details[1]['index'], x_s)
                # interpreter.invoke()

                # output_data = interpreter.get_tensor(output_details[0]['index'])


                
                
                
                # logits = classifier(input_data)
                #print(logits)
                #pred = int(logits.argmax(dim=1, keepdim=False)[0].numpy())
                print(X_test)
                pred= model.predict(X_test)[0]
                pred2= model.predict_proba(X_test)[0]
                print(pred)
                print(pred2)
                print("label: ", LABELS[pred]," , time cost: ", time.time()-start_time)
                #print("label: ", LABELS[pred], ", confidence: ", logits[0][pred].detach().numpy(), ", time cost: ", time.time()-start_time)
                print("X_AUD len: ", len(X_AUD), ", X_SEI len: ", len(X_SEI))
                if pred ==1 :
                    print('FALSE POSITIVE, stop now')
                
                # ind = np.argmax(output_data)
                # label = LABELS[ind]
                # conf_level = output_data[0,ind]
                
                # print(f"{label} - Confident Level: {conf_level} - Time: {datetime.datetime.now()}")

                # if conf_level > 0.9:
                #     if ind == target:
                #         N += 1
                #         if N == 3:
                #             ############################################################
                #             ''' Send a message to corresponding camera '''
                #             ''' The camera would shot a picture and send to the server '''
                #             # send_msg_to_camera()
                #             print(f"Send msg to camera, target = {LABELS[target]}")
                #             N = 0
                #             ############################################################
                #     else:
                #         target = ind
                #         N = 1

            # time.sleep(0.95)

    # When ctrl+c is received
    except KeyboardInterrupt as e:
        # Set the alive attribute to false
        t0.alive = False
        client.terminate()

        stream.stop_stream() # stop data stream
        stream.close()
        audio.terminate() # close the pyaudio connection

        t0.join()
        sys.exit(e)

if __name__ == "__main__":
    main()
