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
import matplotlib.pyplot as plt
import queue
import time
import torch
from scipy import signal
from scipy.io.wavfile import write
import pickle as pkl


from params.test_params import parse_test_params
from models.DeepSense_rasp import DeepSense
from general_utils.weight_utils import load_model_weight

# Parameters for Acousmic Collection
CHUNK = 16000
samp_rate      = 16000 # sample rate [Hz]
pyaudio_format = pyaudio.paInt16 # 16-bit device
buffer_format  = np.int16 # 16-bit for buffer
chans          = 1 # only read 1 channel
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

def prepare_data_for_deepsense(x_aud, x_sei):
    x_aud = signal.resample(x_aud, len(x_aud) // 2)
    x_aud = torch.tensor(x_aud).float()
    x_aud = torch.reshape(x_aud, (1, 1, 10, 1600))

    x_sei = torch.tensor(x_sei).float()
    x_sei = torch.reshape(x_sei, (1, 1, 10, 20))

    input_data = {"shake": {"audio": x_aud, "seismic": x_sei}}
    return input_data

class AudioThread(Thread):
    def __init__(self, q: queue.Queue):

        super().__init__(daemon=True)

        self.q = q

        audio = pyaudio.PyAudio() # create pyaudio instantiation
        self.stream = audio.open(format = pyaudio_format,rate = samp_rate,channels = chans, \
                            input_device_index = dev_index,input = True, \
                            frames_per_buffer=CHUNK, \
                            stream_callback=self.cb)

    def cb(self, in_data, frame_count, time_info, status):
        list_data = np.frombuffer(in_data,dtype=np.int16)
        self.q.put(list_data)
        return (None, pyaudio.paContinue)

    def run(self):
        self.stream.start_stream()

class SeismicThread(Thread):
    def __init__(self, q: queue.Queue):

        super().__init__(daemon=True)

        self.q = q

        self.station_id = "R105D"
        self.client = create_client("127.0.0.1:18000", on_data=self.cb)
        self.client.select_stream("AM", self.station_id, "EHZ")

    def cb(self, trace):
        for d in trace.data:
            self.q.put(d)

    def run(self):
        self.client.run()


def createFeatures(X_acoustic, X_seismic):
    # takes a single second dataframe and returns basic features
    # return pse with welch method for x
    from added_features import applyAndReturnAllFeatures
    #print("X_acoustic: ", X_acoustic.shape)
    ## acoustic
    X = X_acoustic
    sample_len = 16000
    features_acoustic = []
    nperseg= 2000 # fft length up to 500 Hz
    #for index in range(len(X)):
    x = X
    #print("x: ", x.shape)
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
    
    ## seismic
    X = X_seismic
    sample_len = 200
    features_seismic = []
    nperseg= 25 # fft length up to 500 Hz
    #for index in range(len(X)):
    x = X 
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

def main():
    args = parse_test_params()

    queue_audio = queue.Queue(maxsize=10)
    audio_client = AudioThread(queue_audio)
    audio_client.start()

    queue_seismic = queue.Queue(maxsize=1000)
    seismic_client = SeismicThread(queue_seismic)
    seismic_client.start()
    
    classifier = DeepSense(args, self_attention=False)
    classifier = load_model_weight(classifier, args.model_weight)
    classifier.eval()
    
    # simple model
    #model = pkl.load(open("/home/myshake/AutoCuration/src/models/model_mustanglabeled0.pkl", "rb"))
    model = pkl.load(open("/home/myshake/AutoCuration/src/models/model_mustanglabeled1.pkl", "rb"))
    model = pkl.load(open("/home/myshake/AutoCuration/src/models/model_onlypt.pkl", "rb"))
    
    try:
        n = 0
        while True:
            start_time = time.time()

            while queue_audio.qsize() < 2 or queue_seismic.qsize() < 200:
                time.sleep(0.1)

            x_aud = []
            while not queue_audio.empty():
                x_aud += queue_audio.get().tolist()
            x_aud = x_aud[:32000]
            
            x_sei = []
            while not queue_seismic.empty():
                x_sei.append(queue_seismic.get())
            x_sei = x_sei[:200]

            input_data = prepare_data_for_deepsense(x_aud, x_sei)
            
            # Simple model
            seismic= torch.flatten(input_data['shake']['seismic']).numpy()
            acoustic = torch.flatten(input_data['shake']['audio']).numpy()
            
            
            X_acoustic= acoustic
            X_seismic= seismic

            X_test = createFeatures(X_acoustic,X_seismic)
            pred= model.predict(X_test)[0]
            pred2= model.predict_proba(X_test)[0]
            
            print("label: ", LABELS[pred]," , time cost: ", time.time()-start_time)    
            #logits = classifier(input_data)
            #pred = int(logits.argmax(dim=1, keepdim=False)[0].numpy())
            #print("label: ", LABELS[pred], ", confidence: ", logits[0][pred].detach().numpy(), ", time cost: ", time.time()-start_time)

            n += 1

    # When ctrl+c is received
    except KeyboardInterrupt as e:
        sys.exit(e)


if __name__ == "__main__":
    main()