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

            logits = classifier(input_data)
            pred = int(logits.argmax(dim=1, keepdim=False)[0].numpy())
            print("label: ", LABELS[pred], ", confidence: ", logits[0][pred].detach().numpy(), ", time cost: ", time.time()-start_time)

            n += 1

    # When ctrl+c is received
    except KeyboardInterrupt as e:
        sys.exit(e)


if __name__ == "__main__":
    main()