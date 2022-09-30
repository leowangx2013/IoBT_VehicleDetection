import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv1D,Conv2D,LSTM,Dense
from src.feature_extractors_arl import *
import tensorflow_probability as tfp

# The structure of the encoder
def model_LSTM(input_shape1=[256,5],classes=9):
    dr=0.3
    r=1e-4

    input1=Input(input_shape1,name='I/Qchannel')
    x = input1

    x = Conv1D(32, 24,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)
    x = LSTM(units=128,return_sequences=True,name="LSTM1",kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)
    x = tf.keras.layers.Dropout(dr)(x)
    x = LSTM(units=128,return_sequences=True,name="LSTM2",kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)
    x = tf.keras.layers.Dropout(dr)(x)
    x = Conv1D(128, 8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=r))(x)
    x = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last', name='global_max_pooling1d')(x)
    model=Model(inputs=input1,outputs=x)
    
    return model

def model_LSTM_frequency(input_shape=[256, 5]):
    dr=0.3
    r=1e-3
    input = Input(input_shape, name='concat_input')
    
    #input: <KerasTensor: shape=(None, 1024, 5) dtype=float32 (created by layer 'concat_input')>
    x = tf.transpose(input, (0, 2, 1))

    """
    # window energy
    e = tf.reduce_sum(tf.square(x), axis=2, keepdims=True)
    
    # correlation
    auto_correlation= tfp.stats.auto_correlation(
    x[0,:,:],
    axis=-1,
    max_lags=None,
    center=True,
    normalize=True,
    name='auto_correlation')
    
    # correlation of windows with each other in x
    corr = tfp.stats.correlation(x, x, sample_axis=0, event_axis=None)

    # windowed_mean 
    """

    x = tf.signal.stft(x, frame_length=256, frame_step=64, fft_length=256)

    x =mfcc(x) # try mfcc run
    # x: <KerasTensor: shape=(None, 5, 13, 129) dtype=complex64 (created by layer 'tf.signal.stft')>
    x = tf.transpose(x, (0, 2, 3, 1))
    """
    # x = <KerasTensor: shape=(None, 13, 129, 5) dtype=complex64 (created by layer 'tf.compat.v1.transpose_1')>
    x = tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)
    # x = <KerasTensor: shape=(None, 13, 129, 10) dtype=float32 (created by layer 'tf.concat')>
    x = tf.gather(x, indices=[0,2,4,6,8,1,3,5,7,9], axis=-1)

    x1, x2 = tf.split(x, [6, 4], axis=-1)

    x1 = Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=r) , name='acoustic_conv')(x1) # [batch, time, freq, feature]
    x1 = tf.keras.layers.Dropout(dr)(x1)
    x2 = Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=r), name='seismic_conv')(x2)
    x2 = tf.keras.layers.Dropout(dr)(x2)

    x = tf.concat([x1, x2], axis=-1)
    """
    x = Conv2D(64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=r), activation='relu')(x)
    x = tf.keras.layers.Dropout(dr)(x)
    x = tf.reshape(x, (-1, x.shape[1], x.shape[2]*x.shape[3]))

    x = LSTM(units=128,return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l=r),name="fusion_LSTM1",)(x)
    x = tf.keras.layers.Dropout(dr)(x)
    x = LSTM(units=128,return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l=r),name="fusion_LSTM2")(x)
    x = tf.keras.layers.Dropout(dr)(x)

    x = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last', name='global_max_pooling1d')(x)

    model=Model(inputs=input,outputs=x)
    
    return model   

def model_vanilla():
    input = Input([256,5], name='concat_input')

    x = tf.transpose(input, (0, 2, 1))
    x = tf.abs(tf.signal.stft(x, frame_length=16, frame_step=16, fft_length=16))
    x = tf.transpose(x, (0, 2, 3, 1))
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)

    # [batch, time, freq, feature]
    x = tf.keras.layers.GlobalMaxPool2D(data_format='channels_last', name='global_max_pooling2d')(x)
    
    model = Model(inputs=input, outputs=x)
    return model