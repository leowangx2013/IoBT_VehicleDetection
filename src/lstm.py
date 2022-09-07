import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv1D,Conv2D,LSTM,Dense

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

''' 64%
def model_LSTM_frequency(input_shape=[256, 5]):
    dr=0.3

    input = Input(input_shape, name='concat_input')

    x = tf.transpose(input, (0, 2, 1))
    x = tf.signal.stft(x, frame_length=32, frame_step=16, fft_length=32)
    x = tf.transpose(x, (0, 2, 3, 1))
        
    x = tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)
    x = tf.gather(x, indices=[0,2,4,6,8,1,3,5,7,9], axis=-1)

    x1, x2 = tf.split(x, [6, 4], axis=-1)

    x1 = Conv2D(8, 3, padding='same', activation='relu')(x1) # [batch, time, freq, feature]
    x1 = tf.keras.layers.Dropout(dr)(x1)
    x2 = Conv2D(8, 3, padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.Dropout(dr)(x2)

    x = tf.concat([x1, x2], axis=-1)
    x = Conv2D(8, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Dropout(dr)(x)
    # x = tf.reshape(x, (-1, 15, 17*8))
    x = tf.reshape(x, (-1, x.shape[1], x.shape[2]*x.shape[3]))

    x = LSTM(units=32,return_sequences=True,name="fusion_LSTM1",)(x)
    x = tf.keras.layers.Dropout(dr)(x)
    x = LSTM(units=32,return_sequences=True,name="fusion_LSTM2")(x)
    x = tf.keras.layers.Dropout(dr)(x)

    x = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last', name='global_max_pooling1d')(x)

    model=Model(inputs=input,outputs=x)
    
    return model    
'''

def model_LSTM_frequency(input_shape=[256, 5]):
    dr=0.3
    r=1e-3
    input = Input(input_shape, name='concat_input')
    
    x = tf.transpose(input, (0, 2, 1))
    x = tf.signal.stft(x, frame_length=256, frame_step=64, fft_length=256)
    x = tf.transpose(x, (0, 2, 3, 1))
        
    x = tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)
    x = tf.gather(x, indices=[0,2,4,6,8,1,3,5,7,9], axis=-1)

    x1, x2 = tf.split(x, [6, 4], axis=-1)

    x1 = Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=r) , name='acoustic_conv')(x1) # [batch, time, freq, feature]
    x1 = tf.keras.layers.Dropout(dr)(x1)
    x2 = Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=r), name='seismic_conv')(x2)
    x2 = tf.keras.layers.Dropout(dr)(x2)

    x = tf.concat([x1, x2], axis=-1)

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