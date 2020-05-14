
# https://www.researchgate.net/publication/301648615_End_to_End_Learning_for_Self-Driving_Cars
# Load data
import glob, os
from PIL import Image
import re

#import pandas as pd
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
#from scipy import stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, DepthwiseConv2D, MaxPool2D, Input
from tensorflow.keras import regularizers
from tensorflow.keras import Model

def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

# def cnn_layers(model_input):
#     # Normalise
#     #x = model_input / 255.0
#     #x = keras.layers.Lambda(lambda x_input: x_input/255.0)(model_input)
#
#     # Conv layer 1
#     x = keras.layers.Conv2D(24, (5, 5), strides=(2, 2), padding='same', data_format="Channels_first",
#                             activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
#                             bias_initializer='zeros', kernel_regularizer=regularizers.l2(1e-5))(model_input)
#     # Conv layer 2
#     x = keras.layers.Conv2D(36, (5, 5), strides=(2, 2), padding='same', data_format="Channels_first",
#                             activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
#                             bias_initializer='zeros', kernel_regularizer=regularizers.l2(1e-5))(x)
#
#     # Conv layer 3
#     x = keras.layers.Conv2D(48, (3, 3), strides=(1, 1), padding='same', data_format="Channels_first", dilation_rate=(1, 1),
#                             activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
#                             bias_initializer='zeros', kernel_regularizer=regularizers.l2(1e-5))(x)
#     # Conv layer 4
#     x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', data_format="Channels_first", dilation_rate=(1, 1),
#                             activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
#                             bias_initializer='zeros', kernel_regularizer=regularizers.l2(1e-5))(x)
#     # Fully connected
#     x = Flatten()(x)
#     x = Dense(100, activation='relu')(x)
#     x = Dense(50, activation='relu')(x)
#     x = Dense(10, activation='relu')(x)
#     prediction = Dense(1, activation='softmax')(x)
#     return prediction


input_depth = 3
input_height = 66
input_width = 200
num_channels = 3

batch_size = 32

kernel_size = 60
depth = 60
num_hidden = 1000

learning_rate = 0.0001
training_epochs = 5

# Float64 by default in layers
tf.keras.backend.set_floatx('float64')


X_Data = np.load('X_Data.npy')/255.0
Y_Data = np.load('Y_Data.npy')[:, np.newaxis]/180.0


train_x = X_Data[1:6000,:,:,:]
test_x =  X_Data[6001:,:,:,:]
train_y = Y_Data[1:6000]
test_y =  Y_Data[6001:]

# Sequencial (rather than functional)
model = keras.models.Sequential()
# Conv layer 1
model.add( keras.layers.Conv2D(24, (5, 5), strides=(2, 2), padding='valid', data_format="Channels_last",
                        activation='elu', use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', input_shape=(input_height,input_width,num_channels)))
# Conv layer 2
model.add(  keras.layers.Conv2D(36, (5, 5), strides=(2, 2), padding='valid', data_format="Channels_last",
                        activation='elu', use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros'))
# Conv layer 3
model.add( keras.layers.Conv2D(48, (5, 5), strides=(2, 2), padding='valid', data_format="Channels_last",
                        activation='elu', use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros', ))
# Conv layer 4
model.add( keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid', data_format="Channels_last",
                        activation='elu', use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros'))
# Conv layer 4
model.add( keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid', data_format="Channels_last",
                        activation='elu', use_bias=True, kernel_initializer='glorot_uniform',
                        bias_initializer='zeros'))
# Fully connected
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1, activation='tanh'))

model.summary()
model.layers

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Nadam(lr=learning_rate),
              metrics=[tf.keras.metrics.MeanSquaredError()])

history = model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=training_epochs,
          verbose=1,
          validation_data=(test_x, test_y))

score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test mean squared error:', score[1])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()