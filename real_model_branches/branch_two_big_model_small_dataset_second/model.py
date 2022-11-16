import tensorflow as tf
from tensorflow.keras.layers import Input, LeakyReLU
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
#from tensorflow.python.keras import models
from functools import partial
import h5py
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
#from utils import load_dataset

import gc
#import psutil
from tensorflow.keras import initializers
#from tensorflow.python.client import device_lib

import tensorflow.keras.backend as K
K.set_image_data_format('channels_first')

import numpy as np
import os
import time



def create_model(input_dim, load_weights=False, filename_model=''):
  """Creates a Keras model with CNN layers.

  Returns:
    A keras.Model
  """

  inputs = Input(shape = input_dim)        # Image input (real sample)

  x = inputs

  x = Conv2D(32, (9, 9), strides=(1, 1), padding='same', data_format='channels_first', kernel_initializer=initializers.RandomNormal(stddev=0.02))(x)
  x = LeakyReLU(0.2)(x)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)
  x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x)

  x = Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
  x = LeakyReLU(0.2)(x)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

  x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
  x = LeakyReLU(0.2)(x)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)

  x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
  x = LeakyReLU(0.2)(x)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)


  x = Flatten()(x)
  x = Dense(64, activation='relu',kernel_regularizer=l2(0.01))(x)
  x = Dense(32, activation='relu',kernel_regularizer=l2(0.01))(x)
  y = Dense(1, activation='sigmoid')(x)

  model = Model(inputs=inputs, outputs=y)

  # Compile Model
  learning_rate = 0.001
  optimizer = 'adam'
  metrics = ['accuracy']
  loss = 'binary_crossentropy'
  model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  model.summary()

  if load_weights:
      print('Loading weights')
      print(filename_model)
      model.load_weights(filename_model)

  return model