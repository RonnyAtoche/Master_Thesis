# from tensorflow import keras
# #model = keras.models.load_model('/data/s2614855/Intro_Project_Data/Intro_Project/real_model_branches/branch_two_big_model_small_dataset_test/model_image/')

# with open("/data/s2614855/Intro_Project_Data/Intro_Project/real_model_branches/branch_two_big_model_small_dataset_test/model_image/model_1", "r") as f:
#   loaded_json_string = f.read()

# new_model = keras.models.model_from_json(loaded_json_string)

# print(new_model.summary())

import json
import os
import sys
#sys.path += ["/home2/s3744531/Thesis/python/merger/"]

import tensorflow_datasets as tfds
#import datasets
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import model
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np

tfds_name = 'my_dataset'
datasets_path = '/data/s2614855/Intro_Project_Data/Intro_Project/tensorflow_dataset_cropped_images/tensorflow_datasets/'  # path to training dataset
output_dir = '/data/s2614855/Intro_Project_Data/Intro_Project/real_model_branches/branch_two_big_model_small_dataset_test/model_image/'  # path to save model and images


keras_model = model.create_model(input_dim =(1,224,224), load_weights=False, filename_model='')

keras_model.load_weights("/data/s2614855/Intro_Project_Data/Intro_Project/real_model_branches/branch_two_big_model_small_dataset_test/model_image/model.weights_model1.hdf5")

img_path = '/data/s2614855/Intro_Project_Data/Intro_Project/snapnum_78_91_train_val_test/test/merger_objID_26781.fits'

def zoom_in(numpy_array_image, zoom_factor):

    h, w = numpy_array_image.shape
    h_value = int((h-h*zoom_factor)/2)
    w_value = int((w-w*zoom_factor)/2)

    numpy_array_image_zoomed = numpy_array_image[h_value:-(h_value), w_value:-(w_value)]

    return numpy_array_image_zoomed

img = fits.open(img_path, ignore_missing_simple=True)[0].data
img = zoom_in(img, 0.7)
if len(img.shape) < 3:
    img = np.expand_dims(img, axis=0)
img = img.astype('float32')


predictions = keras_model.predict(img)
score = tf.nn.softmax(predictions[0])
print(predictions)
print(score)
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )