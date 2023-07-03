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
import tensorflow as tf
import tensorflow_datasets as tfds
#from cf_matrix import make_confusion_matrix
import numpy as np
import model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics.cluster import completeness_score
import pandas as pd
from tensorflow import keras
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [Input(shape = (1, 224, 224))], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


tfds_name_predict = 'my_dataset' #Name of dataset
datasets_path = '/data/s2614855/Signal_to_Noise_Classification_two/tensorflow_data/snapnum_50_57/tensorflow_datasets/' #Path to dataset
filename_model = '/data/s2614855/Signal_to_Noise_Classification_two/predictions/snapnum_50_57/model_image/model.weights_model1.hdf5' #Path+name to model weights
input_dim = (1, 224, 224) #Image dimensions: (channels, rows, cols)
root_name_plot = 'confusion_matrix_' #Path+name to output plot with confusion matrix
output_folder = '/data/s2614855/Signal_to_Noise_Classification_two/predictions/snapnum_50_57/model_image/'

def predict(data, input_dim, filename_model):


    keras_model = model.create_model(input_dim = input_dim, load_weights=True, filename_model=filename_model)


    y_pred = keras_model.predict(data)
    keras_model.evaluate(data)

    return y_pred
 

(major_than_0, major_than_5, major_than_10, major_than_15, major_than_20, major_than_25, major_than_30, major_than_35, major_than_40, major_than_45) = tfds.load(tfds_name_predict,
                                 data_dir=datasets_path,
                                 split=['major_than_0', 'major_than_5', 'major_than_10', 'major_than_15', 'major_than_20', 'major_than_25', 'major_than_30', 'major_than_35', 'major_than_40', 'major_than_45'],
                                 shuffle_files=False,
                                 as_supervised=True,
                                 )

test_dataset = major_than_25.take(1) 

for images, labels in test_dataset:  # only take first element of dataset
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()

numpy_images_squeezed = np.squeeze(numpy_images)

last_conv_layer_name = 'my_conv2'

heatmap = make_gradcam_heatmap(numpy_images_squeezed, model, last_conv_layer_name, pred_index=260)

plt.matshow(heatmap)
plt.savefig('/data/s2614855/Signal_to_Noise_Classification_two/predictions/snapnum_50_57/heatmap.png')

# save_and_display_gradcam(img_path, heatmap)
# print(test_dataset)
# y_pred = predict(test_dataset, input_dim, filename_model) 