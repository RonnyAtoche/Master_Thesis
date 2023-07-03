from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
#sys.path += ["/home2/s3744531/Thesis/python/merger/"]

import tensorflow_datasets as tfds
#import datasets
import pandas as pd
import matplotlib.pyplot as plt

import model_8 as model
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf



def train_and_evaluate(datasets_path, tfds_name, img_shape, epochs, batch_size, classes,output_dir, file_weights, file_model, file_history,
          plot_name, model_num, load_weights=False, filename_model='', data_augm=False):
    """Trains and evaluates the Keras model.

    Uses the Keras model defined in model.py and trains on data loaded and
    as tensorflow dataset. Saves the trained model in TensorFlow SavedModel
    format.

    """


    input_shape = img_shape[0]

    # Create the Keras Model   
    keras_model = model.create_model(input_dim = img_shape, load_weights=load_weights, filename_model=filename_model)


    #Load dataset
    (ds_train, ds_val, d_test), ds_info = tfds.load(tfds_name,
                                 data_dir=datasets_path,
                                 split=['train', 'val', 'test'],
                                 shuffle_files=True,
                                 as_supervised=True,
                                 with_info=True,
                                 )
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Setup callback.
    mc = tf.keras.callbacks.ModelCheckpoint(
        filepath = output_dir+'model.weights_model{}.hdf5'.format(model_num),
        save_weights_only = True,
        monitor = 'val_accuracy',
        mode = 'max',
        verbose = 1,
        save_best_only = True)
    my_callbacks = [mc]

    keras_model.summary()


    # Train model
    history = keras_model.fit(
        ds_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = ds_val,
        callbacks = my_callbacks)


    plot_training(history, output_dir, name=plot_name)
    save_model(keras_model, output_dir, history, model_num)


    return history


def plot_training(history, output_folder, name):

    # summarize history for accuracy
    fig, axs = plt.subplots(2, 1,figsize=(5,7))
    axs[0].plot(history.history['accuracy'],label='Train',color='blue')
    axs[0].plot(history.history['val_accuracy'],label='Validation',color='orange')
    axs[0].set_title('Model accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()

    # summarize history for loss
    axs[1].plot(history.history['loss'],label='Train',color='blue')
    axs[1].plot(history.history['val_loss'],label='Validation',color='orange')
    axs[1].set_title('Model loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()
    fig.tight_layout(pad=2.0)
    plt.savefig(output_folder+name+'.png')
    plt.close()

def save_model(model, output_dir, history, model_num):
    # Save data
    open(output_dir+'model_{}'.format(model_num), 'w').write(model.to_json())

    # Saving history for future use
    with open(output_dir+'history_model{}'.format(model_num), 'w') as f:
        json.dump(history.history, f)



if __name__ == '__main__':

    model_num = 1

    img_rows = 224
    img_cols = 224
    channels = 1
    img_shape = (channels, img_rows, img_cols)
    classes = ['nonmerger', 'merger']

    batch_size = 64
    epochs = 600

    tfds_name = 'my_dataset'
    datasets_path = '/scratch/s2614855/experiment_4/tensorflow_datasets/'  # path to training dataset
    output_dir = '/home4/s2614855/experiment_5/model_tasks/outputs/task_8/'  # path to save model and images

    load_weights = False
    filename_model = ''
    data_augm = True

    file_weights = 'weights_model{}_train1'.format(model_num)  # name for file with model weights
    file_model = 'model{}_train1'.format(model_num)  # name for file with model
    file_history = 'history_model{}_train1'.format(model_num)  # name for file with history
    plot_name = 'plot_history_model{}_train1'.format(model_num)  # name for file with training plots


    os.makedirs(output_dir, exist_ok=True)
    #plot_batch(output_dir, tfds_name)
    history = train_and_evaluate(datasets_path, tfds_name, img_shape, epochs, batch_size, classes, output_dir, file_weights, file_model, file_history,plot_name, model_num)


