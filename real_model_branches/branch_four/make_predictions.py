import tensorflow as tf
import tensorflow_datasets as tfds
#from cf_matrix import make_confusion_matrix
import numpy as np
import model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix



tfds_name_predict = 'my_dataset' #Name of dataset
datasets_path = '/data/s2614855/Intro_Project_Data/Intro_Project/tensorflow_dataset_cropped_images_second/tensorflow_datasets/' #Path to dataset
filename_model = '/data/s2614855/Intro_Project_Data/Intro_Project/real_model_branches/branch_four/model_image/model.weights_model1.hdf5' #Path+name to model weights
input_dim = (1, 128, 128) #Image dimensions: (channels, rows, cols)
name_plot = 'confusion_matrix' #Path+name to output plot with confusion matrix
output_folder = '/data/s2614855/Intro_Project_Data/Intro_Project/real_model_branches/branch_four/model_image/'

def predict(data, input_dim, filename_model):


    keras_model = model.create_model(input_dim = input_dim, load_weights=True, filename_model=filename_model)


    y_pred = keras_model.predict(data)
    keras_model.evaluate(data)

    return y_pred


(ds_train, ds_val, ds_test) = tfds.load(tfds_name_predict,
                                 data_dir=datasets_path,
                                 split=['train', 'val', 'test'],
                                 shuffle_files=False,
                                 as_supervised=True,
                                 )

#test_dataset = ds_test.take(-1).batch(cf.batch_size).prefetch(tf.data.experimental.AUTOTUNE)    
test_dataset = ds_test.take(-1).batch(64).prefetch(tf.data.experimental.AUTOTUNE) 


#Predictions: array of predictions, values between 0 and 1
#y_pred = predict(test_dataset, filename_model, model_num, n_dense, archite) 
y_pred = predict(test_dataset, input_dim, filename_model) 

 
#Predictions converted to 0 or 1
predicted_labels =  (y_pred > 0.5).astype(np.float32)


true_labels = tf.concat([y for x, y in test_dataset], axis=0).numpy()


#Confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)


fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.savefig(output_folder+name_plot+'.png')

accuracy  = np.trace(cm) / float(np.sum(cm))

precision = cm[0,0] / sum(cm[0,:])
recall    = cm[0,0] / sum(cm[:,0])
f1_score  = 2*precision*recall / (precision + recall)
stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
    accuracy,precision,recall,f1_score)
print('stats:', stats_text)


##################
# cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])

# cm_display.plot()
# plt.savefig(output_folder+name_plot+'.png')





###################
# make_confusion_matrix(cm, name_plot,
#                       categories=class_names, 
#                       cmap='Blues', tipo=tipo, title=tipo)



   