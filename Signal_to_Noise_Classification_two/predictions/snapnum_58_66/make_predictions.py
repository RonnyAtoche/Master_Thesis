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


tfds_name_predict = 'my_dataset' #Name of dataset
datasets_path = '/data/s2614855/Signal_to_Noise_Classification_two/tensorflow_data/snapnum_58_66/tensorflow_datasets/' #Path to dataset
filename_model = '/data/s2614855/Signal_to_Noise_Classification_two/predictions/snapnum_50_57/model_image/model.weights_model1.hdf5' #Path+name to model weights
input_dim = (1, 224, 224) #Image dimensions: (channels, rows, cols)
root_name_plot = 'confusion_matrix_' #Path+name to output plot with confusion matrix
output_folder = '/data/s2614855/Signal_to_Noise_Classification_two/predictions/snapnum_58_66/model_image/'

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



df = pd.DataFrame(columns=['major_than_name','accuracy','reliability','completeness','f_one_score'])




test_dataset = major_than_0.take(-1).batch(64).prefetch(tf.data.experimental.AUTOTUNE) 
y_pred = predict(test_dataset, input_dim, filename_model) 
predicted_labels =  (y_pred > 0.5).astype(np.intc)
true_labels = tf.concat([y for x, y in test_dataset], axis=0).numpy()
cm = confusion_matrix(true_labels, predicted_labels)
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(4, 4), show_absolute=True, show_normed=True, colorbar=True)
plt.savefig(output_folder + root_name_plot + 'major_than_0' + '.png')
accuracy  = np.trace(cm) / float(np.sum(cm))
reliabilty = cm[1,1] / (cm[1,1] + cm[0,1])
completeness    = cm[1,1] / (cm[1,1] + cm[1,0])
f1_score  = 2*reliabilty*completeness / (reliabilty + completeness)
stats_text = "\n\nAccuracy={:0.3f}\nReliabilty={:0.3f}\nCompleteness={:0.3f}\nF1 Score={:0.3f}".format(
    accuracy,reliabilty,completeness,f1_score)
df = df.append({'major_than_name':'major_than_0','accuracy':accuracy,'reliability':reliabilty,'completeness':completeness,'f_one_score':f1_score}, ignore_index=True)

test_dataset = major_than_5.take(-1).batch(64).prefetch(tf.data.experimental.AUTOTUNE) 
y_pred = predict(test_dataset, input_dim, filename_model) 
predicted_labels =  (y_pred > 0.5).astype(np.intc)
true_labels = tf.concat([y for x, y in test_dataset], axis=0).numpy()
cm = confusion_matrix(true_labels, predicted_labels)
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(4, 4), show_absolute=True, show_normed=True, colorbar=True)
plt.savefig(output_folder + root_name_plot + 'major_than_5' + '.png')
accuracy  = np.trace(cm) / float(np.sum(cm))
reliabilty = cm[1,1] / (cm[1,1] + cm[0,1])
completeness    = cm[1,1] / (cm[1,1] + cm[1,0])
f1_score  = 2*reliabilty*completeness / (reliabilty + completeness)
stats_text = "\n\nAccuracy={:0.3f}\nReliabilty={:0.3f}\nCompleteness={:0.3f}\nF1 Score={:0.3f}".format(
    accuracy,reliabilty,completeness,f1_score)
df = df.append({'major_than_name':'major_than_5','accuracy':accuracy,'reliability':reliabilty,'completeness':completeness,'f_one_score':f1_score}, ignore_index=True)

test_dataset = major_than_10.take(-1).batch(64).prefetch(tf.data.experimental.AUTOTUNE) 
y_pred = predict(test_dataset, input_dim, filename_model) 
predicted_labels =  (y_pred > 0.5).astype(np.intc)
true_labels = tf.concat([y for x, y in test_dataset], axis=0).numpy()
cm = confusion_matrix(true_labels, predicted_labels)
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(4, 4), show_absolute=True, show_normed=True, colorbar=True)
plt.savefig(output_folder + root_name_plot + 'major_than_10' + '.png')
accuracy  = np.trace(cm) / float(np.sum(cm))
reliabilty = cm[1,1] / (cm[1,1] + cm[0,1])
completeness    = cm[1,1] / (cm[1,1] + cm[1,0])
f1_score  = 2*reliabilty*completeness / (reliabilty + completeness)
stats_text = "\n\nAccuracy={:0.3f}\nReliabilty={:0.3f}\nCompleteness={:0.3f}\nF1 Score={:0.3f}".format(
    accuracy,reliabilty,completeness,f1_score)
df = df.append({'major_than_name':'major_than_10','accuracy':accuracy,'reliability':reliabilty,'completeness':completeness,'f_one_score':f1_score}, ignore_index=True)

test_dataset = major_than_15.take(-1).batch(64).prefetch(tf.data.experimental.AUTOTUNE) 
y_pred = predict(test_dataset, input_dim, filename_model) 
predicted_labels =  (y_pred > 0.5).astype(np.intc)
true_labels = tf.concat([y for x, y in test_dataset], axis=0).numpy()
cm = confusion_matrix(true_labels, predicted_labels)
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(4, 4), show_absolute=True, show_normed=True, colorbar=True)
plt.savefig(output_folder + root_name_plot + 'major_than_15' + '.png')
accuracy  = np.trace(cm) / float(np.sum(cm))
reliabilty = cm[1,1] / (cm[1,1] + cm[0,1])
completeness    = cm[1,1] / (cm[1,1] + cm[1,0])
f1_score  = 2*reliabilty*completeness / (reliabilty + completeness)
stats_text = "\n\nAccuracy={:0.3f}\nReliabilty={:0.3f}\nCompleteness={:0.3f}\nF1 Score={:0.3f}".format(
    accuracy,reliabilty,completeness,f1_score)
df = df.append({'major_than_name':'major_than_15','accuracy':accuracy,'reliability':reliabilty,'completeness':completeness,'f_one_score':f1_score}, ignore_index=True)

test_dataset = major_than_20.take(-1).batch(64).prefetch(tf.data.experimental.AUTOTUNE) 
y_pred = predict(test_dataset, input_dim, filename_model) 
predicted_labels =  (y_pred > 0.5).astype(np.intc)
true_labels = tf.concat([y for x, y in test_dataset], axis=0).numpy()
cm = confusion_matrix(true_labels, predicted_labels)
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(4, 4), show_absolute=True, show_normed=True, colorbar=True)
plt.savefig(output_folder + root_name_plot + 'major_than_20' + '.png')
accuracy  = np.trace(cm) / float(np.sum(cm))
reliabilty = cm[1,1] / (cm[1,1] + cm[0,1])
completeness    = cm[1,1] / (cm[1,1] + cm[1,0])
f1_score  = 2*reliabilty*completeness / (reliabilty + completeness)
stats_text = "\n\nAccuracy={:0.3f}\nReliabilty={:0.3f}\nCompleteness={:0.3f}\nF1 Score={:0.3f}".format(
    accuracy,reliabilty,completeness,f1_score)
df = df.append({'major_than_name':'major_than_20','accuracy':accuracy,'reliability':reliabilty,'completeness':completeness,'f_one_score':f1_score}, ignore_index=True)

test_dataset = major_than_25.take(-1).batch(64).prefetch(tf.data.experimental.AUTOTUNE) 
y_pred = predict(test_dataset, input_dim, filename_model) 
predicted_labels =  (y_pred > 0.5).astype(np.intc)
true_labels = tf.concat([y for x, y in test_dataset], axis=0).numpy()
cm = confusion_matrix(true_labels, predicted_labels)
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(4, 4), show_absolute=True, show_normed=True, colorbar=True)
plt.savefig(output_folder + root_name_plot + 'major_than_25' + '.png')
accuracy  = np.trace(cm) / float(np.sum(cm))
reliabilty = cm[1,1] / (cm[1,1] + cm[0,1])
completeness    = cm[1,1] / (cm[1,1] + cm[1,0])
f1_score  = 2*reliabilty*completeness / (reliabilty + completeness)
stats_text = "\n\nAccuracy={:0.3f}\nReliabilty={:0.3f}\nCompleteness={:0.3f}\nF1 Score={:0.3f}".format(
    accuracy,reliabilty,completeness,f1_score)
df = df.append({'major_than_name':'major_than_25','accuracy':accuracy,'reliability':reliabilty,'completeness':completeness,'f_one_score':f1_score}, ignore_index=True)

test_dataset = major_than_30.take(-1).batch(64).prefetch(tf.data.experimental.AUTOTUNE) 
y_pred = predict(test_dataset, input_dim, filename_model) 
predicted_labels =  (y_pred > 0.5).astype(np.intc)
true_labels = tf.concat([y for x, y in test_dataset], axis=0).numpy()
cm = confusion_matrix(true_labels, predicted_labels)
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(4, 4), show_absolute=True, show_normed=True, colorbar=True)
plt.savefig(output_folder + root_name_plot + 'major_than_30' + '.png')
accuracy  = np.trace(cm) / float(np.sum(cm))
reliabilty = cm[1,1] / (cm[1,1] + cm[0,1])
completeness    = cm[1,1] / (cm[1,1] + cm[1,0])
f1_score  = 2*reliabilty*completeness / (reliabilty + completeness)
stats_text = "\n\nAccuracy={:0.3f}\nReliabilty={:0.3f}\nCompleteness={:0.3f}\nF1 Score={:0.3f}".format(
    accuracy,reliabilty,completeness,f1_score)
df = df.append({'major_than_name':'major_than_30','accuracy':accuracy,'reliability':reliabilty,'completeness':completeness,'f_one_score':f1_score}, ignore_index=True)

test_dataset = major_than_35.take(-1).batch(64).prefetch(tf.data.experimental.AUTOTUNE) 
y_pred = predict(test_dataset, input_dim, filename_model) 
predicted_labels =  (y_pred > 0.5).astype(np.intc)
true_labels = tf.concat([y for x, y in test_dataset], axis=0).numpy()
cm = confusion_matrix(true_labels, predicted_labels)
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(4, 4), show_absolute=True, show_normed=True, colorbar=True)
plt.savefig(output_folder + root_name_plot + 'major_than_35' + '.png')
accuracy  = np.trace(cm) / float(np.sum(cm))
reliabilty = cm[1,1] / (cm[1,1] + cm[0,1])
completeness    = cm[1,1] / (cm[1,1] + cm[1,0])
f1_score  = 2*reliabilty*completeness / (reliabilty + completeness)
stats_text = "\n\nAccuracy={:0.3f}\nReliabilty={:0.3f}\nCompleteness={:0.3f}\nF1 Score={:0.3f}".format(
    accuracy,reliabilty,completeness,f1_score)
df = df.append({'major_than_name':'major_than_35','accuracy':accuracy,'reliability':reliabilty,'completeness':completeness,'f_one_score':f1_score}, ignore_index=True)

test_dataset = major_than_40.take(-1).batch(64).prefetch(tf.data.experimental.AUTOTUNE) 
y_pred = predict(test_dataset, input_dim, filename_model) 
predicted_labels =  (y_pred > 0.5).astype(np.intc)
true_labels = tf.concat([y for x, y in test_dataset], axis=0).numpy()
cm = confusion_matrix(true_labels, predicted_labels)
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(4, 4), show_absolute=True, show_normed=True, colorbar=True)
plt.savefig(output_folder + root_name_plot + 'major_than_40' + '.png')
accuracy  = np.trace(cm) / float(np.sum(cm))
reliabilty = cm[1,1] / (cm[1,1] + cm[0,1])
completeness    = cm[1,1] / (cm[1,1] + cm[1,0])
f1_score  = 2*reliabilty*completeness / (reliabilty + completeness)
stats_text = "\n\nAccuracy={:0.3f}\nReliabilty={:0.3f}\nCompleteness={:0.3f}\nF1 Score={:0.3f}".format(
    accuracy,reliabilty,completeness,f1_score)
df = df.append({'major_than_name':'major_than_40','accuracy':accuracy,'reliability':reliabilty,'completeness':completeness,'f_one_score':f1_score}, ignore_index=True)

test_dataset = major_than_45.take(-1).batch(64).prefetch(tf.data.experimental.AUTOTUNE) 
y_pred = predict(test_dataset, input_dim, filename_model) 
predicted_labels =  (y_pred > 0.5).astype(np.intc)
true_labels = tf.concat([y for x, y in test_dataset], axis=0).numpy()
cm = confusion_matrix(true_labels, predicted_labels)
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(4, 4), show_absolute=True, show_normed=True, colorbar=True)
plt.savefig(output_folder + root_name_plot + 'major_than_45' + '.png')
accuracy  = np.trace(cm) / float(np.sum(cm))
reliabilty = cm[1,1] / (cm[1,1] + cm[0,1])
completeness    = cm[1,1] / (cm[1,1] + cm[1,0])
f1_score  = 2*reliabilty*completeness / (reliabilty + completeness)
stats_text = "\n\nAccuracy={:0.3f}\nReliabilty={:0.3f}\nCompleteness={:0.3f}\nF1 Score={:0.3f}".format(
    accuracy,reliabilty,completeness,f1_score)
df = df.append({'major_than_name':'major_than_45','accuracy':accuracy,'reliability':reliabilty,'completeness':completeness,'f_one_score':f1_score}, ignore_index=True)





df.to_csv('/data/s2614855/Signal_to_Noise_Classification_two/predictions/snapnum_58_66/model_image/stats.csv',index=False)

# reliabilty = precision
# completeness = recall
