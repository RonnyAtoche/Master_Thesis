import tensorflow as tf
import tensorflow_datasets as tfds
#from cf_matrix import make_confusion_matrix
import numpy as np
import model_8 as model
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
#import matplotlib.pyplot as plt
#from mlxtend.plotting import plot_confusion_matrix
#from sklearn.metrics.cluster import completeness_score
import pandas as pd

tfds_name_predict = 'my_dataset' #Name of dataset
datasets_path = '/scratch/s2614855/Experiment_2/tensorflow_datasets_second/' #Path to dataset
filename_model = '/home4/s2614855/experiment_5/model_tasks/outputs/task_8/model.weights_model1.hdf5' #Path+name to model weights
input_dim = (1, 224, 224) #Image dimensions: (channels, rows, cols)
#root_name_plot = 'confusion_matrix_' #Path+name to output plot with confusion matrix
#output_folder = '/home4/s2614855/Documents/UDS_Analysis/predictions_1/output/'

def predict(data, input_dim, filename_model):

    keras_model = model.create_model(input_dim = input_dim, load_weights=True, filename_model=filename_model)

    y_pred = keras_model.predict(data)
    keras_model.evaluate(data)

    return y_pred

fitsresizednormalizedtestsecond = tfds.load(tfds_name_predict,
                                 data_dir=datasets_path,
                                 split=['fitsresizednormalizedtestsecond'],
                                 shuffle_files=False,
                                 as_supervised=True,
                                 )

df = pd.DataFrame(columns=['Item_Index','Merger_Value'])


fitsresizednormalizedtestsecond=np.array(fitsresizednormalizedtestsecond)

test_dataset = fitsresizednormalizedtestsecond.take(-1).batch(64).prefetch(tf.data.experimental.AUTOTUNE) 

y_pred = predict(test_dataset, input_dim, filename_model) 

# predicted_labels =  (y_pred > 0.5).astype(np.intc)
predicted_labels =  y_pred

for item in range(len(predicted_labels)):
    df = df.append({'Item_Index':item,'Merger_Value':predicted_labels[item][0]}, ignore_index=True)

df.to_csv('/home4/s2614855/FinalAnalysis_2/make_predictions_cropped_local_images/fitsresizednormalizedtestsecond_predicted_values_dataset_raw.csv',index=False)

# true_labels = tf.concat([y for x, y in test_dataset], axis=0).numpy()
# cm = confusion_matrix(true_labels, predicted_labels)
# fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(4, 4), show_absolute=True, show_normed=True, colorbar=True)
# plt.savefig(output_folder + root_name_plot + 'major_than_0' + '.png')
# accuracy  = np.trace(cm) / float(np.sum(cm))
# reliabilty = cm[1,1] / (cm[1,1] + cm[0,1])
# completeness    = cm[1,1] / (cm[1,1] + cm[1,0])
# f1_score  = 2*reliabilty*completeness / (reliabilty + completeness)
# stats_text = "\n\nAccuracy={:0.3f}\nReliabilty={:0.3f}\nCompleteness={:0.3f}\nF1 Score={:0.3f}".format(
#     accuracy,reliabilty,completeness,f1_score)
# df = df.append({'major_than_name':'major_than_0','accuracy':accuracy,'reliability':reliabilty,'completeness':completeness,'f_one_score':f1_score}, ignore_index=True)


# df.to_csv('{0}stats_{1}.csv'.format(output_folder,main_test_set_name),index=False)

# reliabilty = precision
# completeness = recall
