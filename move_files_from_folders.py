import splitfolders
import os
import glob
import shutil

real_path = '/scratch/s2614855/experiment_4/snapnum_78_91_cropped_320_to_224_normalized_log_and_0_1_train_val_test/'

list_values = ['test/','train/','val/']
merger_values = ['merger/','nonmerger/']

for list_values_item in list_values:

  second_path = real_path + list_values_item

  for merger_values_item in merger_values:

    third_path = second_path + merger_values_item

    for img_internal_path in glob.glob(third_path + '*.fits'):

      fit_file_name = img_internal_path.split('/',7)[7]

      os.rename(img_internal_path, second_path + fit_file_name)