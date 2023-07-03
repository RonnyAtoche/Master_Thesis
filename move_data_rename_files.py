import os
import pandas as pd
import glob


# path = '/scratch/s2614855/snapnum_78_91_cropped_320_to_224_normalized_log_and_0_1_train_val_test/'
# destination_path = '/scratch/s2614855/snapnum_78_91_cropped_320_to_224_normalized_log_and_0_1/'

# list_group = ['test/','val/','train/']

# df = pd.read_csv('//scratch//s2614855//snapnum_78_91_cropped_320_to_224_normalized_log_and_0_1_train_val_test//catalog_merger_challenge_TNG_train.csv')

# for list_item in list_group:

#   real_path = path + list_item

#   for img_path in glob.glob(real_path + '*.fits'):
    
#     fit_file_name = img_path.split('/',5)[5]

#     fit_file_name_new = fit_file_name.replace('nonmerger_','')
#     fit_file_name_new = fit_file_name_new.replace('merger_','')
    
#     os.rename(img_path, destination_path+fit_file_name_new)

#     print(fit_file_name_new)


real_path = '/scratch/s2614855/snapnum_78_91_cropped_320_to_224_normalized_log_and_0_1/'
destination_path = '/scratch/s2614855/snapnum_78_91_cropped_320_to_224_normalized_log_and_0_1/'

# df = pd.read_csv('//scratch//s2614855//snapnum_78_91_cropped_320_to_224_normalized_log_and_0_1_train_val_test//catalog_merger_challenge_TNG_train.csv')

for img_path in glob.glob(real_path + '*.fits'):
  
  fit_file_name = img_path.split('/',4)[4]

  fit_file_name_new = fit_file_name.replace('non','')
  
  os.rename(img_path, destination_path+fit_file_name_new)

  print(fit_file_name_new)