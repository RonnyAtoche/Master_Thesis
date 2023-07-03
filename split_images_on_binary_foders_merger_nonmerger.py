import os
import pandas as pd
import glob
import numpy as np

df = pd.read_csv('//home4//s2614855//experiment_4//raw_original_data_catalogue_images.csv')

real_path = '/scratch/s2614855/snapnum_78_91_cropped_320_to_224_normalized_log_and_0_1/'

nonmerger_destination_path = '/scratch/s2614855/experiment_4/snapnum_78_91_cropped_320_to_224_normalized_log_and_0_1_merger_nonmerger/nonmerger/'
merger_destination_path = '/scratch/s2614855/experiment_4/snapnum_78_91_cropped_320_to_224_normalized_log_and_0_1_merger_nonmerger/merger/'

for img_path in glob.glob(real_path + '*.fits'):
  
  fit_file_name = img_path.split('/',4)[4]

  index_value = df.index[df['Object_name'] == fit_file_name].tolist()[0]

  df_filtered_single_row = df.filter(items = [index_value], axis=0)

  if df_filtered_single_row.iloc[0]['is_final_merger'] == 0:

    fit_file_name_new = df_filtered_single_row.iloc[0]['Object_name_labeled']

    os.rename(img_path, nonmerger_destination_path + fit_file_name_new)

  elif df_filtered_single_row.iloc[0]['is_final_merger'] == 1:

    fit_file_name_new = df_filtered_single_row.iloc[0]['Object_name_labeled']

    os.rename(img_path, merger_destination_path + fit_file_name_new)

  else:
    
    print('problem on:{0}'.format(fit_file_name))


# fit_file_name_new = fit_file_name.replace('non','')

# os.rename(img_path, destination_path+fit_file_name_new)

# print(fit_file_name_new)
# print(df_filtered_single_row.iloc[0]['data_type'])
# print(df_filtered_single_row.head())
# print(df.head())
# print(df_filtered_single_row.dtypes)