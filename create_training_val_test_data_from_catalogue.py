import os
import pandas as pd
import glob
import numpy as np


df = pd.read_csv('//scratch//s2614855//snapnum_78_91_cropped_320_to_224_normalized_log_and_0_1_train_val_test//catalog_merger_challenge_TNG_train.csv')


real_path = '/scratch/s2614855/snapnum_78_91_cropped_320_to_224_normalized_log_and_0_1/'
# destination_path = '/scratch/s2614855/snapnum_78_91_cropped_320_to_224_normalized_log_and_0_1/'


# for img_path in glob.glob(real_path + '*.fits'):
  
#   fit_file_name = img_path.split('/',4)[4]

#   fit_file_name_new = fit_file_name.replace('non','')
  
#   os.rename(img_path, destination_path+fit_file_name_new)

#   print(fit_file_name_new)




df['is_final_merger'] = np.where((df['is_ongoing_merger'] == 1) | (df['is_post_merger'] == 1), 1, 0)

df['Object_name'] = np.where(df['is_final_merger'] == 1, 'objID_' + df['ID'].apply(str) + '.fits', 'objID_' + df['ID'].apply(str) + '.fits')
df['Object_name_labeled'] = np.where(df['is_final_merger'] == 1, 'merger_objID_' + df['ID'].apply(str) + '.fits', 'nonmerger_objID_' + df['ID'].apply(str) + '.fits')

print(len(df))

list_file_names = []

for img_path in glob.glob(real_path + '*.fits'):
  
  fit_file_name = img_path.split('/',4)[4]

  list_file_names.append(fit_file_name)

print(len(list_file_names))

df_final = df[df['Object_name'].isin(list_file_names)]

df_final.to_csv('//home4//s2614855//experiment_4//raw_original_data_catalogue_images.csv', encoding='utf-8', index=False)