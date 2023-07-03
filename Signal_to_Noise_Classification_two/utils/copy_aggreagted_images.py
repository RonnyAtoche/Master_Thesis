import os
import shutil
import pandas as pd

# Level 1

data_testSet_folder_name = ['snapnum_50_57','snapnum_58_66','snapnum_67_77']


for data_testSet_folder_name_item in data_testSet_folder_name:

   catalogue_path = '/data/s2614855/Signal_to_Noise_Classification_two/utils/S_to_N_labeled_raw_images_{0}.csv'.format(data_testSet_folder_name_item)

   origin_parent_image = '/data/s2614855/Signal_to_Noise_Classification_two/labeled_raw_images_resized/{0}/'.format(data_testSet_folder_name_item)

   target_parent_parent_image = '/data/s2614855/Signal_to_Noise_Classification_two/labeled_raw_image_resized_aggregated/{0}/'.format(data_testSet_folder_name_item)

   bin_names = [0,5,10,15,20,25,30,35,40,45]

   for bin_value in bin_names:

      target_parent_bin_image = os.path.join(target_parent_parent_image, 'major_than_{0}'.format(bin_value))
      os.mkdir(target_parent_bin_image)

      target_parent_bin_image_real = target_parent_parent_image + 'major_than_{0}/'.format(bin_value)

      df_catalogue = pd.read_csv(catalogue_path)
      files = os.listdir(origin_parent_image)

      for file_name in files:
         print('processing: {0}'.format(file_name))
         single_row = df_catalogue[df_catalogue['file_name']==file_name]

         try:
            single_row_S_to_N_value = float(single_row['signal_to_noise'])
            bin_value_float = float(bin_value)

            if (single_row_S_to_N_value > bin_value_float):

               shutil.copy(origin_parent_image+file_name, target_parent_bin_image_real+file_name)
               print('Copying_processing: {0}'.format(file_name))
            else:
               continue
         except:
            continue