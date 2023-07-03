
import os
import pandas as pd
import glob

data_target_folder_name = 'snapnum_67_77'

raw_images_path = '/data/s2614855/Signal_to_Noise_Classification_two/raw_images/{0}/'.format(data_target_folder_name)
labeled_raw_images_path = '/data/s2614855/Signal_to_Noise_Classification_two/labeled_raw_images/{0}/'.format(data_target_folder_name)
 
df = pd.read_csv('/data/s2614855/Signal_to_Noise_Classification_two/utils/Catalogue_TNT_test.csv')

for each_img_path in glob.glob(raw_images_path+'*.fits'):

    fit_file_name = each_img_path.split('/',6)[6]
    ObjectId_name = fit_file_name.replace('_',' ').replace('.',' ').split()[1]
    ObjectId_number = int(ObjectId_name)
    df_loc = df[df['ID']==ObjectId_number]

    if ((df_loc['is_major_merger'].values[0]) == 1):
        os.rename(each_img_path, labeled_raw_images_path+'merger_'+fit_file_name)
    elif ((df_loc['is_major_merger'].values[0]) == 0):
        os.rename(each_img_path, labeled_raw_images_path+'nonmerger_'+fit_file_name) 
