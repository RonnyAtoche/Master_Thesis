# import splitfolders

# splitfolders.ratio("/data/s2614855/Intro_Project_Data/Intro_Project/snapnum_78_91", output="/data/s2614855/Intro_Project_Data/Intro_Project/snapnum_78_91_split",
#     seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False) # default values
#####################################

import os
import pandas as pd
import glob

path = '/data/s2614855/Intro_Project_Data/Intro_Project/snapnum_78_91/'
new_path_merger = '/data/s2614855/Intro_Project_Data/Intro_Project/snapnum_78_91_split/merger/'
new_path_non_merger = '/data/s2614855/Intro_Project_Data/Intro_Project/snapnum_78_91_split/non_merger/'  
df = pd.read_csv('/home/s2614855/Projects/intro_project/catalog_merger_challenge_TNG_train.csv')

# def _validate_label(self, img_path_var):
#     """Returns True or False wether the file path name correspond to a merger or not"""
#     df = pd.read_csv('/home/s2614855/Projects/intro_project/catalog_merger_challenge_TNG_train.csv')
#     fit_file_name = img_path_var.split('/',6)[6]
#     ObjectId_name = fit_file_name.replace('_',' ').replace('.',' ').split()[1]
#     ObjectId_number = int(ObjectId_name)

#     df_loc = df[df['ID']==ObjectId_number]

#     if ((df_loc['is_major_merger'].values[0]) == 1):
#         return True
#     else:
#         return False

for img_path in glob.glob(path+'*.fits'):
    # print(img_path) ##>>> /data/s2614855/Intro_Project_Data/Intro_Project/snapnum_78_91/objID_466174.fits
    fit_file_name = img_path.split('/',6)[6]
    ObjectId_name = fit_file_name.replace('_',' ').replace('.',' ').split()[1]
    ObjectId_number = int(ObjectId_name)

    df_loc = df[df['ID']==ObjectId_number]

    if ((df_loc['is_major_merger'].values[0]) == 1):
        os.rename(img_path, new_path_merger+fit_file_name)
    elif ((df_loc['is_major_merger'].values[0]) == 0):
        os.rename(img_path, new_path_non_merger+fit_file_name) 
