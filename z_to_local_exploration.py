# Get details about the trein data into stats

import os
import re
import pandas as pd
import numpy as np

#### TRAIN COMPLETE DATASET ######



df = pd.read_csv("/data/s2614855/Final_Analysis_v1/data/snapnum_78_91_test/catalog_Horizon_AGN_snapnum_78_91.csv")
df['file_name'] = np.where(df['is_major_merger']  == 1,'merger_objID_'+df['ID'].apply(str)  + '.fits', 'nonmerger_objID_'+df['ID'].apply(str)  + '.fits')

path = "/data/s2614855/Final_Analysis_v1/data/snapnum_78_91_test/labeled_crop_30_percent_resized_to_224_normalized_log_and_0_1/snapnum_78_91_test"
dir_list = os.listdir(path)

df2 = pd.read_csv("/data/s2614855/Final_Analysis_v1/data/snapnum_78_91_test/S_to_N_snapnum_78_91_test.csv")

print(len(dir_list))
print(df.count())

df = df[df['file_name'].isin(dir_list)]
print(df.count())
print(df.head())
print(df2.count())
print(df2.head())

df_merged = pd.merge(df,df2,how='inner',left_on=['file_name'],right_on=['file_name'])
print(df_merged.count())
print(df_merged.head())

df_merged.to_csv('/data/s2614855/Final_Analysis_v1/data/snapnum_78_91_test/S_to_N_redshift_filename_snapnum_78_91_test.csv',index=False)



### TESTS SETS ONLY ####

# mean_object = 'snapnum_67_77'

# df = pd.read_csv("/data/s2614855/Final_Analysis_v1/utils/shared/Catalogue_TNT_test_sets.csv")
# df['file_name'] = np.where(df['is_major_merger']  == 1,'merger_objID_'+df['ID'].apply(str)  + '.fits', 'nonmerger_objID_'+df['ID'].apply(str)  + '.fits')

# path = "/data/s2614855/Final_Analysis_v1/data/{0}/labeled_crop_30_percent_resized_to_224_normalized_log_and_0_1/{0}/".format(mean_object)
# dir_list = os.listdir(path)

# df2 = pd.read_csv("/data/s2614855/Final_Analysis_v1/data/{0}/S_to_N_{0}.csv".format(mean_object))

# print(len(dir_list))
# print(df.count())

# df = df[df['file_name'].isin(dir_list)]
# print(df.count())
# print(df.head())
# print(df2.count())
# print(df2.head())

# df_merged = pd.merge(df,df2,how='inner',left_on=['file_name'],right_on=['file_name'])
# print(df_merged.count())
# print(df_merged.head())

# df_merged.to_csv('/data/s2614855/Final_Analysis_v1/stats/{0}/S_to_N_redshift_filename_{0}.csv'.format(mean_object),index=False)