# DOCUMENTATION:https://sep.readthedocs.io/en/v1.1.x/tutorial.html

import numpy as np
import sep
import fitsio
import os
import shutil
import pandas as pd

############# Variables ###############
data_target_folder_name = 'snapnum_67_77'
image_size = 192
#######################################

image_size_center = int(image_size/2)

origin = '/data/s2614855/Signal_to_Noise_Classification_two/labeled_raw_images/{0}/'.format(data_target_folder_name)

files = os.listdir(origin)
df = pd.DataFrame(columns=['file_name','flux','flux_error','signal_to_noise','merger'])

for count, file_name in enumerate(files):

    data = fitsio.read(origin + file_name)

    bkg = sep.Background(data)

    objects = sep.extract(data, 1.5, err=bkg.globalrms)     

    centered_object_index = ''      

    for i in range(objects.shape[0]):

        if (objects[i][3] < image_size_center) and (objects[i][4] > image_size_center) and (objects[i][5] < image_size_center) and (objects[i][6] > image_size_center):
            centered_object_index = str(i)
            break
            
    centered_object_index_int = int(centered_object_index)

    flux, fluxerr, flag = sep.sum_circle(data, objects['x'], objects['y'],
                                        3.0, err=bkg.globalrms, gain=1.0)

    try:

        Signal_to_Noise = float(int(flux[centered_object_index_int])/int(fluxerr[centered_object_index_int]))

        if 'nonmerger' in file_name:
            merger_value = 0
        else:
            merger_value = 1

        df = df.append({'file_name':file_name,'flux':flux[centered_object_index_int],'flux_error':fluxerr[centered_object_index_int],'signal_to_noise':Signal_to_Noise, 'merger':merger_value}, ignore_index=True)
        
        print('Object_'+str(count)+'of_'+str(len(files))+':'+ file_name +'done!')

    except:
        continue

df.to_csv('/data/s2614855/Signal_to_Noise_Classification_two/utils/S_to_N_labeled_raw_images_{0}.csv'.format(data_target_folder_name),index=False)
   