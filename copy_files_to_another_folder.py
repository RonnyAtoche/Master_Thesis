import os
import shutil

origin = '/data/s2614855/Signal_to_Noise_Classification/images/fits_images_cropped_testSet_ClassifiedStoN_aggregated/major_20/'
target = '/data/s2614855/Signal_to_Noise_Classification/images/fits_images_cropped_testSet_ClassifiedStoN_aggregated/major_10/'

files = os.listdir(origin)

for file_name in files:
   shutil.copy(origin+file_name, target+file_name)