# Download an example FITS file, create a 2D cutout, and save it to a
# new FITS file, including the updated cutout WCS.
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.utils.data import download_file
from astropy.wcs import WCS
import tensorflow_datasets as tfds
from astropy.io import fits
import tensorflow as tf
import glob
import numpy as np
import pandas as pd

import shutil
import os
import cv2
    

############# Variables ###############
data_target_folder_name = 'snapnum_67_77'
image_size = 192
new_image_size = 192
#######################################

source_dir = '/data/s2614855/Signal_to_Noise_Classification_two/labeled_raw_images/{0}/'.format(data_target_folder_name)
target_dir = '/data/s2614855/Signal_to_Noise_Classification/images/fits_images_cropped_test_set'
    
file_names = os.listdir(source_dir)
    
# DOCUMENTATION:https://sep.readthedocs.io/en/v1.1.x/tutorial.html

import numpy as np
import sep

# additional setup for reading the test image and displaying plots
import fitsio
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
from skimage import data
from skimage.transform import resize, resize_local_mean

#%matplotlib inline
output_folder = '/data/s2614855/Signal_to_Noise_Classification_two/utils/image_TEST/'

rcParams['figure.figsize'] = [10., 8.]


  
# img_path = source_dir+'/{0}'.format(file_name)
img_path = "/data/s2614855/Signal_to_Noise_Classification_two/utils/image_TEST/merger_objID_504753.fits"

position = (112, 112)
size = (224, 224)
    # Download the image

# read image into standard 2-d numpy array

# data = fitsio.read("/data/s2614855/Signal_to_Noise_Classification_two/utils/image_TEST/merger_objID_504753.fits")
# print(data)
# print(data.shape)
# print(type(data))
# # show the image
# m, s = np.mean(data), np.std(data)
# plt.imshow(data, interpolation='nearest', cmap='gray', vmin=m-s, vmax=m+s, origin='lower')
# plt.colorbar()
# plt.savefig(output_folder+'original_image'+'.png')




fits_data = fits.open(img_path, ignore_missing_simple=True)

# if len(img.shape) < 3:
#     img = np.expand_dims(img, axis=0)
# img = img.astype('float32')

# Load the image and the WCS
hdu = fits_data[0]


print(hdu.data)
print(hdu.data.shape)
print(type(hdu.data))
# show the image
m, s = np.mean(hdu.data), np.std(hdu.data)

plt.imshow(hdu.data, interpolation='nearest', cmap='gray', vmin=m-s, vmax=m+s, origin='lower')
plt.colorbar()
plt.savefig(output_folder+'original_image'+'.png')
print(fits_data[0].header)


image = hdu.data
new_img = resize_local_mean(image, (224, 224))

plt.imshow(new_img, interpolation='nearest', cmap='gray', vmin=m-s, vmax=m+s, origin='lower')
plt.colorbar()
plt.savefig(output_folder+'resized_image'+'.png')
print(fits_data[0].header)


wcs = WCS(fits_data[0].header)

# Make the cutout, including the WCS
cutout = Cutout2D(new_img, position=position, size=size, wcs=wcs)

# Put the cutout image in the FITS HDU
hdu.data = cutout.data

# Update the FITS header with the cutout WCS
hdu.header.update(cutout.wcs.to_header())

# Write the cutout to a new FITS file
cutout_filename = output_folder+'/{0}'.format("merger_objID_504753_resized.fits")
hdu.writeto(cutout_filename, overwrite=True)

# #############################