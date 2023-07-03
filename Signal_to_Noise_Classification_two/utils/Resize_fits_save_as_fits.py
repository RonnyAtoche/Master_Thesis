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
import sep
import fitsio
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
from skimage import data
from skimage.transform import resize, resize_local_mean

data_target_folder_name = ['snapnum_50_57','snapnum_58_66','snapnum_67_77']

for data_target_folder_name_item in data_target_folder_name:
  ############# Variables ###############
  size = (224, 224)
  position = (112, 112)
  #######################################

  source_dir = '/data/s2614855/Signal_to_Noise_Classification_two/labeled_raw_images/{0}'.format(data_target_folder_name_item)
  target_dir = '/data/s2614855/Signal_to_Noise_Classification_two/labeled_raw_images_resized/{0}'.format(data_target_folder_name_item)
      
  file_names = os.listdir(source_dir)
      
  for file_name in file_names:

    print('processing:{0}/{1}'.format(source_dir, file_name))
    
    img_path = source_dir+'/{0}'.format(file_name)

    fits_data = fits.open(img_path, ignore_missing_simple=True)

    hdu = fits_data[0]

    image = hdu.data
    new_img = resize_local_mean(image, (224, 224))


    wcs = WCS(fits_data[0].header)

    # Make the cutout, including the WCS
    cutout = Cutout2D(new_img, position=position, size=size, wcs=wcs)

    # Put the cutout image in the FITS HDU
    hdu.data = cutout.data

    # Update the FITS header with the cutout WCS
    hdu.header.update(cutout.wcs.to_header())

    # Write the cutout to a new FITS file
    cutout_filename = target_dir+'/{0}'.format(file_name)
    hdu.writeto(cutout_filename, overwrite=True)

    print('processing ready of:{0}/{1}'.format(source_dir, file_name))
    # #############################



