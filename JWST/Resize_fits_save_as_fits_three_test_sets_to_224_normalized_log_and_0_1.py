# Download an example FITS file, create a 2D cutout, and save it to a
# new FITS file, including the updated cutout WCS.
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.utils.data import download_file
from astropy.wcs import WCS
#import tensorflow_datasets as tfds
from astropy.io import fits
#import tensorflow as tf
import glob
import numpy as np
import pandas as pd
import shutil
import os
import sys
import sep
#import fitsio
#import matplotlib.pyplot as plt
#from matplotlib import rcParams
#from matplotlib.patches import Ellipse
from skimage import data
from skimage.transform import resize, resize_local_mean

def zoom_in(numpy_array_image, zoom_factor):

    h, w = numpy_array_image.shape
    h_value = int((h-h*zoom_factor)/2)
    w_value = int((w-w*zoom_factor)/2)

    numpy_array_image_zoomed = numpy_array_image[h_value:-(h_value), w_value:-(w_value)]

    return numpy_array_image_zoomed

############# Variables ###############
size = (224, 224)
position = (112, 112)
#######################################

source_dir = '/scratch/s2614855/JWST/output/fits'
target_dir = '/scratch/s2614855/JWST/output/fits_resized_normalized'

corrupted_images = []    

file_names = os.listdir(source_dir)
#countitem= 0


def divide_chunks(l, n):
      
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]
  
# How many elements each
# list should have
n = 1000
  
x = list(divide_chunks(file_names, n))

sub_list_item = int(sys.argv[1])

list_to_iterate = x[sub_list_item]

for file_name in list_to_iterate:

  #print(countitem)
  #print('processing:{0}/{1}'.format(source_dir, file_name))
  
  img_path = source_dir+'/{0}'.format(file_name)

  try:
    
    fits_data = fits.open(img_path, ignore_missing_simple=True)

    hdu = fits_data[0]

    image = hdu.data

    image_cropped = zoom_in(image, 0.7)

    new_img = resize_local_mean(image_cropped, (224, 224))

    if 0 in new_img:

      corrupted_images.append(img_path)

      continue
    
    else:

      new_img = np.log10(abs(new_img))

      # 0-1 Normalization
      max_val = np.max(new_img)
      min_val = np.min(new_img)
      new_img = (new_img - max_val) / (max_val - min_val) +1

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

      #print('processing ready of:{0}/{1}'.format(source_dir, file_name))
    #countitem += 1

  except:
    continue

with open("/scratch/s2614855/JWST/corrupted_images_resize_process.txt", "w") as output:
  output.write(str(corrupted_images))
  # #############################



