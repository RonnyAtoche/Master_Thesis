import shutil
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

source_dir = '/scratch/s2614855/JWST/output/fits_resized_normalized'
target_dir = '/scratch/s2614855/JWST/output/fits_resized_normalized_grouped/'

file_names = os.listdir(source_dir)

def divide_chunks(l, n):

    for i in range(0, len(l), n): 
        yield l[i:i + n]
  
n = 5000
  
x = list(divide_chunks(file_names, n))

for i in range(len(x)):

  list_to_iterate = x[i]

  for file_name in list_to_iterate:

    print('processing...{0}'.format(file_name))

    img_path = source_dir+'/{0}'.format(file_name)

    directory_name = "group_{0}".format(i)
      
    path = os.path.join(target_dir, directory_name)
    
    if not (os.path.isdir(path)):

      os.mkdir(path)
    
    shutil.copy(img_path, path + '/' + file_name)