# DOCUMENTATION:https://sep.readthedocs.io/en/v1.1.x/tutorial.html

import numpy as np
import sep
# import fitsio
from astropy.io import fits
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from skimage.transform import resize, resize_local_mean
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import sys
# pip install scikit-image

def download_image_save_cutout(image_path, img_seq, position, size):
    
  # Load the image and the WCS
  hdu = fits.open(image_path)[0]
  wcs = WCS(hdu.header)

  # Make the cutout, including the WCS
  cutout = Cutout2D(hdu.data, position=position, size=size, wcs=wcs)

  # Put the cutout image in the FITS HDU
  hdu.data = cutout.data

  # Update the FITS header with the cutout WCS
  hdu.header.update(cutout.wcs.to_header())

  # Write the cutout to a new FITS file
  cutout_filename = '/scratch/s2614855/JWST/output/fits/ObjID_{0}.fits'.format(img_seq)

  hdu.writeto(cutout_filename, overwrite=True)
  
  plt.imshow(hdu.data, origin='lower')
  plt.savefig('/scratch/s2614855/JWST/output/png/ObjID_{0}.png'.format(img_seq))


dict_slice_values = [
  {
    'start':1,
    'end':1000
  },
  {
    'start':1000,
    'end':2000
  },
  {
    'start':2000,
    'end':3000
  },
  {
    'start':3000,
    'end':4000
  },
  {
    'start':4000,
    'end':5000
  },
  {
    'start':5000,
    'end':6000
  },
  {
    'start':6000,
    'end':7000
  },
  {
    'start':7000,
    'end':8000
  },
  {
    'start':8000,
    'end':9000
  },
  {
    'start':9000,
    'end':10000
  },
  {
    'start':10000,
    'end':11000
  },
  {
    'start':11000,
    'end':12000
  },
  {
    'start':12000,
    'end':13000
  },
  {
    'start':13000,
    'end':14000
  },
  {
    'start':14000,
    'end':15000
  },
  {
    'start':15000,
    'end':16000
  },
  {
    'start':16000,
    'end':17000
  },
  {
    'start':17000,
    'end':18000
  },
  {
    'start':18000,
    'end':19000
  },
  {
    'start':19000,
    'end':20000
  },
  {
    'start':20000,
    'end':21000
  },
  {
    'start':21000,
    'end':22000
  },
  {
    'start':22000,
    'end':23000
  },
  {
    'start':23000,
    'end':24000
  },
  {
    'start':24000,
    'end':25000
  },
  {
    'start':25000,
    'end':26000
  },
  {
    'start':26000,
    'end':27000
  },
  {
    'start':27000,
    'end':28000
  },
  {
    'start':28000,
    'end':29000
  },
  {
    'start':29000,
    'end':30000
  },
  {
    'start':30000,
    'end':31463
  }
]

group_int_value = int(sys.argv[1])

origin = "/scratch/s2614855/JWST/PRIMER_UDS_F444W_sci.fits"

match_df = pd.read_csv("/scratch/s2614855/JWST/JWST_SEX_catalogue.csv")

start_value = dict_slice_values[group_int_value]['start']
end_value = dict_slice_values[group_int_value]['end']

for image_id in range(start_value,end_value):
    
  try:

    match_df_filtered = match_df[match_df['ID']==image_id]

    x_position = match_df_filtered.iloc[0]['X']
    y_position = match_df_filtered.iloc[0]['Y']
    
    major_axis = match_df_filtered.iloc[0]['A_IMAGE']

    final_lateral_size = 22*major_axis

    position = (x_position,y_position)
    size = (final_lateral_size,final_lateral_size)

    download_image_save_cutout(origin, image_id, position, size)
      
  except:
    continue
        
        
        
        
