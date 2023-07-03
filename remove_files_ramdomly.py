import os
import pandas as pd
import glob
import numpy as np
import random

real_path = '/scratch/s2614855/experiment_4/snapnum_78_91_cropped_320_to_224_normalized_log_and_0_1_merger_nonmerger/nonmerger/'

total_list = glob.glob(real_path + '*.fits')

remove_list = random.sample(total_list, 82900)

for remove_item in remove_list:
  os.remove(remove_item)