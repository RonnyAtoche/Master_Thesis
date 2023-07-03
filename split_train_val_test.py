import splitfolders
import os
import shutil

splitfolders.ratio('/scratch/s2614855/experiment_4/snapnum_78_91_cropped_320_to_224_normalized_log_and_0_1_merger_nonmerger', output="/scratch/s2614855/experiment_4/snapnum_78_91_cropped_320_to_224_normalized_log_and_0_1_train_val_test", seed=1337, ratio=(.8, 0.1,0.1)) 