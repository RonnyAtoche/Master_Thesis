import os

# folder path
dir_path = '/data/s2614855/Intro_Project_Data/Intro_Project/snapnum_78_91_train_val_test/test'
count = 0
# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        count += 1
print('File count:', count)