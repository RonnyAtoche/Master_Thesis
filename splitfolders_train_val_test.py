import splitfolders
import os
import shutil

list_split = ['train','val','test']
merger_list = ['nonmerger', 'merger']

for splititem in list_split:
  for mergeritem in merger_list:
    root_src_path = "/data/s2614855/Final_Analysis_v1/data/snapnum_50_57/train_model_data/train_val_test_split/{0}/{1}/".format(splititem,mergeritem)
    files = os.listdir(root_src_path)
    for file_name in files:
    
      root_dst_path = "/data/s2614855/Final_Analysis_v1/data/snapnum_50_57/train_model_data/train_val_test/{0}/".format(splititem)
      shutil.copy(root_src_path+file_name, root_dst_path+file_name)



# root_src_path = ""
# root_dst_path = "/data/s2614855/Final_Analysis_v1/data/snapnum_50_57/train_model_data/splitted_data/"

# files = os.listdir(root_src_path)
# for file_name in files:
#    if 'nonmerger' in file_name:
#       shutil.copy(root_src_path+file_name, root_dst_path+"nonmerger/"+file_name)
#    else:
#       shutil.copy(root_src_path+file_name, root_dst_path+"merger/"+file_name)   



# splitfolders.ratio('/data/s2614855/Final_Analysis_v1/data/snapnum_50_57/train_model_data/splitted_data', output="/data/s2614855/Final_Analysis_v1/data/snapnum_50_57/train_model_data/train_val_test", seed=1337, ratio=(.8, 0.1,0.1)) 