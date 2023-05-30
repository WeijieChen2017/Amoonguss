import glob
import os
import numpy as np
import nibabel as nib

data_dir = "./data_dir/Task1/brain/"
folder_list = sorted(glob.glob(data_dir+"*/mr.nii.gz"))
for folder_path in folder_list:
    print(folder_list)
    case_id = folder_path.split("/")[-2]
    print(case_id)