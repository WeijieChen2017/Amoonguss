import numpy as np
import nibabel as nib
from scipy.ndimage import binary_fill_holes

import os
import glob

# search the MRI images

data_folder = "./data_dir/Task1/brain/"
mri_paths_list = sorted(glob.glob(data_folder+"*/mr.nii.gz"))

def generate_mask(mri_data, threshold):
    # Load the MRI image 
    mri_img = mri_data

    # Apply threshold: create a binary mask where values greater than threshold are set to 1
    binary_mask = np.where(mri_img > threshold, 1, 0)

    # For each axial slice, fill the holes
    filled_mask = np.zeros_like(binary_mask)
    for i in range(binary_mask.shape[2]):  # assuming axial direction is the third dimension
        filled_mask[:,:,i] = binary_fill_holes(binary_mask[:,:,i])

    return filled_mask

for mri_path in mri_paths_list:
    print("mri_path: ", mri_path)
    mri_file = nib.load(mri_path)
    mri_data = mri_file.get_fdata()
    mri_mask = generate_mask(mri_data, 50)
    mask_file = nib.Nifti1Image(mri_mask, mri_file.affine, mri_file.header)
    mask_path = mri_path.replace("mr.nii.gz", "mask_mri_th50.nii.gz")
    nib.save(mask_file, mask_path)
    print("mask_path: ", mask_path)
