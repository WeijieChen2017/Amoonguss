from scipy.ndimage.morphology import binary_fill_holes
# from scipy.ndimage import generate_binary_structure
# from scipy.ndimage.morphology import binary_closing
from scipy.ndimage import gaussian_filter, binary_dilation, binary_fill_holes

import nibabel as nib
import numpy as np

import os
import glob

# search the MRI images

data_folder = "./data_dir/Task1/brain/"
mri_paths_list = sorted(glob.glob(data_folder+"*/mr.nii.gz"))

def generate_mask(mri_data, value_threshold, guassian_threshold, dilation_radius=3, blur_radius=2):
    # Load the MRI image 
    mri_img = mri_data

    # Apply threshold: create a binary mask where values greater than threshold are set to 1
    binary_mask = np.where(mri_img > value_threshold, 1, 0)

    # For each axial slice, fill the holes
    filled_mask = np.zeros_like(binary_mask)
    for i in range(binary_mask.shape[2]):  # assuming axial direction is the third dimension
        filled_mask[:,:,i] = binary_fill_holes(binary_mask[:,:,i])

    # Dilate the filled mask
    dilated_mask = binary_dilation(filled_mask, iterations=dilation_radius)

    # Apply gaussian filter (blur)
    blurred_mask = gaussian_filter(np.float32(dilated_mask), sigma=blur_radius)

    # Re-threshold to keep the mask binary
    final_mask = np.where(blurred_mask > guassian_threshold, 1, 0)  # Assuming values in the mask are 0 or 1

    return final_mask

for mri_path in mri_paths_list:
    print("mri_path: ", mri_path)
    mri_file = nib.load(mri_path)
    mri_data = mri_file.get_fdata()
    mri_mask = generate_mask(
        mri_data=mri_data, 
        value_threshold=50,
        guassian_threshold=0.9,
    )
    mask_file = nib.Nifti1Image(mri_mask, mri_file.affine, mri_file.header)
    mask_path = mri_path.replace("mr.nii.gz", "mask_mri_th50.nii.gz")
    nib.save(mask_file, mask_path)
    print("mask_path: ", mask_path)
