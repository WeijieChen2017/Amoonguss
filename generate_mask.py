from scipy.ndimage.morphology import binary_fill_holes
# from scipy.ndimage import generate_binary_structure
# from scipy.ndimage.morphology import binary_closing
from scipy.ndimage import gaussian_filter, binary_dilation, binary_fill_holes, label

import nibabel as nib
import numpy as np

import os
import glob

# search the MRI images

data_folder_brain = "./data_dir/Task1/brain/"
data_folder_pelvis = "./data_dir/Task1/pelvis/"
mri_paths_list = sorted(glob.glob(data_folder_brain+"*/mr.nii.gz")) + sorted(glob.glob(data_folder_pelvis+"*/mr.nii.gz"))

# mri_paths_list = sorted(glob.glob(data_folder_brain+"*/mr.nii.gz"))
# mri_paths_list = sorted(glob.glob(data_folder_pelvis+"*/mr.nii.gz"))

def generate_mask(mri_data, value_threshold, guassian_threshold, dilation_radius=3, blur_radius=2):
    # Load the MRI image 
    mri_img = mri_data

    # Apply threshold: create a binary mask where values greater than threshold are set to 1
    binary_mask = np.where(mri_img > value_threshold, 1, 0)

    # Dilate the filled mask
    dilated_mask = binary_dilation(binary_mask, iterations=dilation_radius)

    # Apply gaussian filter (blur)
    blurred_mask = gaussian_filter(np.float32(dilated_mask), sigma=blur_radius)

    # Re-threshold to keep the mask binary
    binary_mask = np.where(blurred_mask > guassian_threshold, 1, 0)  # Assuming values in the mask are 0 or 1

    # Apply thresholding again with the original image to ensure that all regions in the mask are above the threshold in the original image
    binary_mask = np.where((mri_img > value_threshold) & (binary_mask == 1), 1, 0)

    # For each axial slice, fill the holes and only keep the largest connected region
    filled_mask = np.zeros_like(binary_mask)
    for i in range(binary_mask.shape[2]):  # assuming axial direction is the third dimension
        slice_mask = binary_fill_holes(binary_mask[:,:,i])
        
        labeled_mask, num_labels = label(slice_mask)

        if num_labels > 0:  # only proceed if there are labels
            largest_label = np.argmax([np.sum(labeled_mask == j) for j in range(1, num_labels+1)]) + 1
            filled_mask[:,:,i] = (labeled_mask == largest_label)

    
    return filled_mask

for mri_path in mri_paths_list:
    print("mri_path: ", mri_path)
    mri_file = nib.load(mri_path)
    mri_data = mri_file.get_fdata()
    mri_mask = generate_mask(
        mri_data=mri_data, 
        value_threshold=60,
        guassian_threshold=0.9,
    )
    mask_file = nib.Nifti1Image(mri_mask, mri_file.affine, mri_file.header)
    mask_path = mri_path.replace("mr.nii.gz", "mask_mri_th60.nii.gz")
    nib.save(mask_file, mask_path)
    print("mask_path: ", mask_path)
