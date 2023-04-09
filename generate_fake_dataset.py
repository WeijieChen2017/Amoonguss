# read everything from the original dataset and generate a fake dataset
# with the same format as the original dataset
# the original dataset is stored in the folder 
# CT "./data_dir/Task11_SynthRad_task1/t1_ct_norm/*.nii.gz"
# MR "./data_dir/Task11_SynthRad_task1/t1_mr_norm/*.nii.gz"

import os
import numpy as np
import nibabel as nib
import glob

CT_list = sorted(glob.glob("./data_dir/Task11_SynthRad_task1/t1_ct_norm/*.nii.gz"))
MR_list = sorted(glob.glob("./data_dir/Task11_SynthRad_task1/t1_mr_norm/*.nii.gz"))

fake_dataset_folder = "./data_dir/Task11_SynthRad_task1_fake/"
if not os.path.exists(fake_dataset_folder):
    os.mkdir(fake_dataset_folder)

fake_mr_folder = fake_dataset_folder + "t1_mr_norm/"
fake_ct_folder = fake_dataset_folder + "t1_ct_norm/"
if not os.path.exists(fake_mr_folder):
    os.mkdir(fake_mr_folder)
if not os.path.exists(fake_ct_folder):
    os.mkdir(fake_ct_folder)

for file_path in MR_list:
    file_name = os.path.basename(file_path)
    print(file_name, end=" -> ")
    MR_file = nib.load(file_path)
    MR_data = MR_file.get_fdata()
    print(MR_data.shape)
    new_MR_data = np.zeros(MR_data.shape)
    new_MR_file = nib.Nifti1Image(new_MR_data, MR_file.affine, MR_file.header)
    new_save_name = fake_mr_folder + file_name
    nib.save(new_MR_file, new_save_name)

for file_path in CT_list:
    file_name = os.path.basename(file_path)
    print(file_name, end=" -> ")
    CT_file = nib.load(file_path)
    CT_data = CT_file.get_fdata()
    print(CT_data.shape)
    new_CT_data = np.zeros(CT_data.shape)
    new_CT_file = nib.Nifti1Image(new_CT_data, CT_file.affine, CT_file.header)
    new_save_name = fake_ct_folder + file_name
    nib.save(new_CT_file, new_save_name)
