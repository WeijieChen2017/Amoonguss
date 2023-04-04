import nibabel as nib
import numpy as np
import os
import glob

MR_list = sorted(glob.glob("./data/Task1/*/*/mr.nii.gz"))
CT_list = sorted(glob.glob("./data/Task1/*/*/ct.nii.gz"))

for folder_path in ["./data/t1_mr_norm/", "./data/t1_ct_norm/"]:
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

# MR_range_min = []
# MR_range_max = []
# MR_range_q999 = []

# CT_range_min = []
# CT_range_max = []
# CT_range_q999 = []

largest_z_MR = 0
largest_z_CT = 0

for file_path in MR_list:

    path_parts = file_path.split(os.sep)
    wildcard1, wildcard2 = path_parts[3], path_parts[4]
    new_filename = f"{wildcard1}_{wildcard2}_mr.nii.gz"
    print(new_filename, end=" -> ")

    # file_name = os.path.basename(file_path)
    # print(file_name, end=" -> ")
    MR_file = nib.load(file_path)
    MR_data = MR_file.get_fdata()
    print(MR_data.shape)
    if MR_data.shape[2] > largest_z_MR:
        largest_z_MR = MR_data.shape[2]
    # MR_mask = nib.load(file_path.replace("mr.nii.gz", "mask.nii.gz")).get_fdata()
    # MR_data = MR_data[MR_mask.astype(bool)]
    # data_min = np.amin(MR_data)
    # data_max = np.amax(MR_data)
    # data_q999 = np.percentile(MR_data, 99.9)
    # print(MR_data.shape, data_min, data_max, data_q999)
    # MR_range_min.append(data_min)
    # MR_range_max.append(data_max)
    # MR_range_q999.append(data_q999)
    # print(np.percentile(NAC_data, 99.9))
    # NAC_data = np.resize(NAC_data, (256, 256, NAC_data.shape[2]))
    # MR_data = MR_data / 3000
    # new_MR_file = nib.Nifti1Image(MR_data, MR_file.affine, MR_file.header)
    # new_save_name = "./data/t1_mr_norm/" + new_filename
    # nib.save(new_MR_file, new_save_name)

# np.save("./data/t1_mr_norm/MR_range.npy", [MR_range_min, MR_range_max, MR_range_q999])

for file_path in CT_list:

    path_parts = file_path.split(os.sep)
    wildcard1, wildcard2 = path_parts[3], path_parts[4]
    new_filename = f"{wildcard1}_{wildcard2}_ct.nii.gz"
    print(new_filename, end=" -> ")

    # file_name = os.path.basename(file_path)
    # print(file_name, end=" -> ")
    CT_file = nib.load(file_path)
    CT_data = CT_file.get_fdata()
    print(CT_data.shape)
    if CT_data.shape[2] > largest_z_CT:
        largest_z_CT = CT_data.shape[2]
    # CT_mask = nib.load(file_path.replace("ct.nii.gz", "mask.nii.gz")).get_fdata()
    # CT_data = CT_data[CT_mask.astype(bool)]
    # data_min = np.amin(CT_data)
    # data_max = np.amax(CT_data)
    # data_q999 = np.percentile(CT_data, 99.9)
    # print(CT_data.shape, data_min, data_max, data_q999)
    # CT_range_min.append(data_min)
    # CT_range_max.append(data_max)
    # CT_range_q999.append(data_q999)
    # print(np.amax(CT_data), np.amin(CT_data)) [2000.0, 0.0]
    # CT_data = np.resize(CT_data, (256, 256, CT_data.shape[2]))
    # CT_data = (CT_data + 1024) / (3000+1024)
    # new_CT_file = nib.Nifti1Image(CT_data, CT_file.affine, CT_file.header)
    # new_save_name = "./data/t1_ct_norm/" + new_filename
    # nib.save(new_CT_file, new_save_name)
    # print(new_filename, "saved")

# np.save("./data/t1_ct_norm/CT_range.npy", [CT_range_min, CT_range_max, CT_range_q999])

print(largest_z_MR, largest_z_CT)