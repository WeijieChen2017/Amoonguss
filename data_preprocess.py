import nibabel as nib
import numpy as np
import os
import glob

MR_list = sorted(glob.glob("./data/Task1/*/*/mr.nii.gz"))
CT_list = sorted(glob.glob("./data/Task1/*/*/ct.nii.gz"))

for folder_path in ["./data/t1_mr_norm/", "./data/t1_ct_norm/"]:
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

MR_range_min = []
MR_range_max = []
MR_range_q999 = []

CT_range_min = []
CT_range_max = []
CT_range_q999 = []

for file_path in MR_list:
    file_name = os.path.basename(file_path)
    print(file_name, end=" -> ")
    MR_file = nib.load(file_path)
    MR_data = MR_file.get_fdata()
    MR_mask = nib.load(file_path.replace("mr.nii.gz", "mask.nii.gz")).get_fdata()
    print(MR_data)
    MR_data = MR_data[MR_mask]
    data_min = np.amin(MR_data)
    data_max = np.amax(MR_data)
    data_q999 = np.percentile(MR_data, 99.9)
    print(MR_data.shape, data_min, data_max, data_q999)
    MR_range_min.append(data_min)
    MR_range_max.append(data_max)
    MR_range_q999.append(data_q999)
    # print(np.percentile(NAC_data, 99.9))
    # NAC_data = np.resize(NAC_data, (256, 256, NAC_data.shape[2]))
    # NAC_data = NAC_data / 10000
    # new_NAC_file = nib.Nifti1Image(NAC_data, NAC_file.affine, NAC_file.header)
    # new_save_name = "./data/NAC_wb_norm/" + file_name + ".gz"
    # nib.save(new_NAC_file, new_save_name)

np.save("./data/t1_mr_norm/MR_range.npy", [MR_range_min, MR_range_max, MR_range_q999])

for file_path in CT_list:
    file_name = os.path.basename(file_path)
    print(file_name, end=" -> ")
    CT_file = nib.load(file_path)
    CT_data = CT_file.get_fdata()
    CT_mask = nib.load(file_path.replace("ct.nii.gz", "mask.nii.gz")).get_fdata()
    CT_data = CT_data[CT_mask]
    data_min = np.amin(CT_data)
    data_max = np.amax(CT_data)
    data_q999 = np.percentile(CT_data, 99.9)
    print(CT_data.shape, data_min, data_max, data_q999)
    CT_range_min.append(data_min)
    CT_range_max.append(data_max)
    CT_range_q999.append(data_q999)
    # print(np.amax(CT_data), np.amin(CT_data)) [2000.0, 0.0]
    # CT_data = np.resize(CT_data, (256, 256, CT_data.shape[2]))
    # CT_data = CT_data / 2000
    # new_CT_file = nib.Nifti1Image(CT_data, CT_file.affine, CT_file.header)
    # new_save_name = "./data/CT_wb_norm/" + file_name + ".gz"
    # nib.save(new_CT_file, new_save_name)

np.save("./data/t1_ct_norm/CT_range.npy", [CT_range_min, CT_range_max, CT_range_q999])