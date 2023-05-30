import glob
import os
import json
import numpy as np
import nibabel as nib

data_dir = "./data_dir/Task1/brain/"
folder_list = sorted(glob.glob(data_dir+"*/mr.nii.gz"))
case_id_list = []
for folder_path in folder_list:
    case_id = folder_path.split("/")[-2]
    # print(case_id)
    case_id_list.append(case_id)
n_case_id = len(case_id_list)

# create an excel file to store the metrics
import xlsxwriter

workbook = xlsxwriter.Workbook('./results/brain_v3b_last.xlsx')
worksheet_mae = workbook.add_worksheet("MAE")
# the first row, the header, bold
bold = workbook.add_format({'bold': True})
worksheet_mae.write('A1', 'Case ID', bold)
worksheet_mae.write('B1', 'v3a_best', bold)
worksheet_mae.write('C1', 'v3a_last', bold)
worksheet_mae.write('D1', 'v3b_best', bold)
worksheet_mae.write('E1', 'v3b_last', bold)
worksheet_mae.write('F1', 'v3bq_best', bold)
worksheet_mae.write('G1', 'v3bq_last', bold)
worksheet_mae.write('H1', 'n_fold', bold)


def cal_mae_with_idx(sct_path, col_idx, worksheet_mae, idx_case, ct, mask_data):
    if os.path.exists(sct_path):
        sct_file = nib.load(sct_path)
        sct_data = sct_file.get_fdata()
        sct = np.clip(sct_data, -1024, 3000)
        masked_mae = np.sum(np.abs(ct * mask_data - sct * mask_data)) / np.sum(mask_data)
        worksheet_mae.write(idx_case+1, col_idx, masked_mae)
        diff_sct = np.abs(ct - sct) * mask_data
        diff_name = sct_path.replace(".nii.gz", "_diff.nii.gz")
        diff_file = nib.Nifti1Image(diff_sct, sct_file.affine, sct_file.header)
        nib.save(diff_file, diff_name)
        print("Saved", diff_name)


for idx_case in range(n_case_id):

    case_id = case_id_list[idx_case]
    print(case_id)
    worksheet_mae.write(idx_case+1, 0, case_id)

    ct_data = nib.load("./data_dir/Task1/brain/"+case_id+"/ct.nii.gz").get_fdata()
    mask_data = nib.load("./data_dir/Task1/brain/"+case_id+"/mask.nii.gz").get_fdata()
    ct = np.clip(ct_data, -1024, 3000)

    # v3a_best
    sct_path = "./data_dir/Task1/brain/"+case_id+"/sct_v3a_best.nii.gz"
    cal_mae_with_idx(sct_path, 1, worksheet_mae, idx_case, ct, mask_data)

    # v3a_last
    sct_path = "./data_dir/Task1/brain/"+case_id+"/sct_v3a_last.nii.gz"
    cal_mae_with_idx(sct_path, 2, worksheet_mae, idx_case, ct, mask_data)

    # v3b_best
    sct_path = "./data_dir/Task1/brain/"+case_id+"/sct_v3b_best.nii.gz"
    # chcek if the file exists
    if os.path.exists(sct_path):
        cal_mae_with_idx(sct_path, 3, worksheet_mae, idx_case, ct, mask_data)

    # v3b_last
    sct_path = "./data_dir/Task1/brain/"+case_id+"/sct_v3b_last.nii.gz"
    if os.path.exists(sct_path):
        cal_mae_with_idx(sct_path, 4, worksheet_mae, idx_case, ct, mask_data)

    # v3bq_best
    sct_path = "./data_dir/Task1/brain/"+case_id+"/sct_v3bq_best.nii.gz"
    if os.path.exists(sct_path):
        cal_mae_with_idx(sct_path, 5, worksheet_mae, idx_case, ct, mask_data)

    # v3bq_last
    sct_path = "./data_dir/Task1/brain/"+case_id+"/sct_v3bq_last.nii.gz"
    if os.path.exists(sct_path):
        cal_mae_with_idx(sct_path, 6, worksheet_mae, idx_case, ct, mask_data)

for idx_fold in range(6):
    json_file = "./project_dir/Quaxly_brain_v3b/fold_{:01d}.json".format(idx_fold+1)
    # read the json file
    with open(json_file) as f:
        data = json.load(f)
    
    val_list = data["validation"]
    for idx_json_file in range(len(val_list)):
        sample = val_list[idx_json_file]
        curr_id = sample["MR"].split("/")[-2]
        curr_row = case_id_list.index(curr_id)
        worksheet_mae.write(curr_row+1, 7, idx_fold+1)

# save the excel file
workbook.close()



    


