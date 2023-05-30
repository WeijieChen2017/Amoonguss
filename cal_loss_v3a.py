import glob
import os
# import numpy as np
# import nibabel as nib

data_dir = "./data_dir/Task1/brain/"
folder_list = sorted(glob.glob(data_dir+"*/mr.nii.gz"))
case_id_list = []
for folder_path in folder_list:
    case_id = folder_path.split("/")[-2]
    print(case_id)
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
worksheet_mae.write('F1', 'v3b_last', bold)
worksheet_mae.write('G1', 'v3bq_best', bold)
worksheet_mae.write('H1', 'v3bq_last', bold)
worksheet_mae.write('I1', 'n_fold', bold)

for idx_case in range(n_case_id):
    


