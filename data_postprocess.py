import numpy
import glob
import os

# the ct file is in ./data_dir/Task1/brain/1BA001/ct.nii.gz
# the predicted sct is in ./project_dir/Quaxly_brain_v3a/ensemble_last/brain_1BA001_sct.nii.gz

# sct_file_list = sorted(glob.glob("./project_dir/Quaxly_brain_v3b/eval_best/*_sct.nii.gz"))
# for sct_path in sct_file_list:
#     organ = sct_path.split("/")[-1].split("_")[0]
#     case_id = sct_path.split("/")[-1].split("_")[1]
#     # ct_path = "./data_dir/Task_1/"+organ+"/"+case_id+"/ct.nii.gz"
#     sct_dst = "./data_dir/Task1/"+organ+"/"+case_id+"/sct_v3b_best.nii.gz"
#     std_dst = "./data_dir/Task1/"+organ+"/"+case_id+"/std_v3b_best.nii.gz"
#     # std_path = "./project_dir/Quaxly_brain_v3a/ensemble_last/"+organ+"_"+case_id+"_std.nii.gz"
#     os.system("cp "+sct_path+" "+sct_dst)
#     print("Copied: ", sct_path, " to ", sct_dst)
#     # os.system("cp "+std_path+" "+std_dst)
#     # print("Copied: ", std_path, " to ", std_dst)

# sct_file_list = sorted(glob.glob("./project_dir/Quaxly_brain_v3b/eval_last/*_sct.nii.gz"))
# for sct_path in sct_file_list:
#     organ = sct_path.split("/")[-1].split("_")[0]
#     case_id = sct_path.split("/")[-1].split("_")[1]
#     # ct_path = "./data_dir/Task_1/"+organ+"/"+case_id+"/ct.nii.gz"
#     sct_dst = "./data_dir/Task1/"+organ+"/"+case_id+"/sct_v3b_last.nii.gz"
#     std_dst = "./data_dir/Task1/"+organ+"/"+case_id+"/std_v3b_last.nii.gz"
#     # std_path = "./project_dir/Quaxly_brain_v3a/ensemble_last/"+organ+"_"+case_id+"_std.nii.gz"
#     os.system("cp "+sct_path+" "+sct_dst)
#     print("Copied: ", sct_path, " to ", sct_dst)
#     # os.system("cp "+std_path+" "+std_dst)
#     # print("Copied: ", std_path, " to ", std_dst)

sct_file_list = sorted(glob.glob("./project_dir/Quaxwell_brain_v3b/eval_best/*_sct.nii.gz"))
for sct_path in sct_file_list:
    # organ = sct_path.split("/")[-1].split("_")[0]
    organ = "brain"
    case_id = sct_path.split("/")[-1].split("_")[1]
    # ct_path = "./data_dir/Task_1/"+organ+"/"+case_id+"/ct.nii.gz"
    sct_dst = "./data_dir/Task1/"+organ+"/"+case_id+"/sct_v3bq_best.nii.gz"
    std_dst = "./data_dir/Task1/"+organ+"/"+case_id+"/std_v3bq_best.nii.gz"
    # std_path = "./project_dir/Quaxly_brain_v3a/ensemble_last/"+organ+"_"+case_id+"_std.nii.gz"
    os.system("cp "+sct_path+" "+sct_dst)
    print("Copied: ", sct_path, " to ", sct_dst)
    # os.system("cp "+std_path+" "+std_dst)
    # print("Copied: ", std_path, " to ", std_dst)



