import os
import time

model_list = [
    ["unet_v1_8066", [7], "unet", 8066],
    # ["unet_v1_5541", [7], "unet", 5541],
    # ["unet_v1_7363", [7], "unet", 7363],
    # ["dynunet_v1", [7], "dynunet"],
]

print("Model index: ", end="")
current_model_idx = int(input()) - 1
print(model_list[current_model_idx])
time.sleep(1)


project_name = model_list[current_model_idx][0]
gpu_list = ','.join(str(x) for x in model_list[current_model_idx][1])
# model_term = model_list[current_model_idx][2]
# random_seed = model_list[current_model_idx][3]

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
import glob
import time

import numpy as np
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F


# from monai.networks.nets.unet import UNet
from monai.networks.layers.factories import Act, Norm

# from utils import add_noise, weighted_L1Loss
from monai.networks.nets.unet import UNet as unet
from monai.inferers import sliding_window_inference

from util import cal_rmse_mae_ssim_psnr_acut_dice

# for name in model_list:
test_dict = {}
test_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
test_dict["project_name"] = project_name
test_dict["save_folder"] = "./project_dir/"+test_dict["project_name"]+"/"
test_dict["gpu_ids"] = gpu_list
test_dict["eval_file_cnt"] = 0
test_dict["eval_save_folder"] = "pred_vanilla/"
test_dict["special_cases"] = []

test_dict["save_tag"] = ""

train_dict = np.load(test_dict["save_folder"]+"dict.npy", allow_pickle=True)[()]

test_dict["random_seed"] = train_dict["random_seed"]
np.random.seed(train_dict["random_seed"])
test_dict["input_size"] = train_dict["input_size"]

print("input size:", test_dict["input_size"])

for path in [test_dict["save_folder"], test_dict["save_folder"]+test_dict["eval_save_folder"]]:
    if not os.path.exists(path):
        os.mkdir(path)

np.save(test_dict["save_folder"]+"test_dict.npy", test_dict)

# ==================== denorm CT ====================

def denorm_CT(x):
    # CT_data = (CT_data + 1024) / (3000+1024)
    x = x * (3000+1024) - 1024
    return x

# ==================== basic settings ====================

model_list = sorted(glob.glob(os.path.join(test_dict["save_folder"], "model_best_*.pth")))
if "curr" in model_list[-1]:
    print("Remove model_best_curr")
    model_list.pop()
target_model = model_list[-1]
model_state_dict = torch.load(target_model, map_location=torch.device('cpu'))
print("--->", target_model, " is loaded.")

model = unet( 
    spatial_dims=train_dict["model_related"]["spatial_dims"],
    in_channels=train_dict["model_related"]["in_channels"],
    out_channels=train_dict["model_related"]["out_channels"],
    channels=train_dict["model_related"]["channels"],
    strides=train_dict["model_related"]["strides"],
    num_res_units=train_dict["model_related"]["num_res_units"]
    )

model.load_state_dict(model_state_dict)

# ==================== data division ====================

X_list = sorted(glob.glob(os.path.join(test_dict["save_folder"], test_dict["eval_save_folder"], "*.nii.gz")))

cnt_total_file = len(X_list)

for cnt_file, file_path in enumerate(X_list):
    
    x_path = file_path
    file_name = os.path.basename(file_path)
    iter_tag = file_name.split("_")[0]
    folder_tag = file_name.split("_")[1]
    # brain_1BA054_mr.nii.gz
    y_path = "./data_dir/t1_ct_norm/"+file_name.replace("mr", "ct")
    mask_path = "./data_dir/Task1/"+iter_tag+"/"+folder_tag+"/mask.nii.gz"

    x_file = nib.load(x_path)
    y_file = nib.load(y_path)
    mask_file = nib.load(mask_path)

    x_data = x_file.get_fdata()
    y_data = y_file.get_fdata()
    mask_data = mask_file.get_fdata()

    print(iter_tag + " ===> Case[{:03d}/{:03d}]: ".format(cnt_file+1, cnt_total_file), x_path, "<---", end="") # 
    
    ax, ay, az = x_data.shape
    curr_pred_denorm = denorm_CT(x_data)
    y_data_denorm = denorm_CT(y_data)
    curr_pred_denorm_masked = np.multiply(curr_pred_denorm, mask_data)
    y_data_denorm_masked = np.multiply()

    metric_list = cal_rmse_mae_ssim_psnr_acut_dice(curr_pred_denorm, y_data_denorm)
    metric_keys = ["RMSE", "MAE", "SSIM", "PSNR", "ACUT", "DICE_AIR", "DICE_BONE", "DICE_SOFT"]
    # metric_list = cal_mae(curr_pred_denorm, y_data_denorm)
    key_name = file_name.replace(".nii.gz", "")
    for idx, key in enumerate(metric_keys):
        output_metric[key] = metric_list[idx]
    print("MAE: ", output_metric["MAE"], end=" ")

    # save nifty prediction
    save_path = os.path.join(test_dict["save_folder"], test_dict["eval_save_folder"], file_name)
    save_file = nib.Nifti1Image(curr_pred_denorm, x_file.affine, x_file.header)
    nib.save(save_file, save_path)
    print("save to", save_path)

    # save metrics
    save_path = os.path.join(test_dict["save_folder"], test_dict["eval_save_folder"], file_name.replace(".nii.gz", ".npy"))
    np.save(save_path, output_metric)
