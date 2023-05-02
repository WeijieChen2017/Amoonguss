import os
import time

model_list = [
    # ["unet_GROWTH_v1_8066", [4], "unet", 8066, 0],
    # ["unet_GROWTH_v1_8066", [4], "unet", 8066, 1],
    # ["unet_GROWTH_v1_8066", [4], "unet", 8066, 2],
    # ["unet_GROWTH_v1_8066", [4], "unet", 8066, 3],
    # ["unet_GROWTH_v1_8066", [4], "unet", 8066, 4],
    # ["unet_GROWTH_v1_8066_16-24-32", [3], "unet", 8066, 2],
    # ["unet_GROWTH_v1_8066_32", [3], "unet", 8066, 0],
    # ["unet_GROWTH_v1_8066_8-16-24-32-40", [3], "unet", 8066, 0],
    ["unet_GROWTH_v1_8066_8-16-24-32-40", [3], "unet", 8066, 4],
]

print("Model index: ", end="")
current_model_idx = int(input()) - 1
print(model_list[current_model_idx])
time.sleep(1)


project_name = model_list[current_model_idx][0]
gpu_list = ','.join(str(x) for x in model_list[current_model_idx][1])
# model_term = model_list[current_model_idx][2]
# random_seed = model_list[current_model_idx][3]
stage_idx = model_list[current_model_idx][4]

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
from model import UNet_GROWTH
from monai.inferers import sliding_window_inference

from util import cal_rmse_mae_ssim_psnr_acut_dice

# for name in model_list:
test_dict = {}
test_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
test_dict["project_name"] = project_name
test_dict["save_folder"] = "./project_dir/"+test_dict["project_name"]+"/"

train_dict = np.load(test_dict["save_folder"]+"dict.npy", allow_pickle=True)[()]

test_dict["gpu_ids"] = gpu_list
test_dict["eval_file_cnt"] = 5
test_dict["eval_save_folder"] = "pred_stage_{:03d}/".format(train_dict["GROWTH_epochs"][stage_idx]["stage"])
test_dict["special_cases"] = []
test_dict["save_tag"] = ""



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
print("Search: ", os.path.join(test_dict["save_folder"], "stage_{:03d}_model_*.pth".format(train_dict["GROWTH_epochs"][stage_idx]["stage"])))
model_list = sorted(glob.glob(os.path.join(test_dict["save_folder"], "stage_{:03d}_model_*.pth".format(train_dict["GROWTH_epochs"][stage_idx]["stage"]))))
if "curr" in model_list[-1]:
    print("Remove model_best_curr")
    model_list.pop()
target_model = model_list[-1]
model_state_dict = torch.load(target_model, map_location=torch.device('cpu'))
print("--->", target_model, " is loaded.")

model = UNet_GROWTH( 
    spatial_dims=train_dict["model_related"]["spatial_dims"],
    in_channels=train_dict["model_related"]["in_channels"],
    out_channels=train_dict["model_related"]["out_channels"],
    # channels=train_dict["model_related"]["channels"],
    channels=train_dict["GROWTH_epochs"][stage_idx]["model_channels"],
    strides=train_dict["model_related"]["strides"],
    num_res_units=train_dict["model_related"]["num_res_units"]
    )

model.load_state_dict(model_state_dict)

# ==================== data division ====================

data_div = np.load(os.path.join(test_dict["save_folder"], "data_division.npy"), allow_pickle=True)[()]
# X_list = data_div['test_list_X']
X_list = data_div['test_list_X']
if test_dict["eval_file_cnt"] > 0:
    X_list = X_list[:test_dict["eval_file_cnt"]]
X_list.sort()


# ==================== training ====================
file_list = []
if len(test_dict["special_cases"]) > 0:
    for case_name in X_list:
        for spc_case_name in test_dict["special_cases"]:
            if spc_case_name in os.path.basename(case_name):
                file_list.append(case_name)
else:
    file_list = X_list

iter_tag = "test"
cnt_total_file = len(file_list)
cnt_each_cube = 1
model.eval()
model = model.to(device)

for cnt_file, file_path in enumerate(file_list):
    
    x_path = file_path
    y_path = file_path.replace("mr", "ct")
    file_name = os.path.basename(file_path)
    print(iter_tag + " ===> Case[{:03d}/{:03d}]: ".format(cnt_file+1, cnt_total_file), x_path, "<---", end="") # 
    x_file = nib.load(x_path)
    y_file = nib.load(y_path)
    x_data = x_file.get_fdata()
    y_data = y_file.get_fdata()
    y_data_denorm = denorm_CT(y_data)

    ax, ay, az = x_data.shape

    input_data = x_data
    input_data = np.expand_dims(input_data, (0,1))
    input_data = torch.from_numpy(input_data).float().to(device)
    output_metric = dict()

    with torch.no_grad():
        # print(order_list[idx_es])
        y_hat = sliding_window_inference(
                inputs = input_data, 
                roi_size = test_dict["input_size"], 
                sw_batch_size = 64, 
                predictor = model,
                overlap=1/8, 
                mode="gaussian", 
                sigma_scale=0.125, 
                padding_mode="constant", 
                cval=0.0, 
                sw_device=device, 
                device=device,
                )
        curr_pred = np.squeeze(y_hat.cpu().detach().numpy())
        curr_pred_denorm = denorm_CT(curr_pred)
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
