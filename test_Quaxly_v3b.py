import os
import time
import glob
import numpy as np
import nibabel as nib

model_list = [
    ["Quaxly_brain_v3b", [7], 912, 6, 0],
    # ["Quaxly_pelvis_v2", [1], 912, 6, 0],
]

print("Model index: ", end="")
current_model_idx = int(input()) - 1
print(model_list[current_model_idx])
time.sleep(1)

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = model_list[current_model_idx][0]
train_dict["gpu_ids"] = model_list[current_model_idx][1]
train_dict["random_seed"] = model_list[current_model_idx][2]
train_dict["organ"] = "brain" if "brain" in train_dict["project_name"].lower() else "pelvis"
train_dict["num_fold"] = model_list[current_model_idx][3]
train_dict["current_fold"] = model_list[current_model_idx][4]

gpu_list = ','.join(str(x) for x in train_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dict["optimizer"] = "AdamW"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["input_size"] = (64, 64, 64)

train_dict["GROWTH_epochs"] = [
    {"stage": 0, "model_channels": (8, 16, 32, 64), "epochs" : 25, "batch" : 32, "lr": 1e-3, "loss": "l2",},
    {"stage": 1, "model_channels": (16, 32, 64, 128), "epochs" : 50, "batch" : 32, "lr": 7e-4, "loss": "l2",},
    {"stage": 2, "model_channels": (24, 48, 96, 192), "epochs" : 75, "batch" : 16, "lr": 5e-4, "loss": "l1",},
    {"stage": 3, "model_channels": (32, 64, 128, 256), "epochs" : 100, "batch" : 16, "lr": 3e-4, "loss": "l1",},
    {"stage": 4, "model_channels": (40, 80, 160, 320), "epochs" : 150, "batch" : 8, "lr": 1e-4, "loss": "l1",},    
]

# train_dict["train_epochs"] = train_dict["GROWTH_epochs"][3]["epochs"]
train_dict["train_epochs"] = 2500
train_dict["eval_per_epochs"] = 25
train_dict["save_per_epochs"] = 100
train_dict["continue_training_epoch"] = 0
# train_dict["batch"] = train_dict["GROWTH_epochs"][3]["batch"]
train_dict["start_batch"] = 16
train_dict["batch_decay"] = 500 # batch into half every 500 epochs
unet_dict = {}
unet_dict["spatial_dims"] = 3
unet_dict["in_channels"] = 2
unet_dict["out_channels"] = 1
unet_dict["channels"] = (16, 32, 64, 128, 256)
unet_dict["strides"] = (2, 2, 2, 2)
unet_dict["num_res_units"] = 4

train_dict["model_para"] = unet_dict

for path in [train_dict["save_folder"]+"eval_best/", train_dict["save_folder"]+"eval_last/"]:
    if not os.path.exists(path):
        os.mkdir(path)


from model import UNet_Quaxly

import os
import json

import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.inferers import sliding_window_inference

data_dir = "./data_dir/Task1/"
data_json = data_dir+"brain.json" if train_dict["organ"] == "brain" else data_dir+"pelvis.json"
print("data_json: ", data_json)
n_fold = train_dict["num_fold"]
organ = train_dict["organ"]
root_dir = "./project_dir/"+train_dict["project_name"]+"/"

# for idx_fold in range(n_fold):
# for idx_fold in [0,1,4,5]:
for idx_fold in [1,2,3]:
    curr_fold = idx_fold
    split_json = root_dir + f"fold_{curr_fold + 1}.json"
    with open(split_json, "r") as f:
        datasets = json.load(f)
        val_files = datasets["validation"]

    # test for the best model
    # best_model = root_dir + "model/fold_{:02d}_model_best.pth".format(curr_fold)
    # print("best_model: ", best_model)
    # best_model = torch.load(best_model)

    # model = UNet_Quaxly( 
    #     spatial_dims=unet_dict["spatial_dims"],
    #     in_channels=unet_dict["in_channels"],
    #     out_channels=unet_dict["out_channels"],
    #     channels=unet_dict["channels"],
    #     strides=unet_dict["strides"],
    #     num_res_units=unet_dict["num_res_units"],
    #     )
    # model.load_state_dict(best_model)
    # model.eval()
    # model.to(device)
    # n_val_files = len(val_files)
    # mae_val = np.zeros(n_val_files)

    # for idx_case, val_case in enumerate(val_files):
    #     print("[{:03d}]/[{:03d}] Processing: ".format(idx_case+1, n_val_files), end="<--->")
    #     mr_path = val_case["MR"]
    #     ct_path = val_case["CT"]
    #     v3a_last_path = val_case["v3a_last"]
    #     # mr, ct, mask = (batch["MR"], batch["CT"], batch["MASK"])
    #     # print("step[", step, "]mr", mr.shape, "ct", ct.shape, "mask", mask.shape)
    #     mask_path = val_case["MASK"]
    #     mr_file = nib.load(mr_path)
    #     ct_file = nib.load(ct_path)
    #     v3a_last_file = nib.load(v3a_last_path)
    #     mask_file = nib.load(mask_path)
    #     # mr_path is like ./data_dir/Task_1/brain/1BA001/mr.nii.gz
    #     organ_case = mr_path.split("/")[-3]+"_"+mr_path.split("/")[-2]
    #     print("Loaded: ", mr_path, end="<--->")

    #     mr_data = mr_file.get_fdata()
    #     ct_data = ct_file.get_fdata()
    #     mask_data = mask_file.get_fdata()
    #     v3a_last_data = v3a_last_file.get_fdata()

    #     mr_data = mr_data / 3000
    #     v3a_last_data = (v3a_last_data + 1024) / 4024
    #     mr_data = np.expand_dims(mr_data, (0,1))
    #     v3a_last_data = np.expand_dims(v3a_last_data, (0,1))
    #     mr_data = torch.from_numpy(mr_data).float()
    #     v3a_last_data = torch.from_numpy(v3a_last_data).float()
    #     input_mr_v3a_last = torch.concat((mr_data, v3a_last_data), dim=1).to(device)
    #     # input_data = np.expand_dims(mr_data, (0,1))
    #     # input_data = torch.from_numpy(input_data).float().to(device)

    #     sct = sliding_window_inference(
    #     inputs = input_mr_v3a_last, 
    #     roi_size = train_dict["input_size"][0], 
    #     sw_batch_size = 32, 
    #     predictor = model,
    #     overlap=1/4, 
    #     mode="gaussian", 
    #     sigma_scale=1/4, 
    #     padding_mode="constant", 
    #     cval=0.0, 
    #     sw_device=device, 
    #     device=device,
    #     # is_deep_supervision = False,
    #     )

    #     sct = np.squeeze(sct.cpu().detach().numpy()) # 0->1
    #     sct = sct * mask_data # 0->1
    #     sct = sct * 4024 - 1024 # -1024->3000
    #     ct = (ct_data + 1024)/4024 # 0->1
    #     ct = ct * mask_data
    #     ct = ct * 4024 - 1024 # -1024 -> 3000
    #     sct = np.clip(sct, -1024, 3000)
    #     ct = np.clip(ct, -1024, 3000)


    #     masked_mae = np.sum(np.abs(ct - sct)) / np.sum(mask_data)
    #     print("Masked MAE: ", masked_mae)
    #     # whole_mae = np.mean(np.abs(ct - sct))
    #     # print("Whole MAE: ", whole_mae)
    #     sct_file = nib.Nifti1Image(sct, ct_file.affine, ct_file.header)
    #     sct_savename = train_dict["save_folder"]+"eval_best/"+organ_case+"_sct.nii.gz"
    #     nib.save(sct_file, sct_savename)
    #     print("Saved: ", sct_savename)
    #     np.save(train_dict["save_folder"]+"eval_best/"+organ_case+"_mae.npy", masked_mae)
    #     mae_val[idx_case] = masked_mae
    # print("Mean MAE: ", np.mean(mae_val))

    # test for the last model
    # model_list = sorted(glob.glob(root_dir + "model/fold_{:02d}_model_*000.pth".format(curr_fold)))

    last_model = root_dir + "model/fold_{:02d}_model_6000.pth".format(curr_fold)
    print("last_model: ", last_model)
    last_model = torch.load(last_model)

    model = UNet_Quaxly( 
        spatial_dims=unet_dict["spatial_dims"],
        in_channels=unet_dict["in_channels"],
        out_channels=unet_dict["out_channels"],
        channels=unet_dict["channels"],
        strides=unet_dict["strides"],
        num_res_units=unet_dict["num_res_units"],
        )
    model.load_state_dict(last_model)
    model.eval()
    model.to(device)
    n_val_files = len(val_files)
    mae_val = np.zeros(n_val_files)

    for idx_case, val_case in enumerate(val_files):
        print("[{:03d}]/[{:03d}] Processing: ".format(idx_case+1, n_val_files), val_case["MR"], end="")
        mr_path = val_case["MR"]
        ct_path = val_case["CT"]
        mask_path = val_case["MASK"]
        v3a_last_path = val_case["v3a_last"]
        mr_file = nib.load(mr_path)
        ct_file = nib.load(ct_path)
        mask_file = nib.load(mask_path)
        v3a_last_file = nib.load(v3a_last_path)
        # mr_path is like ./data_dir/Task_1/brain/1BA001/mr.nii.gz
        organ_case = mr_path.split("/")[-2]+"_"+mr_path.split("/")[-1]
        print("Loaded: ", mr_path, end="")

        mr_data = mr_file.get_fdata() / 3000
        ct_data = ct_file.get_fdata()
        mask_data = mask_file.get_fdata()
        v3a_last_data = (v3a_last_file.get_fdata() + 1024) / 4024

        # input_data = np.expand_dims(mr_data, (0,1))
        # input_data = torch.from_numpy(input_data).float().to(device)
        mr_data = np.expand_dims(mr_data, (0,1))
        v3a_last_data = np.expand_dims(v3a_last_data, (0,1))
        mr_data = torch.from_numpy(mr_data).float()
        v3a_last_data = torch.from_numpy(v3a_last_data).float()
        input_mr_v3a_last = torch.concat((mr_data, v3a_last_data), dim=1).float().to(device)

        sct = sliding_window_inference(
        inputs = input_mr_v3a_last, 
        roi_size = train_dict["input_size"][0], 
        sw_batch_size = 32, 
        predictor = model,
        overlap=1/4, 
        mode="gaussian", 
        sigma_scale=1/4, 
        padding_mode="constant", 
        cval=0.0, 
        sw_device=device, 
        device=device,
        # is_deep_supervision = False,
        )

        sct = np.squeeze(sct.cpu().detach().numpy()) # 0->1
        sct = sct * mask_data # 0->1
        sct = sct * 4024 - 1024 # -1024->3000
        ct = (ct_data + 1024)/4024 # 0->1
        ct = ct * mask_data
        ct = ct * 4024 - 1024 # -1024 -> 3000
        sct = np.clip(sct, -1024, 3000)
        ct = np.clip(ct, -1024, 3000)

        masked_mae = np.sum(np.abs(ct * mask_data - sct * mask_data)) / np.sum(mask_data)
        print("Masked MAE: ", masked_mae)
        # whole_mae = np.mean(np.abs(ct - sct))
        # print("Whole MAE: ", whole_mae)
        sct_file = nib.Nifti1Image(sct, ct_file.affine, ct_file.header)
        sct_savename = train_dict["save_folder"]+"eval_last/"+organ_case+"_sct.nii.gz"
        nib.save(sct_file, sct_savename)
        print("Saved: ", sct_savename)
        np.save(train_dict["save_folder"]+"eval_last/"+organ_case+"_mae.npy", masked_mae)
        mae_val[idx_case] = masked_mae

    print("Mean MAE: ", np.mean(mae_val))



