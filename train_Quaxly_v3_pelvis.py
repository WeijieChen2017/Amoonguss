import os
import time
import numpy as np

import argparse

# Create the parser
parser = argparse.ArgumentParser(description="This is my program description")

# Add the arguments
parser.add_argument('-m',
                    '--model_number',
                    type=int,
                    help='Model number to be processed')

# Execute the parse_args() method
args = parser.parse_args()

current_model_idx = args.model_number - 1

model_list = [
    ["Quaxly_pelvis_v3mri_mask", [0], 912, 6, 0],
    ["Quaxly_pelvis_v3mri_mask", [0], 912, 6, 1],
    ["Quaxly_pelvis_v3mri_mask", [0], 912, 6, 2],
    ["Quaxly_pelvis_v3mri_mask", [0], 912, 6, 3],
    ["Quaxly_pelvis_v3mri_mask", [0], 912, 6, 4],
    ["Quaxly_pelvis_v3mri_mask", [0], 912, 6, 5],
]

print("Model index: ", end="")
# current_model_idx = int(input()) - 1
print(model_list[current_model_idx])
# time.sleep(1)

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
train_dict["batch_size"] = 16

# train_dict["train_epochs"] = train_dict["GROWTH_epochs"][3]["epochs"]
train_dict["train_epochs"] = 5000
train_dict["eval_per_epochs"] = 100
train_dict["save_per_epochs"] = 1000
train_dict["continue_training_epoch"] = 0
# train_dict["batch"] = train_dict["GROWTH_epochs"][3]["batch"]
# train_dict["start_batch"] = 16
train_dict["batch_decay"] = 2000 # batch into half every 500 epochs
unet_dict = {}
unet_dict["spatial_dims"] = 3
unet_dict["in_channels"] = 1
unet_dict["out_channels"] = 1
unet_dict["channels"] = (16, 32, 64, 128, 256)
unet_dict["strides"] = (2, 2, 2, 2)
unet_dict["num_res_units"] = 4

train_dict["model_para"] = unet_dict

train_dict["opt_betas"] = (0.9, 0.999) # default
train_dict["opt_eps"] = 1e-8 # default
train_dict["opt_lr"] = 1e-3 # default
train_dict["opt_weight_decay"] = 0.01 # default
train_dict["amsgrad"] = False # default

folders_to_create = [
    train_dict["save_folder"],
    train_dict["save_folder"]+"model/",
    train_dict["save_folder"]+"loss/",
    train_dict["save_folder"]+"sample_cache/",
]

for path in folders_to_create:
    if not os.path.exists(path):
        os.mkdir(path)


from torch.nn import SmoothL1Loss
from model import UNet_Quaxly
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, _LRScheduler, CosineAnnealingLR

import os
import json
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm


from monai.inferers import sliding_window_inference
from monai.transforms import (
    # AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    ScaleIntensityRanged,
    Orientationd,
    RandFlipd,
    Spacingd,
    RandRotate90d,
    # RandSpatialCropd,
    RandSpatialCropSamplesd,
    SpatialPadd,
)
from util import (
    CustomNormalize,
    AddRicianNoise,
    # create_nfold_json,
    create_nfold_json_MASK_MR,
)

from monai.config import print_config

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

print_config()

root_dir = train_dict["save_folder"]
print(root_dir)

train_transforms = Compose(
    [
        LoadImaged(keys=["MR", "CT", "MASK_MR"]),
        EnsureChannelFirstd(keys=["MR", "CT", "MASK_MR"]),
        Orientationd(keys=["MR", "CT", "MASK_MR"], axcodes="RAS"),
        Spacingd(
            keys=["MR", "CT", "MASK_MR"],
            pixdim=(1., 1., 1.),
            mode=("bilinear", "bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["MR"],
            a_min=0,
            a_max=3000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ScaleIntensityRanged(
            keys=["CT"],
            a_min=-1024,
            a_max=3000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        AddRicianNoise(keys=["MR"], noise_std=0.01),
        # CropForegroundd(
        #     keys=["MR", "CT", "MASK"],
        #     source_key="MASK",
        #     margin=(0, 0, 0),
        #     select_fn=lambda x: x != 0,
        #     return_transform=False,
        # ),
        RandSpatialCropSamplesd(
            keys=["MR", "CT", "MASK_MR"],
            num_samples = 4, 
            roi_size=train_dict["input_size"], 
            random_size=False,
        ),
        RandFlipd(
            keys=["MR", "CT", "MASK_MR"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["MR", "CT", "MASK_MR"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["MR", "CT", "MASK_MR"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["MR", "CT", "MASK_MR"],
            prob=0.10,
            max_k=3,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["MR", "CT", "MASK_MR"]),
        EnsureChannelFirstd(keys=["MR", "CT", "MASK_MR"]),
        Orientationd(keys=["MR", "CT", "MASK_MR"], axcodes="RAS"),
        Spacingd(
            keys=["MR", "CT", "MASK_MR"],
            pixdim=(1, 1, 1),
            mode=("bilinear", "bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["MR"],
            a_min=0,
            a_max=3000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ScaleIntensityRanged(
            keys=["CT"],
            a_min=-1024,
            a_max=3000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        SpatialPadd(
            keys=["MR", "CT", "MASK_MR"],
            spatial_size=(288, 288, 288) if train_dict["organ"] == "brain" else (640, 440, 160),
            mode=("constant", "constant", "constant"),
        ),
        # CropForegroundd(
        #     keys=["MR", "CT", "MASK"],
        #     source_key="MASK",
        #     margin=(0, 0, 0),
        #     select_fn=lambda x: x != 0,
        #     return_transform=False,
        # ),
        # RandSpatialCropSamplesd(
        #     keys=["MR", "CT", "MASK"],
        #     num_samples = 16, 
        #     roi_size=(64, 64, 64), 
        #     random_size=False,
        # ),
    ]
)

data_dir = "./data_dir/Task1/"
# data_json = data_dir+"brain.json" if train_dict["organ"] == "brain" else data_dir+"pelvis.json"
data_json = data_dir+"pelvis_mri_mask.json"
print("data_json: ", data_json)
curr_fold = train_dict["current_fold"]
if train_dict["current_fold"] == 0:
    create_nfold_json_MASK_MR(data_json, train_dict["num_fold"], train_dict["random_seed"], train_dict["save_folder"])

# n_stage = len(train_dict["GROWTH_epochs"])
n_fold = train_dict["num_fold"]
curr_fold = train_dict["current_fold"]
organ = train_dict["organ"]

split_json = root_dir + f"fold_{curr_fold + 1}.json"
# with open(data_json, "r") as f:
#     datasets = json.load(f)

train_files = load_decathlon_datalist(split_json, True, "training")
val_files = load_decathlon_datalist(split_json, True, "validation")
n_train_files = len(train_files)
n_val_files = len(val_files)
print("Load Training Files: ", n_train_files, "Load Validation Files: ", n_val_files)
train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    # cache_num=24,
    cache_rate=0.5,
    num_workers=3,
)
val_ds = CacheDataset(
    data=val_files, 
    transform=val_transforms, 
    # cache_num=6, 
    cache_rate=0.5, 
    num_workers=1,
)

model = UNet_Quaxly( 
    spatial_dims=unet_dict["spatial_dims"],
    in_channels=unet_dict["in_channels"],
    out_channels=unet_dict["out_channels"],
    channels=unet_dict["channels"],
    strides=unet_dict["strides"],
    num_res_units=unet_dict["num_res_units"],
    # partial_init=partial_init,
    )

torch.backends.cudnn.benchmark = True
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = train_dict["opt_lr"],
    betas = train_dict["opt_betas"],
    eps = train_dict["opt_eps"],
    weight_decay = train_dict["opt_weight_decay"],
    amsgrad = train_dict["amsgrad"]
    )

# scheduler = lr_scheduler.CosineAnnealingLR(
#     optim, 
#     T_max=500, 
#     eta_min=1e-5,
# )

# def custom_coefficient(x: int) -> float:
#     # return 10^(-x/3750) using numpy
#     # (2000*5+2000*10+2000*19+2000*38+2000*76) ~= 300000 = 3e5
#     return np.power(10, -x/1.5e5)

# class CustomCosineAnnealingWarmRestarts(_LRScheduler):
#     def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
#         self.base_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult, eta_min, last_epoch, verbose)
#         super(CustomCosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch, verbose)

#     def get_lr(self):
#         # Get the learning rates from the base scheduler
#         base_lrs = self.base_scheduler.get_lr()

#         # Multiply the learning rates with the custom coefficient
#         coeff = custom_coefficient(self.last_epoch)
#         return [lr * coeff for lr in base_lrs]

#     def step(self, epoch=None):
#         # Update the base scheduler's epoch counter
#         self.base_scheduler.step(epoch)
#         super().step(epoch)

# Create the custom scheduler
# T_0 = 1000  # The number of epochs for the first restart
# scheduler = CustomCosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=5e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=5000, eta_min=5e-5)

criterion = SmoothL1Loss()

# print("Test successful, now exiting...")
# exit()

# build new dataloader at epoch 0, 500, 1000, 1500, 2000, 2500

best_val_loss = 1000
best_epoch = 0
model.to(device)
loss_to_save = {"train": [], "val": []}

for idx_epoch_new in range(train_dict["train_epochs"]):
    idx_epoch = idx_epoch_new + train_dict["continue_training_epoch"]
    # print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))

    # check the idx_epoch to determine the batch size
    #                 32,   16,    8,    4,    2,     1
    # if idx_epoch in [0, 1000, 2000, 3000, 4000, 5000, ]:
    #     batch_stage = 5 - idx_epoch // train_dict["batch_decay"]
    #     batch_size = 2 ** batch_stage
    #     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    #     val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    batch_size = train_dict["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


    # training
    model.train()
    curr_iter = n_train_files // batch_size + 1
    # print("Training: ", curr_iter, "iterations")
    case_loss = np.zeros((curr_iter, 1))
    loss_batch_to_save = []
    for step, batch in enumerate(train_loader):
        mr, ct, mask_mr = (batch["MR"].float().to(device), batch["CT"].float().to(device), batch["MASK_MR"].float().to(device))
        # mr, ct, mask = (batch["MR"], batch["CT"], batch["MASK"])
        # print("step[", step, "]mr", mr.shape, "ct", ct.shape, "mask", mask.shape)
        # print(" ===> Train:Epoch[{:03d}]:[{:03d}]/[{:03d}] --->".format(idx_epoch+1, step, curr_iter), end="")
            
        optimizer.zero_grad()
        sct, ds_1, ds_2, ds_3 = model(mr, is_deep_supervision=True)
        loss_out = criterion(ct, sct)
        loss_ds_1 = criterion(ct, ds_1)
        loss_ds_2 = criterion(ct, ds_2)
        loss_ds_3 = criterion(ct, ds_3)
        loss = loss_out + loss_ds_1 + loss_ds_2 + loss_ds_3
        final_loss = torch.sum(loss * mask_mr) / torch.sum(mask_mr)
        final_loss = loss
        final_loss.backward()
        optimizer.step()
        case_loss[step] = final_loss.item()
        # current_lr = scheduler.get_last_lr()[0]
        # print(f" lr:{current_lr}")
        # step += 1
    scheduler.step()


    loss_batch_to_save.append(case_loss)
    loss_to_save["train"].append(loss_batch_to_save)
    current_lr = scheduler.get_last_lr()[0]
    print(" ===> Train:Epoch[{:04d}]: Iter: {:03d}, Loss: {:06f}, lr:{:06f}".format(idx_epoch+1, curr_iter, np.mean(case_loss), current_lr))
        # print("Loss: ", case_loss[step], end="")
        
        # np.save(train_dict["save_folder"]+"loss/fold_{:02d}_train_{:04d}.npy".format(curr_fold, idx_epoch+1), case_loss)

    # validation
    if (idx_epoch+1) % train_dict["eval_per_epochs"] == 0:
        model.eval()
        curr_iter = n_val_files
        # print("Validation: ", curr_iter, "iterations")
        case_loss = np.zeros((curr_iter, 1))
        loss_batch_to_save = []
        for step, batch in enumerate(val_loader):
            mr, ct, mask_mr = (batch["MR"].float().to(device), batch["CT"].float().to(device), batch["MASK_MR"].float().to(device))
            # mr, ct, mask = (batch["MR"], batch["CT"], batch["MASK"])
            # print("step[", step, "]mr", mr.shape, "ct", ct.shape, "mask", mask.shape)
            # print(" ===> Validation: Epoch[{:03d}]:[{:03d}]/[{:03d}] --->".format(idx_epoch+1, step, curr_iter), end="")
            
            with torch.no_grad():
                sct = sliding_window_inference(
                inputs = mr, 
                roi_size = train_dict["input_size"], 
                sw_batch_size = 32, 
                predictor = model,
                overlap=1/8, 
                mode="gaussian", 
                sigma_scale=0.125, 
                padding_mode="constant", 
                cval=0.0, 
                sw_device=device, 
                device=device,
                )
                loss = criterion(ct, sct)
                final_loss = torch.sum(loss * mask_mr) / torch.sum(mask_mr)
                # final_loss = loss
                case_loss[step] = final_loss.item()
            # print("Loss: ", case_loss[step])
            # np.save(train_dict["save_folder"]+"loss/fold_{:02d}_val_{:04d}.npy".format(curr_fold, idx_epoch+1), case_loss)
            loss_batch_to_save.append(case_loss)
            # loss_batch_to_save
            # step += 1
        
        loss_to_save["val"].append(loss_batch_to_save)

        curr_mae = np.mean(case_loss) * 4024
        print(" ===> Validation: Epoch[{:04d}]: Iter:{:03d}, Loss:{:06f}".format(idx_epoch+1, curr_iter, np.mean(case_loss)), end="")
        print("Curr MAE: ", curr_mae, "Best MAE: ", best_val_loss, "Best Epoch: ", best_epoch)
        if curr_mae < best_val_loss:
            best_val_loss = curr_mae
            best_epoch = idx_epoch+1
            torch.save(model.state_dict(), train_dict["save_folder"]+"model/fold_{:02d}_model_best.pth".format(curr_fold))
            torch.save(optimizer.state_dict(), train_dict["save_folder"]+"model/fold_{:02d}_optim_best.pth".format(curr_fold))
            torch.save(scheduler.state_dict(), train_dict["save_folder"]+"model/fold_{:02d}_scheduler_best.pth".format(curr_fold))
            print("Best model saved at epoch {:03d} with MAE {:03f}".format(best_epoch, best_val_loss))

    # save the model and data every train_dict["save_per_epochs"] epochs
    if (idx_epoch+1) % train_dict["save_per_epochs"] == 0:
        torch.save(model.state_dict(), train_dict["save_folder"]+"model/fold_{:02d}_model_{:04d}.pth".format(curr_fold, idx_epoch+1))
        torch.save(optimizer.state_dict(), train_dict["save_folder"]+"model/fold_{:02d}_optim_{:04d}.pth".format(curr_fold, idx_epoch+1))
        torch.save(scheduler.state_dict(), train_dict["save_folder"]+"model/fold_{:02d}_scheduler_{:04d}.pth".format(curr_fold, idx_epoch+1))
        np.save(train_dict["save_folder"]+"loss/fold_{:02d}_val_{:04d}.npy".format(curr_fold, idx_epoch+1), loss_to_save)
        loss_to_save = {"train": [], "val": []}
        print("Model saved at epoch {:03d}".format(idx_epoch+1))

        # mr_cache = mr.detach().cpu().numpy()
        # ct_cache = ct.detach().cpu().numpy()
        # sct_cache = sct.detach().cpu().numpy()
        # mask_mr_cache = mask_mr.detach().cpu().numpy()
        # sample_cache = {
        #     "MR": mr_cache,
        #     "CT": ct_cache,
        #     "SCT": sct_cache,
        #     "MASK_MR": mask_mr_cache,
        # }
        # np.save(train_dict["save_folder"]+"sample_cache/fold_{:02d}_sample_{:04d}.npy".format(curr_fold, idx_epoch+1), sample_cache)
        # print("Sample saved at epoch {:03d}".format(idx_epoch+1))
    

print("Training finished!")
print("The best model is saved at epoch {:03d} with MAE {:03f}".format(best_epoch, best_val_loss))
