
import os
import time

model_list = [
    ["dynunet_v1_8066", [7], "dynunet", 8066],
    ["dynunet_v1_5541", [7], "dynunet", 5541],
    ["dynunet_v1_7363", [7], "dynunet", 7363],
    # ["dynunet_v1", [7], "dynunet"],
]

print("Model index: ", end="")
current_model_idx = int(input()) - 1
print(model_list[current_model_idx])
time.sleep(1)

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = model_list[current_model_idx][0]
train_dict["gpu_ids"] = model_list[current_model_idx][1]
train_dict["model_term"] = model_list[current_model_idx][2]
train_dict["random_seed"] = model_list[current_model_idx][3]

gpu_list = ','.join(str(x) for x in train_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import glob
import time
import random

import numpy as np
import nibabel as nib
import torch.nn as nn

# import torch
import torchvision
import requests

from monai.networks.nets.unet import UNet as unet
from monai.networks.nets.dynunet import DynUnet as  dynunet

# ==================== dict and config ====================

train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["input_size"] = [80, 80, 80]
train_dict["epochs"] = 200
train_dict["batch"] = 32
train_dict["dropout"] = 0

train_dict["model_related"] = {}
train_dict["model_related"]["spatial_dims"] = 3
train_dict["model_related"]["in_channels"] = 1
train_dict["model_related"]["out_channels"] = 1
train_dict["model_related"]["kernel_size"] = [3, 3, 3, 1, 1]
train_dict["model_related"]["strides"] = [2, 2, 2, 1, 1]
train_dict["model_related"]["filters"] = [64, 128, 256, 384, 512]
train_dict["model_related"]["norm_name"] = "instance"
train_dict["model_related"]["deep_supervision"] = True
train_dict["model_related"]["deep_supr_num"] = 3

train_dict["folder_X"] = "./data_dir/t1_mr_norm/"
train_dict["folder_Y"] = "./data_dir/t1_ct_norm/"
train_dict["val_ratio"] = 0.3
train_dict["test_ratio"] = 0.2

train_dict["loss_term"] = "SmoothL1Loss"
train_dict["optimizer"] = "AdamW"
train_dict["opt_lr"] = 1e-3 # default
train_dict["opt_betas"] = (0.9, 0.999) # default
train_dict["opt_eps"] = 1e-8 # default
train_dict["opt_weight_decay"] = 0.01 # default
train_dict["amsgrad"] = False # default

for path in [train_dict["save_folder"], train_dict["save_folder"]+"npy/", train_dict["save_folder"]+"loss/"]:
    if not os.path.exists(path):
        os.mkdir(path)

np.save(train_dict["save_folder"]+"dict.npy", train_dict)


# ==================== basic settings ====================

np.random.seed(train_dict["random_seed"])

if train_dict["model_term"] == "unet":
    model = unet( 
        spatial_dims=train_dict["model_related"]["spatial_dims"],
        in_channels=train_dict["model_related"]["in_channels"],
        out_channels=train_dict["model_related"]["out_channels"],
        channels=train_dict["model_related"]["channels"],
        strides=train_dict["model_related"]["strides"],
        num_res_units=train_dict["model_related"]["num_res_units"]
    )

if train_dict["model_term"] == "dynunet":
    model = dynunet(
        spatial_dims=train_dict["model_related"]["spatial_dims"],
        in_channels=train_dict["model_related"]["in_channels"],
        out_channels=train_dict["model_related"]["out_channels"],
        kernel_size=train_dict["model_related"]["kernel_size"],
        strides=train_dict["model_related"]["strides"],
        upsample_kernel_size=train_dict["model_related"]["strides"][1:],
        filters=train_dict["model_related"]["filters"],
        norm_name=train_dict["model_related"]["norm_name"],
        deep_supervision=train_dict["model_related"]["deep_supervision"],
        deep_supr_num=train_dict["model_related"]["deep_supr_num"],
    )

model.train()
model = model.to(device)
criterion = nn.SmoothL1Loss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = train_dict["opt_lr"],
    betas = train_dict["opt_betas"],
    eps = train_dict["opt_eps"],
    weight_decay = train_dict["opt_weight_decay"],
    amsgrad = train_dict["amsgrad"]
    )

# ==================== data division ====================

X_list = sorted(glob.glob(train_dict["folder_X"]+"*.nii.gz"))
Y_list = sorted(glob.glob(train_dict["folder_Y"]+"*.nii.gz"))

selected_list = np.asarray(X_list)
np.random.shuffle(selected_list)
selected_list = list(selected_list)

val_list = selected_list[:int(len(selected_list)*train_dict["val_ratio"])]
val_list.sort()
test_list = selected_list[-int(len(selected_list)*train_dict["test_ratio"]):]
test_list.sort()
train_list = list(set(selected_list) - set(val_list) - set(test_list))
train_list.sort()

data_division_dict = {
    "train_list_X" : train_list,
    "val_list_X" : val_list,
    "test_list_X" : test_list}
np.save(train_dict["save_folder"]+"data_division.npy", data_division_dict)

print("Train: ", len(train_list))
print("Val: ", len(val_list))
print("Test: ", len(test_list))

# ==================== training ====================

best_val_loss = 100
best_epoch = 0
# wandb.watch(model)

package_train = [train_list, True, False, "train"]
package_val = [val_list, False, True, "val"]
# package_test = [test_list, False, False, "test"]

for idx_epoch_new in range(train_dict["epochs"]):
    idx_epoch = idx_epoch_new
    print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))

    for package in [package_train, package_val]:

        file_list = package[0]
        isTrain = package[1]
        isVal = package[2]
        iter_tag = package[3]

        if isTrain:
            model.train()
        else:
            model.eval()

        random.shuffle(file_list)
        n_file = len(file_list)
        
        case_loss = np.zeros((len(file_list)))

        # N, C, D, H, W
        # x_data = nib.load(file_list[0]).get_fdata()

        for cnt_file, file_path in enumerate(file_list):
            
            
            x_path = file_path
            y_path = file_path.replace("mr", "ct")
            file_name = os.path.basename(file_path)
            print(iter_tag + " ===> Epoch[{:03d}]:[{:03d}]/[{:03d}] --->".format(idx_epoch+1, cnt_file+1, n_file), x_path, "<---", end="")
            x_file = nib.load(x_path)
            y_file = nib.load(y_path)
            x_data = x_file.get_fdata()
            y_data = y_file.get_fdata()
            # x_data = x_data / np.amax(x_data)

            batch_x = np.zeros((train_dict["batch"], 1, train_dict["input_size"][0], train_dict["input_size"][1], train_dict["input_size"][2]))
            batch_y = np.zeros((train_dict["batch"], 1, train_dict["input_size"][0], train_dict["input_size"][1], train_dict["input_size"][2]))

            for idx_batch in range(train_dict["batch"]):
                
                d0_offset = np.random.randint(x_data.shape[0] - train_dict["input_size"][0])
                d1_offset = np.random.randint(x_data.shape[1] - train_dict["input_size"][1])
                d2_offset = np.random.randint(x_data.shape[2] - train_dict["input_size"][2])

                x_slice = x_data[d0_offset:d0_offset+train_dict["input_size"][0],
                                 d1_offset:d1_offset+train_dict["input_size"][1],
                                 d2_offset:d2_offset+train_dict["input_size"][2]
                                 ]
                y_slice = y_data[d0_offset:d0_offset+train_dict["input_size"][0],
                                 d1_offset:d1_offset+train_dict["input_size"][1],
                                 d2_offset:d2_offset+train_dict["input_size"][2]
                                 ]
                batch_x[idx_batch, 0, :, :, :] = x_slice
                batch_y[idx_batch, 0, :, :, :] = y_slice

            batch_x = torch.from_numpy(batch_x).float().to(device)
            batch_y = torch.from_numpy(batch_y).float().to(device)
                
            optimizer.zero_grad()
            y_hat = model(batch_x)

            # save the y_hat for observation
            y_hat = y_hat.cpu().detach()
            torch.save(y_hat, train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_y_hat.pt")
            exit()


            loss = criterion(y_hat, batch_y)
            if isTrain:
                loss.backward()
                optimizer.step()
            case_loss[cnt_file] = loss.item()
            print("Loss: ", case_loss[cnt_file])

        print(iter_tag + " ===>===> Epoch[{:03d}]: ".format(idx_epoch+1), end='')
        print("  Loss: ", np.mean(case_loss))
        np.save(train_dict["save_folder"]+"loss/epoch_loss_"+iter_tag+"_{:03d}.npy".format(idx_epoch+1), case_loss)

        if isVal:
            # np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_x.npy", batch_x.cpu().detach().numpy())
            # np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_y.npy", batch_y.cpu().detach().numpy())
            # np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_z.npy", y_hat.cpu().detach().numpy())

            # torch.save(model, train_dict["save_folder"]+"model_curr.pth".format(idx_epoch + 1))
            torch.save(model.state_dict(), train_dict["save_folder"]+"model_curr.pth")
            torch.save(optimizer.state_dict(), train_dict["save_folder"]+"optim_curr.pth")
            if np.mean(case_loss) < best_val_loss:
                # save the best model
                torch.save(model.state_dict(), train_dict["save_folder"]+"model_best_{:03d}.pth".format(idx_epoch + 1))
                torch.save(optimizer.state_dict(), train_dict["save_folder"]+"optim_{:03d}.pth".format(idx_epoch + 1))
                print("Checkpoint saved at Epoch {:03d}".format(idx_epoch + 1))
                best_val_loss = np.mean(case_loss)

        # del batch_x, batch_y
        # gc.collect()
        # torch.cuda.empty_cache()
