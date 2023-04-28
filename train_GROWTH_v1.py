import os
import time

model_list = [
    ["unet_GROWTH_v1_8066", [4], "unet", 8066],
    # ["unet_v1_5541", [7], "unet", 5541],
    # ["unet_v1_7363", [7], "unet", 7363],
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

# from monai.networks.nets.unet import UNet as unet
# from monai.networks.nets.dynunet import DynUnet as  dynunet
from model import UNet_GROWTH

# ==================== dict and config ====================

train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["input_size"] = [64, 64, 64]
train_dict["epochs"] = 200
train_dict["batch"] = 32
train_dict["dropout"] = 0
train_dict["GROWTH_epochs"] = [
    {"stage": 0, "model_channels": (8, 16, 32, 64), "epochs" : 5, "batch" : 32, "lr": 1e-3, "loss": "l2",},
    {"stage": 1, "model_channels": (16, 32, 64, 128), "epochs" : 5, "batch" : 32, "lr": 1e-3, "loss": "l2",},
    {"stage": 2, "model_channels": (24, 48, 96, 192), "epochs" : 10, "batch" : 16, "lr": 5e-4, "loss": "l1",},
    {"stage": 3, "model_channels": (32, 64, 128, 256), "epochs" : 10, "batch" : 16, "lr": 5e-4, "loss": "l1",},
    {"stage": 4, "model_channels": (40, 80, 160, 320), "epochs" : 15, "batch" : 8, "lr": 1e-4, "loss": "l1",},    
]

train_dict["model_related"] = {}
train_dict["model_related"]["spatial_dims"] = 3
train_dict["model_related"]["in_channels"] = 1
train_dict["model_related"]["out_channels"] = 1
train_dict["model_related"]["channels"] = (32, 64, 128, 256)
train_dict["model_related"]["strides"] = (2, 2, 2)
train_dict["model_related"]["num_res_units"] = 6
            


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

train_list = train_list[:10]
val_list = val_list[:5]

# ==================== training ====================

first_stage = True
best_val_loss = 100
best_epoch = 0
# wandb.watch(model)

package_train = [train_list, True, False, "train"]
package_val = [val_list, False, True, "val"]
# package_test = [test_list, False, False, "test"]

for training_dict in train_dict["GROWTH_epochs"]:

    train_stage = training_dict["stage"]
    train_channels = training_dict["model_channels"]
    train_epochs = training_dict["epochs"]
    train_batch = training_dict["batch"]
    train_lr = training_dict["lr"]
    train_loss = training_dict["loss"]

    if not first_stage:
        model.to("cpu")
        del model
        torch.cuda.empty_cache()

    model = UNet_GROWTH( 
        spatial_dims=train_dict["model_related"]["spatial_dims"],
        in_channels=train_dict["model_related"]["in_channels"],
        out_channels=train_dict["model_related"]["out_channels"],
        channels=train_channels,
        strides=train_dict["model_related"]["strides"],
        num_res_units=train_dict["model_related"]["num_res_units"]
        )
    
    if not first_stage:
        before_list = sorted(glob.glob(train_dict["save_folder"]+"stage_{:03d}_model_*.pth".format(train_stage-1)))
        before_path = before_list[-1]
        before_state_dict = torch.load(before_path)
        new_state_dict = model.state_dict()
        for key in new_state_dict.keys():
            # get the size of corresponing layer
            new_size = new_state_dict[key].size()
            before_size = before_state_dict[key].size()
            # create a new layer with the same size
            # if the key contains conv.weight or conv.bias, copy the weight from the old layer to the new layer, if size do not match, put the weight in the beginning
            if "down1.residual.weight" in key or "down1.conv.unit0.conv.weight" in key:
                new_state_dict[key][:before_state_dict[key].size()[0], :, :, :, :] = before_state_dict[key]
                # print("conv.weight", key, new_size, before_size)
            elif "conv.bias" in key or "residual.bias" or "down1.conv.unit0.conv.bias" in key: # conv.bias is a 1d tensor, the first dimension is the number of output channels            
                # print("conv.bias", key, new_size, before_size)
                new_state_dict[key][:before_state_dict[key].size()[0]] = before_state_dict[key]
            elif "conv.weight" in key or "residual.weight": # conv.weight is a 5d tensor, the first dimension is the number of output channels, the second dimension is the number of input channels
                # print("conv.weight", key, new_size, before_size)
                new_state_dict[key][:before_state_dict[key].size()[0], :before_state_dict[key].size()[1], :, :, :] = before_state_dict[key]
            else:
                new_state_dict[key] = before_state_dict[key]
                # print("else", key, new_size, before_size)

        for key in new_state_dict.keys():
            print(key, new_state_dict[key].size())
        model.load_state_dict(new_state_dict)

    first_stage = False
    model.train()
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = train_lr,
        betas = train_dict["opt_betas"],
        eps = train_dict["opt_eps"],
        weight_decay = train_dict["opt_weight_decay"],
        amsgrad = train_dict["amsgrad"]
        )

    if train_loss == "l1":
        criterion = nn.SmoothL1Loss()
    elif train_loss == "l2":
        criterion = nn.MSELoss()

    for idx_epoch_new in range(train_epochs):
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
                print(iter_tag + " ===> Stage[{:03d}]-Epoch[{:03d}]:[{:03d}]/[{:03d}] --->".format(train_stage, idx_epoch+1, cnt_file+1, n_file), x_path, "<---", end="")
                x_file = nib.load(x_path)
                y_file = nib.load(y_path)
                x_data = x_file.get_fdata()
                y_data = y_file.get_fdata()

                batch_x = np.zeros((train_batch, 1, train_dict["input_size"][0], train_dict["input_size"][1], train_dict["input_size"][2]))
                batch_y = np.zeros((train_batch, 1, train_dict["input_size"][0], train_dict["input_size"][1], train_dict["input_size"][2]))

                for idx_batch in range(train_batch):
                    
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
                loss = criterion(y_hat, batch_y)
                if isTrain:
                    loss.backward()
                    optimizer.step()
                case_loss[cnt_file] = loss.item()
                print("Loss: ", case_loss[cnt_file])

            print(iter_tag + " ===>Stage[{:03d}]-Epoch[{:03d}]: ".format(train_stage, idx_epoch+1), end='')
            print("  Loss: ", np.mean(case_loss))
            np.save(train_dict["save_folder"]+"loss/stage_{:03d}_loss_".format(train_stage)+iter_tag+"_{:03d}.npy".format(idx_epoch+1), case_loss)

            if isVal:
                # np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_x.npy", batch_x.cpu().detach().numpy())
                # np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_y.npy", batch_y.cpu().detach().numpy())
                # np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_z.npy", y_hat.cpu().detach().numpy())

                # torch.save(model, train_dict["save_folder"]+"model_curr.pth".format(idx_epoch + 1))
                torch.save(model.state_dict(), train_dict["save_folder"]+"stage_{:03d}_model_curr.pth".format(train_stage))
                torch.save(optimizer.state_dict(), train_dict["save_folder"]+"stage_{:3d}_optim_curr.pth".format(train_stage))
                if np.mean(case_loss) < best_val_loss:
                    # save the best model
                    torch.save(model.state_dict(), train_dict["save_folder"]+"stage_{:03d}_model_{:03d}.pth".format(train_stage, idx_epoch + 1))
                    torch.save(optimizer.state_dict(), train_dict["save_folder"]+"stage_{:03d}_optim_{:03d}.pth".format(train_stage, idx_epoch + 1))
                    print("Checkpoint saved at Epoch {:03d}".format(idx_epoch + 1))
                    best_val_loss = np.mean(case_loss)
                    best_epoch = idx_epoch + 1

            # del batch_x, batch_y
            # gc.collect()
            # torch.cuda.empty_cache()
