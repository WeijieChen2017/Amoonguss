import os
import time
import numpy as np

model_list = [
    ["Quaxly_brain_v1", [4], 912, 5, 0],
    ["Quaxly_pelvis_v1", [4], 912, 5, 0],
    ["Quaxly_brain_v1", [4], 912, 5, 1],
    ["Quaxly_pelvis_v1", [4], 912, 5, 1],
    ["Quaxly_brain_v1", [4], 912, 5, 2],
    ["Quaxly_pelvis_v1", [4], 912, 5, 2],
    ["Quaxly_brain_v1", [4], 912, 5, 3],
    ["Quaxly_pelvis_v1", [4], 912, 5, 3],
    ["Quaxly_brain_v1", [4], 912, 5, 4],
    ["Quaxly_pelvis_v1", [4], 912, 5, 4],
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
train_dict["input_size"] = [96, 96, 96]

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
train_dict["continue_training_epoch"] = 0
# train_dict["batch"] = train_dict["GROWTH_epochs"][3]["batch"]
train_dict["start_batch"] = 32
train_dict["batch_decay"] = 500 # batch into half every 500 epochs
unet_dict = {}
unet_dict["spatial_dims"] = 3
unet_dict["in_channels"] = 1
unet_dict["out_channels"] = 1
unet_dict["channels"] = train_dict["GROWTH_epochs"][3]["model_channels"]
unet_dict["strides"] = (2, 2, 2)
unet_dict["num_res_units"] = 6

train_dict["model_para"] = unet_dict

train_dict["opt_betas"] = (0.9, 0.999) # default
train_dict["opt_eps"] = 1e-8 # default
train_dict["opt_lr"] = train_dict["GROWTH_epochs"][3]["lr"]
train_dict["opt_weight_decay"] = 0.01 # default
train_dict["amsgrad"] = False # default

for path in [train_dict["save_folder"], train_dict["save_folder"]+"npy/", train_dict["save_folder"]+"loss/"]:
    if not os.path.exists(path):
        os.mkdir(path)


from torch.nn import SmoothL1Loss
from model import UNet_GROWTH


import os
import json
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm


from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    ScaleIntensityRanged,
    Orientationd,
    RandFlipd,
    Spacingd,
    RandRotate90d,
    RandSpatialCropd,
)
from util import (
    CustomNormalize,
    AddRicianNoise,
    create_nfold_json,
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
        LoadImaged(keys=["MR", "CT", "MASK"]),
        EnsureChannelFirstd(keys=["MR", "CT", "MASK"]),
        Orientationd(keys=["MR", "CT", "MASK"], axcodes="RAS"),
        Spacingd(
            keys=["MR", "CT", "MASK"],
            pixdim=(1., 1, 1),
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
        CropForegroundd(
            keys=["MR", "CT", "MASK"],
            source_key="MASK",
            margin=(0, 0, 0),
            select_fn=lambda x: x != 0,
            return_transform=False,
        ),
        RandSpatialCropd(
            keys=["MR", "CT", "MASK"],
            roi_size=(64, 64, 64), 
            random_size=False,
        ),
        RandFlipd(
            keys=["MR", "CT", "MASK"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["MR", "CT", "MASK"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["MR", "CT", "MASK"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["MR", "CT", "MASK"],
            prob=0.10,
            max_k=3,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["MR", "CT", "MASK"]),
        EnsureChannelFirstd(keys=["MR", "CT", "MASK"]),
        Orientationd(keys=["MR", "CT", "MASK"], axcodes="RAS"),
        Spacingd(
            keys=["MR", "CT", "MASK"],
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
        CropForegroundd(
            keys=["MR", "CT", "MASK"],
            source_key="MASK",
            margin=(0, 0, 0),
            select_fn=lambda x: x != 0,
            return_transform=False,
        ),
        RandSpatialCropd(
            keys=["MR", "CT", "MASK"],
            roi_size=(64, 64, 64), 
            random_size=False,
        ),
    ]
)

data_dir = "./data_dir/Task1/"
data_json = data_dir+"brain.json" if train_dict["organ"] == "brain" else "pelvis.json"
if train_dict["current_fold"] == 0:
    create_nfold_json(data_json, train_dict["num_fold"], train_dict["random_seed"], train_dict["save_folder"])

# n_stage = len(train_dict["GROWTH_epochs"])
n_fold = train_dict["num_fold"]
curr_fold = train_dict["current_fold"]

split_json = root_dir + f"fold_{curr_fold + 1}.json"
# with open(data_json, "r") as f:
#     datasets = json.load(f)

train_files = load_decathlon_datalist(split_json, True, "training")
val_files = load_decathlon_datalist(split_json, True, "validation")
train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    # cache_num=24,
    cache_rate=1.0,
    num_workers=8,
)
val_ds = CacheDataset(
    data=val_files, 
    transform=val_transforms, 
    # cache_num=6, 
    cache_rate=1.0, 
    num_workers=4,
)

model = UNet_GROWTH( 
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

criterion = SmoothL1Loss()

# print("Test successful, now exiting...")
# exit()

# build new dataloader at epoch 0, 500, 1000, 1500, 2000, 2500

best_val_loss = 1e1
best_epoch = 0
model.to(device)

for idx_epoch_new in range(train_dict["train_epochs"]):
    idx_epoch = idx_epoch_new + train_dict["continue_training_epoch"]
    print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))

    # check the idx_epoch to determine the batch size
    if idx_epoch in [0, 500, 1000, 1500, 2000, 2500]:
        batch_stage = 5 - idx_epoch // train_dict["batch_decay"]
        batch_size = 2 ** batch_stage
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # training
    model.train()
    n_train_files = len(train_loader)
    case_loss = np.zeros((n_train_files, 1))
    for step, batch in enumerate(train_loader):
        step += 1
        # mr, ct, mask = (batch["MR"].cuda(), batch["CT"].cuda(), batch["MASK"].cuda())
        mr, ct, mask = (batch["MR"], batch["CT"], batch["MASK"])
        print("step[", step, "]mr", mr.shape, "ct", ct.shape, "mask", mask.shape)
        if step == 10:
            exit()



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
        
        case_loss = np.zeros((len(file_list), 2))

        # N, C, D, H, W
        x_data = nib.load(file_list[0]).get_fdata()

        for cnt_file, file_path in enumerate(file_list):
            
            x_path = file_path
            y_path = file_path.replace("mr", "ct")
            file_name = os.path.basename(file_path)
            print(iter_tag + " ===> Epoch[{:03d}]:[{:03d}]/[{:03d}] --->".format(idx_epoch+1, cnt_file+1, n_file), x_path, "<---", end="")
            x_file = nib.load(x_path)
            y_file = nib.load(y_path)
            x_data = x_file.get_fdata()
            y_data = y_file.get_fdata()

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
            
            if isTrain:

                optim.zero_grad()
                y_hat = model(batch_x)
                y_ref = model(batch_x)
                loss_recon = loss_fnc(y_hat, batch_y)
                loss_rdrop = loss_doc(y_ref, y_hat)
                loss = loss_recon + loss_rdrop * train_dict["alpha_dropout_consistency"]
                loss.backward()
                optim.step()
                case_loss[cnt_file, 0] = loss_recon.item()
                case_loss[cnt_file, 1] = loss_rdrop.item()
                print("Loss: ", np.sum(case_loss[cnt_file, :]), "Recon: ", loss_recon.item(), "Rdropout: ", loss_rdrop.item())

            if isVal:

                with torch.no_grad():
                    y_hat = model(batch_x)
                    y_ref = model(batch_x)
                    loss_recon = loss_fnc(y_hat, batch_y)
                    loss_rdrop = loss_doc(y_ref, y_hat)
                    loss = loss_recon + loss_rdrop * train_dict["alpha_dropout_consistency"]

                case_loss[cnt_file, 0] = loss_recon.item()
                case_loss[cnt_file, 1] = loss_rdrop.item()
                print("Loss: ", np.sum(case_loss[cnt_file, :]), "Recon: ", loss_recon.item(), "Rdropout: ", loss_rdrop.item())

        epoch_loss_recon = np.mean(case_loss[:, 0])
        epoch_loss_rdrop = np.mean(case_loss[:, 1])
        # epoch_loss = np.mean(case_loss)
        epoch_loss = epoch_loss_recon
        print(iter_tag + " ===>===> Epoch[{:03d}]: ".format(idx_epoch+1), end='')
        print("Loss: ", epoch_loss, "Recon: ", epoch_loss_recon, "Rdropout: ", epoch_loss_rdrop)
        np.save(train_dict["save_folder"]+"loss/epoch_loss_"+iter_tag+"_{:03d}.npy".format(idx_epoch+1), case_loss)

        if isVal:
            # np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_xf.npy", batch_xf.cpu().detach().numpy())
            # np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_fmap.npy", batch_fmap.cpu().detach().numpy())
            torch.save(model.state_dict(), train_dict["save_folder"]+"model_curr.pth")
            torch.save(optim.state_dict(), train_dict["save_folder"]+"optim_curr.pth")
            
            if epoch_loss < best_val_loss:
                # save the best model
                torch.save(model.state_dict(), train_dict["save_folder"]+"model_best_{:03d}.pth".format(idx_epoch + 1))
                torch.save(optim.state_dict(), train_dict["save_folder"]+"optim_best_{:03d}.pth".format(idx_epoch + 1))
                print("Checkpoint saved at Epoch {:03d}".format(idx_epoch + 1))
                best_val_loss = epoch_loss

        # del batch_x, batch_y
        # gc.collect()
        # torch.cuda.empty_cache()



def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best





# max_iterations = 25000
# eval_num = 500
# post_label = AsDiscrete(to_onehot=14)
# post_pred = AsDiscrete(argmax=True, to_onehot=14)
# dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
# global_step = 0
# dice_val_best = 0.0
# global_step_best = 0
# epoch_loss_values = []
# metric_values = []
# while global_step < max_iterations:
#     global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))


# print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")


# plt.figure("train", (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Iteration Average Loss")
# x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
# y = epoch_loss_values
# plt.xlabel("Iteration")
# plt.plot(x, y)
# plt.subplot(1, 2, 2)
# plt.title("Val Mean Dice")
# x = [eval_num * (i + 1) for i in range(len(metric_values))]
# y = metric_values
# plt.xlabel("Iteration")
# plt.plot(x, y)
# plt.show()


# case_num = 4
# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
# model.eval()
# with torch.no_grad():
#     img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
#     img = val_ds[case_num]["image"]
#     label = val_ds[case_num]["label"]
#     val_inputs = torch.unsqueeze(img, 1).cuda()
#     val_labels = torch.unsqueeze(label, 1).cuda()
#     val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.8)
#     plt.figure("check", (18, 6))
#     plt.subplot(1, 3, 1)
#     plt.title("image")
#     plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
#     plt.subplot(1, 3, 2)
#     plt.title("label")
#     plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
#     plt.subplot(1, 3, 3)
#     plt.title("output")
#     plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]])
#     plt.show()


# if directory is None:
#     shutil.rmtree(root_dir)