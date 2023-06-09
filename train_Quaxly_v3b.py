# brain Maximum size: [280 284 262]
# brainBrain Minimum size: [173. 224. 164.]
# pelvis Maximum size: [586 410 153]
# pelvisBrain Minimum size: [390. 248.  84.]

import os
import time
import numpy as np

model_list = [
    ["Quaxly_brain_v3b", [7], 912, 6, 0, "brain"], # ongoing
    ["Quaxly_brain_v3b", [4], 912, 6, 1, "brain"], # ongoing
    ["Quaxly_brain_v3b", [4], 912, 6, 2, "brain"], # ongoing best at 3500
    ["Quaxly_brain_v3b", [5], 912, 6, 3, "brain"], # ongoing
    ["Quaxly_brain_v3b", [4], 912, 6, 4, "brain"], # ongoing
    ["Quaxly_brain_v3b", [3], 912, 6, 5, "brain"], # dgx1
    # ["Quaxly_task1_v4", [7], 912, 6, 0, "task1"], #dgx2 runing
    # ["Quaxly_task1_v4", [7], 912, 6, 1, "task1"], #dgx2 runing
    # ["Quaxly_task1_v4", [3], 912, 6, 2, "task1"], #dgx1
    # ["Quaxly_task1_v4", [3], 912, 6, 3, "task1"], #dgx1
    # ["Quaxly_task1_v4", [5], 912, 6, 5, "task1"], #dgx1 runing
    # ["Quaxly_task1_v4", [5], 912, 6, 6, "task1"], #dgx1
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
train_dict["num_fold"] = model_list[current_model_idx][3]
train_dict["current_fold"] = model_list[current_model_idx][4]
train_dict["organ"] = model_list[current_model_idx][5]

gpu_list = ','.join(str(x) for x in train_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dict["optimizer"] = "AdamW"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["input_size"] = (64, 64, 64)

# train_dict["GROWTH_epochs"] = [
#     {"stage": 0, "model_channels": (8, 16, 32, 64), "epochs" : 25, "batch" : 32, "lr": 1e-3, "loss": "l2",},
#     {"stage": 1, "model_channels": (16, 32, 64, 128), "epochs" : 50, "batch" : 32, "lr": 7e-4, "loss": "l2",},
#     {"stage": 2, "model_channels": (24, 48, 96, 192), "epochs" : 75, "batch" : 16, "lr": 5e-4, "loss": "l1",},
#     {"stage": 3, "model_channels": (32, 64, 128, 256), "epochs" : 100, "batch" : 16, "lr": 3e-4, "loss": "l1",},
#     {"stage": 4, "model_channels": (40, 80, 160, 320), "epochs" : 150, "batch" : 8, "lr": 1e-4, "loss": "l1",},    
# ]

# train_dict["train_epochs"] = train_dict["GROWTH_epochs"][3]["epochs"]
train_dict["train_epochs"] = 12000
train_dict["eval_per_epochs"] = 100
train_dict["save_per_epochs"] = 2000
train_dict["sample_per_epochs"] = 1000
train_dict["continue_training_epoch"] = 0
# train_dict["batch"] = train_dict["GROWTH_epochs"][3]["batch"]
# train_dict["start_batch"] = 16
train_dict["batch_decay"] = 2000 # batch into half every 500 epochs
unet_dict = {}
unet_dict["spatial_dims"] = 3
unet_dict["in_channels"] = 2
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, _LRScheduler

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
    create_nfold_json_v3b,
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
        LoadImaged(keys=["MR", "CT", "v3a_last"]),
        EnsureChannelFirstd(keys=["MR", "CT", "v3a_last"]),
        Orientationd(keys=["MR", "CT", "v3a_last"], axcodes="RAS"),
        Spacingd(
            keys=["MR", "CT", "v3a_last"],
            pixdim=(1., 1, 1),
            mode=("bilinear", "bilinear", "bilinear"),
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
        ScaleIntensityRanged(
            keys=["v3a_last"],
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
            keys=["MR", "CT", "v3a_last"],
            num_samples = 2, 
            roi_size=train_dict["input_size"], 
            random_size=False,
        ),
        RandFlipd(
            keys=["MR", "CT", "v3a_last"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["MR", "CT", "v3a_last"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["MR", "CT", "v3a_last"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["MR", "CT", "v3a_last"],
            prob=0.10,
            max_k=3,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["MR", "CT", "v3a_last"]),
        EnsureChannelFirstd(keys=["MR", "CT", "v3a_last"]),
        Orientationd(keys=["MR", "CT", "v3a_last"], axcodes="RAS"),
        Spacingd(
            keys=["MR", "CT", "v3a_last"],
            pixdim=(1, 1, 1),
            mode=("bilinear", "bilinear", "bilinear"),
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
        ScaleIntensityRanged(
            keys=["v3a_last"],
            a_min=-1024,
            a_max=3000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        SpatialPadd(
            keys=["MR", "CT", "v3a_last"],
            spatial_size=(288, 288, 288), #if train_dict["organ"] == "brain" else (640, 440, 160),
            # spatial_size=(640, 440, 288),
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
curr_fold = train_dict["current_fold"]
organ = train_dict["organ"]
# data_json = data_dir+"brain.json" if train_dict["organ"] == "brain" else data_dir+"pelvis.json"
# print("data_json: ", data_json)
# curr_fold = train_dict["current_fold"]
# if train_dict["current_fold"] == 0:
#     create_nfold_json(data_json, train_dict["num_fold"], train_dict["random_seed"], train_dict["save_folder"])
if train_dict["current_fold"] == 0 and not os.path.exists(root_dir + f"fold_0.json"):
    data_json = data_dir+"brain_v3a.json"
    print("data_json: ", data_json)
    create_nfold_json_v3b(data_json, train_dict["num_fold"], train_dict["random_seed"], train_dict["save_folder"])

# n_stage = len(train_dict["GROWTH_epochs"])
n_fold = train_dict["num_fold"]
curr_fold = train_dict["current_fold"]


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

def custom_coefficient(x: int) -> float:
    # return 10^(-x/3750) using numpy
    # (2000*5+2000*10+2000*19+2000*38+2000*76) ~= 300000 = 3e5
    return np.power(10, -x/1.5e5)

class CustomCosineAnnealingWarmRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
        self.base_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult, eta_min, last_epoch, verbose)
        super(CustomCosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # Get the learning rates from the base scheduler
        base_lrs = self.base_scheduler.get_lr()

        # Multiply the learning rates with the custom coefficient
        coeff = custom_coefficient(self.last_epoch)
        return [lr * coeff for lr in base_lrs]

    def step(self, epoch=None):
        # Update the base scheduler's epoch counter
        self.base_scheduler.step(epoch)
        super().step(epoch)

# Create the custom scheduler
T_0 = 500  # The number of epochs for the first restart
scheduler = CustomCosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=1e-4)

criterion = SmoothL1Loss()

# print("Test successful, now exiting...")
# exit()

# build new dataloader at epoch 0, 500, 1000, 1500, 2000, 2500

best_val_loss = 1000
best_epoch = 0
model.cuda()

# I need to check the cuda device info
print("cuda device info: ", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0))

for idx_epoch_new in range(train_dict["train_epochs"]):
    idx_epoch = idx_epoch_new + train_dict["continue_training_epoch"]
    print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))

    # check the idx_epoch to determine the batch size
    #                 32,   16,    8,    4,    2,     1
    if idx_epoch in [0, 2000, 4000, 6000, 8000, 10000, ]:
        batch_stage = 5 - idx_epoch // train_dict["batch_decay"]
        batch_size = 2 ** batch_stage
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # training
    model.train()
    curr_iter = n_train_files // batch_size + 1
    print("Training: ", curr_iter, "iterations")
    case_loss = np.zeros((curr_iter, 1))
    for step, batch in enumerate(train_loader):
        mr, ct, v3a_last = (batch["MR"].float().cuda(),
                            batch["CT"].float().cuda(),
                            batch["v3a_last"].float().cuda())
        # mr, ct, mask = (batch["MR"], batch["CT"], batch["MASK"])
        # print("step[", step, "]mr", mr.shape, "ct", ct.shape, "mask", mask.shape)
        print(" ===> Train:Epoch[{:03d}]:[{:03d}]/[{:03d}] --->".format(idx_epoch+1, step, curr_iter), end="")
            
        optimizer.zero_grad()
        input_mr_v3a_last = torch.concat((mr, v3a_last), dim=1).float().cuda()
        sct, ds_1, ds_2, ds_3 = model(input_mr_v3a_last, is_deep_supervision=True)
        loss_out = criterion(ct, sct)
        loss_ds_1 = criterion(ct, ds_1)
        loss_ds_2 = criterion(ct, ds_2)
        loss_ds_3 = criterion(ct, ds_3)
        loss = loss_out + loss_ds_1 + loss_ds_2 + loss_ds_3
        # final_loss = torch.sum(loss * mask) / torch.sum(mask)
        final_loss = loss
        final_loss.backward()
        optimizer.step()
        case_loss[step] = final_loss.item()
        print("Loss: ", case_loss[step], end="")
        np.save(train_dict["save_folder"]+"loss/fold_{:02d}_train_{:04d}.npy".format(curr_fold, idx_epoch+1), case_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(f" lr:{current_lr}")
        scheduler.step()
        step += 1

    # validation
    if (idx_epoch+1) % train_dict["eval_per_epochs"] == 0:
        model.eval()
        curr_iter = n_val_files
        print("Validation: ", curr_iter, "iterations")
        case_loss = np.zeros((curr_iter, 1))
        for step, batch in enumerate(val_loader):
            mr, ct, v3a_last = (batch["MR"].float().cuda(),
                                batch["CT"].float().cuda(),
                                batch["v3a_last"].float().cuda())
            # mr, ct, mask = (batch["MR"], batch["CT"], batch["MASK"])
            # print("step[", step, "]mr", mr.shape, "ct", ct.shape, "mask", mask.shape)
            print(" ===> Validation: Epoch[{:03d}]:[{:03d}]/[{:03d}] --->".format(idx_epoch+1, step, curr_iter), end="")
            input_mr_v3a_last = torch.concat((mr, v3a_last), dim=1).float().cuda()

            with torch.no_grad():
                sct = sliding_window_inference(
                inputs = input_mr_v3a_last, 
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
                # final_loss = torch.sum(loss * mask) / torch.sum(mask)
                final_loss = loss
                case_loss[step] = final_loss.item()
            print("Loss: ", case_loss[step])
            np.save(train_dict["save_folder"]+"loss/fold_{:02d}_val_{:04d}.npy".format(curr_fold, idx_epoch+1), case_loss)
            step += 1

        curr_mae = np.mean(case_loss) * 4024
        print("Validation MAE: ", curr_mae, "Best MAE: ", best_val_loss, "Best Epoch: ", best_epoch)
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
        print("Model saved at epoch {:03d}".format(idx_epoch+1))

    # save the sample every train_dict["save_per_epochs"] epochs
    if (idx_epoch+1) % train_dict["save_per_epochs"] == 0:
        mr_cache = mr.detach().cpu().numpy()
        ct_cache = ct.detach().cpu().numpy()
        sct_cache = sct.detach().cpu().numpy()
        sample_cache = {
            "MR": mr_cache,
            "CT": ct_cache,
            "SCT": sct_cache,
        }
        np.save(train_dict["save_folder"]+"sample_cache/fold_{:02d}_sample_{:04d}.npy".format(curr_fold, idx_epoch+1), sample_cache)
        print("Sample saved at epoch {:03d}".format(idx_epoch+1))
    

print("Training finished!")
print("The best model is saved at epoch {:03d} with MAE {:03f}".format(best_epoch, best_val_loss))
# def validation(epoch_iterator_val):
#     model.eval()
#     with torch.no_grad():
#         for batch in epoch_iterator_val:
#             val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
#             val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
#             val_labels_list = decollate_batch(val_labels)
#             val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
#             val_outputs_list = decollate_batch(val_outputs)
#             val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
#             dice_metric(y_pred=val_output_convert, y=val_labels_convert)
#             epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
#         mean_dice_val = dice_metric.aggregate().item()
#         dice_metric.reset()
#     return mean_dice_val


# def train(global_step, train_loader, dice_val_best, global_step_best):
#     model.train()
#     epoch_loss = 0
#     step = 0
#     epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
#     for step, batch in enumerate(epoch_iterator):
#         step += 1
#         x, y = (batch["image"].cuda(), batch["label"].cuda())
#         logit_map = model(x)
#         loss = loss_function(logit_map, y)
#         loss.backward()
#         epoch_loss += loss.item()
#         optimizer.step()
#         optimizer.zero_grad()
#         epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))
#         if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
#             epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
#             dice_val = validation(epoch_iterator_val)
#             epoch_loss /= step
#             epoch_loss_values.append(epoch_loss)
#             metric_values.append(dice_val)
#             if dice_val > dice_val_best:
#                 dice_val_best = dice_val
#                 global_step_best = global_step
#                 torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
#                 print(
#                     "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
#                 )
#             else:
#                 print(
#                     "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
#                         dice_val_best, dice_val
#                     )
#                 )
#         global_step += 1
#     return global_step, dice_val_best, global_step_best





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