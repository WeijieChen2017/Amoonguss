import os
import time

model_list = [
    ["Quaxly_brain_v1", [4], 912, 5],
    ["Quaxly_pelvis_v1", [4], 912, 5],
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

gpu_list = ','.join(str(x) for x in train_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dict["loss_term"] = "SmoothL1Loss"
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

unet_dict = {}
unet_dict["spatial_dims"] = 3
unet_dict["in_channels"] = 1
unet_dict["out_channels"] = 1
# unet_dict["channels"] = (40, 80, 160, 320)
unet_dict["strides"] = (2, 2, 2)
unet_dict["num_res_units"] = 6

train_dict["model_para"] = unet_dict

train_dict["opt_betas"] = (0.9, 0.999) # default
train_dict["opt_eps"] = 1e-8 # default
train_dict["opt_weight_decay"] = 0.01 # default
train_dict["amsgrad"] = False # default

for path in [train_dict["save_folder"], train_dict["save_folder"]+"npy/", train_dict["save_folder"]+"loss/"]:
    if not os.path.exists(path):
        os.mkdir(path)


import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    Spacingd,
    RandRotate90d,
)
from util import (
    CustomNormalize,
    AddRicianNoise,
    create_nfold_json,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

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
        CustomNormalize(
            keys_group1=["MR"],
            a_min_group1=[0],
            a_max_group1=[3000],
            b_min_group1=[0],
            b_max_group1=[1],
            keys_group2=["CT"],
            a_min_group2=[-1024],
            a_max_group2=[3000],
            b_min_group2=[0],
            b_max_group2=[1],
        ),
        AddRicianNoise(keys=["MR"], noise_std=0.01),
        CropForegroundd(
            keys=["MR", "CT", "MASK"],
            source_key="MASK",
            margin=(0, 0, 0),
            select_fn=lambda x: x != 0,
            return_transform=False,
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
            keys=["image", "label"],
            pixdim=(1, 1, 1),
            mode=("bilinear", "bilinear", "nearest"),
        ),
        CustomNormalize(
            keys_group1=["MR"],
            a_min_group1=[0],
            a_max_group1=[3000],
            b_min_group1=[0],
            b_max_group1=[1],
            keys_group2=["CT"],
            a_min_group2=[-1024],
            a_max_group2=[3000],
            b_min_group2=[0],
            b_max_group2=[1],
        ),
        CropForegroundd(
            keys=["MR", "CT", "MASK"],
            source_key="MASK",
            margin=(0, 0, 0),
            select_fn=lambda x: x != 0,
            return_transform=False,
        ),
    ]
)

data_dir = "./data_dir/Task1/"
data_json = data_dir+"brain.json" if train_dict["organ"] == "brain" else "pelvis.json"
create_nfold_json(data_json, train_dict["num_fold"], train_dict["random_seed"])


split_json = "dataset_0.json"

datasets = data_dir + split_json
datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")
train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=24,
    cache_rate=1.0,
    num_workers=8,
)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)


slice_map = {
    "img0035.nii.gz": 170,
    "img0036.nii.gz": 230,
    "img0037.nii.gz": 204,
    "img0038.nii.gz": 204,
    "img0039.nii.gz": 204,
    "img0040.nii.gz": 180,
}
case_num = 0
img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
img = val_ds[case_num]["image"]
label = val_ds[case_num]["label"]
img_shape = img.shape
label_shape = label.shape
print(f"image shape: {img_shape}, label shape: {label_shape}")
plt.figure("image", (18, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(img[0, :, :, slice_map[img_name]].detach().cpu(), cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[0, :, :, slice_map[img_name]].detach().cpu())
plt.show()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNETR(
    in_channels=1,
    out_channels=14,
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)


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


max_iterations = 25000
eval_num = 500
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))


print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")


plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [eval_num * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.show()


case_num = 4
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
with torch.no_grad():
    img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
    img = val_ds[case_num]["image"]
    label = val_ds[case_num]["label"]
    val_inputs = torch.unsqueeze(img, 1).cuda()
    val_labels = torch.unsqueeze(label, 1).cuda()
    val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.8)
    plt.figure("check", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("image")
    plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("label")
    plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
    plt.subplot(1, 3, 3)
    plt.title("output")
    plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]])
    plt.show()


if directory is None:
    shutil.rmtree(root_dir)