import numpy as np
import matplotlib.pyplot as plt
import glob
import copy
import os
import time

n_fold = 6
folder = "./project_dir/Quaxly_brain_v3a/"

fold_hub = {}
val_epoch = []
for idx_fold in range(n_fold):
    fold_hub[idx_fold] = {"train":[], "val":{}}
    train_npy_list = sorted(glob.glob(folder+"loss/fold_{:02d}_train_*.npy".format(idx_fold)))
    val_npy_list = sorted(glob.glob(folder+"loss/fold_{:02d}_val_*.npy".format(idx_fold)))
    for npy_path in train_npy_list:
        # print(npy_path)
        data = np.load(npy_path)
        MAE_HU = np.mean(data) * 4024 - 1024
        fold_hub[idx_fold]["train"].append(MAE_HU)
    print("Fold {:02d}:".format(idx_fold), "Epoch_train:", len(fold_hub[idx_fold]["train"]))

    min_epoch = 0
    for npy_path in val_npy_list:
        # print(npy_path)
        data = np.load(npy_path)
        # fold_03_val_9900.npy
        epoch = int(os.path.basename(npy_path).split("_")[3].split(".")[0])
        min_epoch = epoch
        if not epoch in val_epoch:
            val_epoch.append(epoch)
        MAE_HU = np.mean(data) * 4024 - 1024
        fold_hub[idx_fold]["val"][epoch] = MAE_HU
    print("Fold {:02d}:".format(idx_fold), "Epoch_val:", max(min_epoch))

timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
savename = folder + "overall_loss_{:02d}_{}.jpg".format(n_fold, timestamp)

    

# # npy_list = sorted(glob.glob(folder+"loss/epoch_loss_*.npy"))
# for npy_path in npy_list:
#     print(npy_path)
#     stage_name = os.path.basename(npy_path)
#     stage_name = stage_name.split("_")[2]
#     print(stage_name)
#     if not stage_name in stage_hub:
#         stage_hub.append(stage_name)

# loss = np.zeros((n_epoch))
# plot_target = []
# for stage_name in stage_hub:
#     current_package = [stage_name]
#     for idx in range(n_epoch):
#         num = "{:03d}".format(idx+1)
#         name = folder+"loss/epoch_loss_{}_{}.npy".format(stage_name, num)
#         data = np.load(name)
#         loss[idx] = np.mean(data)
#     current_package.append(copy.deepcopy(loss))
#     plot_target.append(current_package)

# legend_list = []
# plt.figure(figsize=(9,6), dpi=300)
# for package in plot_target:
#     loss_array = package[1]
#     loss_tag = package[0]
#     legend_list.append(loss_tag)
#     print(loss_tag, np.mean(loss_array))
#     plt.plot(range(n_epoch), loss_array)

# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.yscale("log")
# plt.legend(legend_list)
# plt.title("Training curve of "+folder.split("/")[-2])

# plt.savefig(folder + "loss_{}.jpg".format(n_epoch))