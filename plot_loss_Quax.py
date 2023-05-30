import numpy as np
import matplotlib.pyplot as plt
import glob
import copy
import os
import time

n_fold = 6
folder = "./project_dir/Quaxly_brain_v3b/"

# fold_hub = {}
# for idx_fold in range(n_fold):
#     fold_hub[idx_fold] = {"train":[], "val":{}, "val_epoch":[]}
#     train_npy_list = sorted(glob.glob(folder+"loss/fold_{:02d}_train_*.npy".format(idx_fold)))
#     val_npy_list = sorted(glob.glob(folder+"loss/fold_{:02d}_val_*.npy".format(idx_fold)))
#     for npy_path in train_npy_list:
#         # print(npy_path)
#         data = np.load(npy_path)
#         MAE_HU = np.mean(data) * 4024
#         fold_hub[idx_fold]["train"].append(MAE_HU)
#     print("Fold {:02d}:".format(idx_fold), "Epoch_train:", len(fold_hub[idx_fold]["train"]))

#     max_epoch = 0
#     for npy_path in val_npy_list:
#         # print(npy_path)
#         data = np.load(npy_path)
#         # fold_03_val_9900.npy
#         epoch = int(os.path.basename(npy_path).split("_")[3].split(".")[0])
#         min_epoch = epoch
#         if not epoch in fold_hub[idx_fold]["val_epoch"]:
#             fold_hub[idx_fold]["val_epoch"].append(epoch)
#         max_epoch = max(max_epoch, epoch)
#         MAE_HU = np.mean(data) * 4024
#         fold_hub[idx_fold]["val"][epoch] = MAE_HU
#     print("Fold {:02d}:".format(idx_fold), "Epoch_val:", max_epoch)

# timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
# savename = folder + "overall_loss_{:02d}_{}.npy".format(n_fold, timestamp)
# np.save(savename, fold_hub)

savename = folder + "overall_loss_06_20230530_014157.npy"
fold_hub = np.load(savename, allow_pickle=True).item()
print(fold_hub.keys())

legend_list = []
fold_list = fold_hub.keys()
plt.figure(figsize=(18,6), dpi=300)
for idx_fold in fold_list:
    legend_list.append("Fold{:02d}_train".format(idx_fold))
    xmesh = np.asarray(range(1, len(fold_hub[idx_fold]["train"])+1))
    data = np.asarray(fold_hub[idx_fold]["train"]) + 1024
    plt.plot(xmesh, data, alpha=0.5)
plt.xlabel("epoch")
plt.ylabel("MAE (HU)")
plt.yscale("log")
plt.legend(legend_list)
plt.savefig(folder+"overall_loss_train.png")
