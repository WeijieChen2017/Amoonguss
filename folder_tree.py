import os
import json

def build_tree(folder, data):
    entries = sorted(os.listdir(folder))

    for entry in entries:
        entry_path = os.path.join(folder, entry)
        if os.path.isfile(entry_path) and entry.lower().endswith(".nii.gz"):
            category = "brain" if "brain" in entry_path.lower() else "pelvis"
            for key, file_name in zip(["MR", "CT", "MASK_MR"], ["mr.nii.gz", "ct.nii.gz", "mask_mri_th50.nii.gz",]):
                if entry.lower() == file_name:
                    data[category][key].append(entry_path)
        elif os.path.isdir(entry_path):
            build_tree(entry_path, data)


# Replace 'your_folder_path' with the path to your folder
your_folder_path = './data_dir/Task1/'

data = {
    "brain": {"MR": [], "CT": [], "MASK_MR": []},
    "pelvis": {"MR": [], "CT": [], "MASK_MR": []}
}

build_tree(your_folder_path, data)

# create a "task1" key in data dict to combine brain and pelvis data with the same key
data["task1"] = {key: data["brain"][key] + data["pelvis"][key] for key in ["MR", "CT", "MASK_MR"]}

# Save brain_data and pelvis_data as JSON files
# with open(your_folder_path+"brain.json", "w") as outfile:
#     json.dump(data["brain"], outfile)

# with open(your_folder_path+"pelvis.json", "w") as outfile:
#     json.dump(data["pelvis"], outfile)

with open(your_folder_path+"task1_mri_mask.json", "w") as outfile:
    json.dump(data["task1"], outfile)