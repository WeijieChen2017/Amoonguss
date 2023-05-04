import os
import json

def build_tree(folder, data, category):
    for entry in sorted(os.listdir(folder)):
        entry_path = os.path.join(folder, entry)
        if os.path.isfile(entry_path):
            for key, file_name in zip(["MR", "CT", "MASK"], ["mr.nii.gz", "ct.nii.gz", "mask.nii.gz"]):
                if entry.lower() == file_name:
                    data[key].append(entry_path)
        elif os.path.isdir(entry_path):
            if category in entry:
                build_tree(entry_path, data, category)

# Replace 'your_folder_path' with the path to your folder
your_folder_path = './data_dir/Task1'

brain_data = {"MR": [], "CT": [], "MASK": []}
pelvis_data = {"MR": [], "CT": [], "MASK": []}

build_tree(your_folder_path, brain_data, category="brain")
build_tree(your_folder_path, pelvis_data, category="pelvis")

# Save brain_data and pelvis_data as JSON files
with open("brain.json", "w") as outfile:
    json.dump(brain_data, outfile)

with open("pelvis.json", "w") as outfile:
    json.dump(pelvis_data, outfile)