import os

def build_tree(folder, file_ext=".nii.gz", level=0):
    contains_target_file = False
    entries = sorted(os.listdir(folder))
    
    for entry in entries:
        entry_path = os.path.join(folder, entry)
        if os.path.isfile(entry_path) and entry.endswith(file_ext):
            contains_target_file = True
            break
    
    if contains_target_file:
        print("  " * level + os.path.basename(folder))
    
    for entry in entries:
        entry_path = os.path.join(folder, entry)
        if os.path.isdir(entry_path):
            build_tree(entry_path, file_ext, level + 1)

# Replace 'your_folder_path' with the path to your folder
your_folder_path = './data_dir/Task1/'
build_tree(your_folder_path)