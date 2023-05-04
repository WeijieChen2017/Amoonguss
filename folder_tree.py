import os

def build_tree(folder, file_ext=".nii.gz", level=0):
    for entry in sorted(os.listdir(folder)):
        entry_path = os.path.join(folder, entry)
        if os.path.isfile(entry_path) and entry.endswith(file_ext):
            print("  " * level + entry)
        elif os.path.isdir(entry_path):
            print("  " * level + entry)
            build_tree(entry_path, file_ext, level + 1)

# Replace 'your_folder_path' with the path to your folder
your_folder_path = './data_dir/Task1'
build_tree(your_folder_path)
