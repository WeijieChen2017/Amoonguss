import os

def build_tree(folder, file_ext=".nii.gz", level=0, first_n=5, last_n=3):
    contains_target_file = False
    entries = sorted(os.listdir(folder))
    
    for entry in entries:
        entry_path = os.path.join(folder, entry)
        if os.path.isfile(entry_path) and entry.endswith(file_ext):
            contains_target_file = True
            break
    
    if contains_target_file:
        folder_name = os.path.basename(folder)
        if len(folder_name) > first_n + last_n:
            folder_name = folder_name[:first_n] + "..." + folder_name[-last_n:]
        print("  " * level + folder_name)
    
    for entry in entries:
        entry_path = os.path.join(folder, entry)
        if os.path.isdir(entry_path):
            build_tree(entry_path, file_ext, level + 1, first_n, last_n)

# Replace 'your_folder_path' with the path to your folder
your_folder_path = './data_dir/Task1/'
build_tree(your_folder_path)