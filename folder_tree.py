import os

def build_tree(folder, level=0):
    entries = sorted(os.listdir(folder))

    for entry in entries:
        entry_path = os.path.join(folder, entry)
        if os.path.isdir(entry_path):
            print("  " * level + entry)
            build_tree(entry_path, level + 1)

# Replace 'your_folder_path' with the path to your folder
your_folder_path = './data_dir/Task1/'
build_tree(your_folder_path)