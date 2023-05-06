import os
import json
import random
from sklearn.model_selection import StratifiedKFold
import monai.transforms as transforms
import numpy as np


class CustomNormalize:
    def __init__(self, keys_group1, a_min_group1, a_max_group1, b_min_group1, b_max_group1,
                 keys_group2, a_min_group2, a_max_group2, b_min_group2, b_max_group2):
        self.keys = keys_group1 + keys_group2
        self.normalizers = {}
        
        for key, amin, amax, bmin, bmax in zip(keys_group1, a_min_group1, a_max_group1, b_min_group1, b_max_group1):
            self.normalizers[key] = transforms.ScaleIntensityRanged(
                a_min=amin, a_max=amax, b_min=bmin, b_max=bmax, clip=True
            )
            
        for key, amin, amax, bmin, bmax in zip(keys_group2, a_min_group2, a_max_group2, b_min_group2, b_max_group2):
            self.normalizers[key] = transforms.ScaleIntensityRanged(
                a_min=amin, a_max=amax, b_min=bmin, b_max=bmax, clip=True
            )

    def __call__(self, data):
        for key in self.keys:
            data[key] = self.normalizers[key](data[key])
        return data
    

class AddRicianNoise(transforms.MapTransform):
    def __init__(self, keys, noise_std):
        super().__init__(keys)
        self.noise_std = noise_std

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            noise_real = np.random.normal(loc=0, scale=self.noise_std, size=image.shape)
            noise_imag = np.random.normal(loc=0, scale=self.noise_std, size=image.shape)
            noisy_image = np.sqrt((image + noise_real) ** 2 + noise_imag ** 2)
            d[key] = noisy_image
        return d

def create_nfold_json(data_json, nfold, random_seed, save_dir):
    # Load the JSON data
    with open(data_json, "r") as f:
        data = json.load(f)

    # Extract case IDs and labels for stratification
    case_ids = []
    labels = []
    for key in ["MR", "CT", "MASK"]:
        for filepath in data[key]:
            case_id = os.path.dirname(filepath)
            if case_id not in case_ids:
                case_ids.append(case_id)
                labels.append(key)

    # Set the random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Perform stratified k-fold splitting
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=random_seed)
    folds = list(skf.split(case_ids, labels))

    # Function to group files by case
    def group_files_by_case(files, cases):
        case_dict = {case: {"MR": None, "CT": None, "MASK": None} for case in cases}
        for filepath in files:
            case_id = os.path.dirname(filepath)
            file_type = os.path.basename(filepath).lower()
            if file_type == "mr.nii.gz":
                case_dict[case_id]["MR"] = filepath
            elif file_type == "ct.nii.gz":
                case_dict[case_id]["CT"] = filepath
            elif file_type == "mask.nii.gz":
                case_dict[case_id]["MASK"] = filepath
        return list(case_dict.values())

    # Create training and validation sets for each fold
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_cases = [case_ids[i] for i in train_idx]
        val_cases = [case_ids[i] for i in val_idx]

        train_files = [path for key in data for path in data[key] if os.path.dirname(path) in train_cases]
        val_files = [path for key in data for path in data[key] if os.path.dirname(path) in val_cases]

        train_data = group_files_by_case(train_files, train_cases)
        val_data = group_files_by_case(val_files, val_cases)

        # Save training and validation sets as JSON files
        fold_data = {
            "training": train_data,
            "validation": val_data
        }

        with open(save_dir+f"fold_{fold_idx + 1}.json", "w") as outfile:
            json.dump(fold_data, outfile)

