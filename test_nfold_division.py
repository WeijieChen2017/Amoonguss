import json
import os
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random

def create_folds(data_json, nfold, random_seed):
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

    # Create training and validation sets for each fold
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_cases = [case_ids[i] for i in train_idx]
        val_cases = [case_ids[i] for i in val_idx]

        train_data = {key: [path for path in data[key] if os.path.dirname(path) in train_cases] for key in data}
        val_data = {key: [path for path in data[key] if os.path.dirname(path) in val_cases] for key in data}

        # Save training and validation sets as JSON files
        fold_data = {
            "training": train_data,
            "validation": val_data
        }

        with open(f"fold_{fold_idx + 1}.json", "w") as outfile:
            json.dump(fold_data, outfile)

# Example usage
data_json = "./data_dir/Task1/brain.json"
nfold = 5
random_seed = 912
create_folds(data_json, nfold, random_seed)
