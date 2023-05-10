import glob
import nibabel as nib
import numpy as np
from monai.transforms import Spacing

def process_volumes(data_dir, task, item_code):
    # Get a list of all NIfTI files
    file_list = glob.glob(f"{data_dir}/{task}/{item_code}/*/mr.nii.gz")

    # Initialize variables to track the maximum and minimum sizes
    max_size = [0, 0, 0]
    min_size = [float("inf"), float("inf"), float("inf")]

    # Create an instance of the Spacing transform
    spacing_transform = Spacing(pixdim=(1, 1, 1), mode="bilinear")

    # Iterate over the files and process each volume
    for file_path in file_list:
        # Load the volume using nibabel
        volume = nib.load(file_path)

        # Get the volume's data as a NumPy array
        data = volume.get_fdata()

        # Apply the Spacing transform
        resampled_data = spacing_transform(data)

        # Update the maximum and minimum sizes
        max_size = np.maximum(max_size, resampled_data.shape)
        min_size = np.minimum(min_size, resampled_data.shape)

    return max_size, min_size

def test_organ(organ, data_dir, task):
    item_code = organ
    max_size, min_size = process_volumes(data_dir, task, item_code)

    print(organ+" Maximum size:", max_size)
    print(organ+"Brain Minimum size:", min_size)

data_dir = "./data_dir"
task = "Task1"

test_organ("brain", data_dir, task)
test_organ("pelvis", data_dir, task)


