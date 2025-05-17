import os
import sys
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    EnsureTyped, Compose, MapTransform
)
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from scipy.ndimage import center_of_mass

set_determinism(42)
os.environ["MONAI_USE_META_DICT"] = "1"

# Check for dataset input
if len(sys.argv) < 2:
    raise ValueError(" Please provide a dataset name, e.g. `python monai_patch_extraction_safe.py Dataset003_Solid`")

datasets_to_process = [sys.argv[1]]
base_path = "/SAN/medic/nn_unet/nnUNet_raw"


# Custom transform to crop a fixed-size patch centered on the lesion
class CenteredFixedCropd(MapTransform):
    def __init__(self, keys, patch_size):
        super().__init__(keys)
        self.patch_size = patch_size

    def __call__(self, data):
        d = dict(data)
        label = d["label"][0].detach().cpu().numpy()
        com = center_of_mass(label)
        com = tuple(int(round(c)) for c in com)

        slices = []
        for i in range(3):
            half = self.patch_size[i] // 2
            start = max(com[i] - half, 0)
            end = start + self.patch_size[i]
            max_end = d["image"].shape[1 + i]
            if end > max_end:
                start = max_end - self.patch_size[i]
                end = max_end
            slices.append(slice(start, end))

        d["image"] = d["image"][:, slices[0], slices[1], slices[2]]
        d["label"] = d["label"][:, slices[0], slices[1], slices[2]]
        return d


# Define transform pipeline
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    EnsureTyped(keys=["image", "label"], track_meta=True),
    CenteredFixedCropd(keys=["image", "label"], patch_size=(96, 96, 96)),
])


# Safe NIfTI saving function
def safe_save_nifti(tensor, meta_dict, save_path):
    affine = meta_dict.get("affine", np.eye(4))
    if isinstance(affine, torch.Tensor):
        affine = affine.cpu().numpy()
    if affine.ndim != 2:
        print(f" Warning: Invalid affine shape ({affine.shape}) — resetting to identity.")
        affine = np.eye(4)

    array = tensor.detach().cpu().numpy().squeeze()
    nib.save(nib.Nifti1Image(array, affine), save_path)
    print(f"✔ Saved: {os.path.basename(save_path)}")


# Process each dataset
for dataset_name in datasets_to_process:
    print(f"\n Processing: {dataset_name}")
    dataset_path = os.path.join(base_path, dataset_name)
    output_base = os.path.join(base_path, dataset_name + "_fixed_patches")

    splits = ["imagesTr", "imagesTs"]
    data = []

    for split in splits:
        img_dir = os.path.join(dataset_path, split)
        lbl_dir = os.path.join(dataset_path, split.replace("images", "labels"))
        out_img_dir = os.path.join(output_base, split)
        out_lbl_dir = os.path.join(output_base, split.replace("images", "labels"))
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        for f in os.listdir(img_dir):
            if f.endswith("_0000.nii.gz"):
                image_path = os.path.join(img_dir, f)
                label_path = os.path.join(lbl_dir, f.replace("_0000.nii.gz", ".nii.gz"))
                out_image = os.path.join(out_img_dir, f)
                out_label = os.path.join(out_lbl_dir, f.replace("_0000.nii.gz", ".nii.gz"))

                if not os.path.exists(label_path):
                    print(f" Missing label: {label_path}")
                    continue

                if os.path.exists(out_image) and os.path.exists(out_label):
                    print(f" Skipping (already processed): {f}")
                    continue

                data.append({
                    "image": image_path,
                    "label": label_path,
                    "out_image": out_image,
                    "out_label": out_label
                })

    if len(data) == 0:
        print(f" No data to process in {dataset_name}. Skipping...")
        continue

    dataset = Dataset(data=data, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    for batch in tqdm(dataloader, total=len(dataloader)):
        try:
            safe_save_nifti(batch["image"], batch.get("image_meta_dict", {}), batch["out_image"][0])
            safe_save_nifti(batch["label"], batch.get("label_meta_dict", {}), batch["out_label"][0])
        except Exception as e:
            print(f" Failed batch → {str(e)}")
            continue
