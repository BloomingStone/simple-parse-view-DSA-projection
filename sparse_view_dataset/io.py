from pathlib import Path

import nibabel as nib
import numpy as np


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.loadsave.load(str(path))
    assert isinstance(img, nib.nifti1.Nifti1Image), "Only Nifti1Image format is supported."
    data = img.get_fdata(dtype=np.float32)
    assert img.affine is not None, "Input nifti file has no affine."
    return data, img.affine.copy()


def read_nii_data(nii_file: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.loadsave.load(nii_file)
    assert isinstance(img, nib.nifti1.Nifti1Image)
    data = img.get_fdata()
    affine = img.affine
    assert affine is not None
    return data, affine


def save_nii(out_dir: str | Path, base_name: str, branch_type: str, data: np.ndarray, affine: np.ndarray) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_nii = out_dir / f"{base_name}_{branch_type}.nii.gz"
    nib.loadsave.save(nib.nifti1.Nifti1Image(data, affine), p_nii)
    return p_nii


def save_pt(out_dir: str | Path, base_name: str, branch_type: str, data: np.ndarray, affine: np.ndarray) -> Path:
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_pt = out_dir / f"{base_name}_{branch_type}.pt"
    torch.save({"volume": data, "affine": affine}, p_pt)
    return p_pt
