from typing import Literal, Annotated, Any, Tuple
from pathlib import Path

import typer
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

def separate_coronary(coronary: np.ndarray) -> dict[Literal["lca", "rca"], np.ndarray]:
    """
    separate coronary to LCA(1) and RCA(2)
    Args:
        coronary (np.ndarray): coronary segmentation
    Returns:
        lca (np.ndarray): LCA segmentation, dtype: torch.bool
        rca (np.ndarray): RCA segmentation, dtype: torch.bool
    """
    coronary = coronary.squeeze()
    assert coronary.ndim == 3, "Coronary np.ndarray after squeeze must be in shape (D, H, W)"

    outs = ndi.label(coronary.astype(np.int8))
    assert isinstance(outs, tuple) and len(outs) == 2
    labeled_array: np.ndarray = outs[0]
    num_features: int = outs[1]
    
    if num_features <= 1:
        raise ValueError("Coronary segmentation must have at least 2 components")
    
    assert labeled_array is not None
    component_sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background (0)
    
    largest_indices = np.argsort(component_sizes)[-2:][::-1] + 1
    
    region_0: np.ndarray = (labeled_array == largest_indices[0]).astype(np.bool_)
    region_1: np.ndarray = (labeled_array == largest_indices[1]).astype(np.bool_)
    
    center_0 = ndi.center_of_mass(region_0)
    center_1 = ndi.center_of_mass(region_1)
    
    if center_0[0] > center_1[0]:
        lca, rca = region_0, region_1
    else:
        lca, rca = region_1, region_0
    
    return {"lca": lca, "rca": rca}


def load_nifti(path: Path):
    img = nib.loadsave.load(str(path))
    assert isinstance(img, nib.nifti1.Nifti1Image), "Only Nifti1Image format is supported."
    data = img.get_fdata(dtype=np.float32)
    assert img.affine is not None, "Input nifti file has no affine."
    affine = img.affine.copy()
    return data, affine


def save_nii(
        out_dir: str|Path,
        base_name: str,
        branch_type: Literal["lca", "rca"],
        data: np.ndarray,
        affine: np.ndarray
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_nii = out_dir / f"{base_name}_{branch_type}.nii.gz"
    nib.loadsave.save(nib.nifti1.Nifti1Image(data, affine), p_nii)
    return p_nii

def save_pt(
    out_dir: str|Path,
    base_name: str,
    branch_type: Literal["lca", "rca"],
    data: np.ndarray,
    affine: np.ndarray
) -> Path:
    import torch
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_pt = out_dir / f"{base_name}_{branch_type}.pt"
    torch.save(
        {
            "volume": data,
            "affine": affine
        }, 
        p_pt
    )
    return p_pt


def crop_expanded_roi(label: np.ndarray, affine: np.ndarray, iterations: int=5) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute an axis-aligned bounding box (cuboid) around True values in `label`,
    expand the box by `iterations` voxels on each side (clamped to array bounds),
    and return the cropped sub-label together with the voxel offset of the crop.

    Args:
        label: boolean array, original data
        affine: 4x4 affine matrix
        iterations: number of voxels to expand the bounding box on each side
    
    Returns: 
        cropped_mask: boolean array, label cropped to the bounding box
        new_affine: 4x4 affine matrix, adjusted to account for cropping
    """
    # label: boolean array
    coords = np.where(label)
    if coords[0].size == 0:
        # no positive voxels: return full label and zero offset
        return label.copy(), affine.copy()

    mins = [int(np.min(c)) for c in coords]
    maxs = [int(np.max(c)) + 1 for c in coords]  # exclusive
    shape = label.shape

    mins_exp = [max(0, mins[i] - iterations) for i in range(3)]
    maxs_exp = [min(shape[i], maxs[i] + iterations) for i in range(3)]

    slices = tuple(slice(mins_exp[i], maxs_exp[i]) for i in range(3))
    cropped = label[slices].copy()
    offset = (mins_exp[0], mins_exp[1], mins_exp[2])
    
    # Adjust affine to account for cropping: new_affine_crop = orig_affine @ T_offset
    T = np.eye(4, dtype=float)
    # offset is in voxels (ox,oy,oz)
    T[:3, 3] = np.array(offset, dtype=float)
    affine_cropped = affine @ T
    
    return cropped, affine_cropped

def make_affine_spacing_positive(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform data and affine to make the spacing in affine positive.
    ATTENTIONE: after flip, the origin (0,0,0) will change, and therefore T in the affine will also change 
    accordingly to make sure the world coordinates of voxels remain unchanged.
    e.g. if flip along x axis, the voxel that was originally at (0,0,0) will now be at (W-1,0,0), therefore 
    the new affine should satisfy: A_new @ [W-1,0,0] + T_new = A_orig @ [0,0,0] + T_orig = T_orig
    T_new = T_orig - A_new @ [W-1,0,0] = T_orig - (-A_orig[:3,0]) * (W-1)
    (because A_new and A_orig only differ by the sign of the first column)
    
    Args:
        data: numpy array in voxel space
        affine: 4x4 affine matrix
        
    Returns:
        data: numpy array in voxel space
        affine: 4x4 affine matrix
    """
    spacing = np.diag(affine)[:3]
    data = data.copy()
    affine = affine.copy()
    shape = np.array(data.shape, dtype=int)
    A = affine[:3, :3]
    T = affine[:3, 3]
    if spacing[0] < 0:
        A[:, 0] = -A[:, 0]
        T = T - A[:, 0] * (shape[0] - 1)
        data = np.flip(data, axis=0)
    if spacing[1] < 0:
        A[:, 1] = -A[:, 1]
        T = T - A[:, 1] * (shape[1] - 1)
        data = np.flip(data, axis=1)
    if spacing[2] < 0:
        A[:, 2] = -A[:, 2]
        T = T - A[:, 2] * (shape[2] - 1)
        data = np.flip(data, axis=2)
    
    affine[:3, :3] = A
    affine[:3, 3] = T
        
    return data, affine


def resample_to_shape(
    label: np.ndarray, 
    orig_affine: np.ndarray, 
    target_shape: tuple[int, int, int], 
    order: int=0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        label: numpy array in voxel space with shape (w,h,d)
        orig_affine: 4x4 affine mapping original voxels -> world
        target_shape: tuple of ints (w,h,d)
        order: interpolation order (0 for nearest)
    Returns: 
        resampled_mask (shape target_shape), 
        new_affine (4x4)
    """
    orig_shape = np.array(label.shape, dtype=float)
    target_shape_ = np.array(target_shape, dtype=float)

    # compute zoom factors for scipy.ndimage.zoom: zoom = target / orig
    zoom_factors = target_shape_ / orig_shape
    resampled = ndi.zoom(label.astype(np.uint8), zoom_factors.tolist(), order=order)
    assert isinstance(resampled, np.ndarray)
    resampled=(resampled > 0.5).astype(np.uint8)

    # new_affine = orig_affine @ diag(orig/target)
    scale_diag = np.ones(4, dtype=float)
    scale_diag[:3] = orig_shape / target_shape_  # orig/target
    scale_mat = np.diag(scale_diag)
    new_affine = orig_affine @ scale_mat
    
    return resampled, new_affine


def resample_to_shape_and_spacing(
    data: np.ndarray, 
    affine: np.ndarray, 
    target_shape: tuple[int, int, int],
    target_spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample data to target spacing while preserving voxel shape.
    
    Args:
        data: numpy array in voxel space
        affine: 4x4 affine matrix
        target_spacing: target voxel spacing in mm
        
    Returns:
        resampled_data: numpy array with original shape but new spacing
        new_affine: adjusted affine matrix
    """
    orig_spacing = np.abs(np.diag(affine))[:3]
    scale_factors = orig_spacing / target_spacing
    
    # Resample data using zoom
    resampled = ndi.zoom(data, scale_factors, order=0)
    assert isinstance(resampled, np.ndarray)
    
    # Calculate padding needed to maintain original shape
    pad_before = [max(0, (target_shape[i] - resampled.shape[i]) // 2) for i in range(3)]
    pad_after = [max(0, target_shape[i] - resampled.shape[i] - pad_before[i]) for i in range(3)]
    pad_width_tuple = tuple((pad_before[i], pad_after[i]) for i in range(3))
    
    # Apply symmetric padding with explicit type casting
    resampled_array: Any = resampled
    if not isinstance(resampled_array, np.ndarray):
        resampled_array = np.array(resampled_array)
    resampled = np.pad(resampled_array, pad_width_tuple, mode='constant', constant_values=0)
    
    # Adjust affine matrix for the zoom (linear part)
    scale_diag = np.ones(4, dtype=float)
    scale_diag[:3] = 1.0 / scale_factors
    new_affine = affine @ np.diag(scale_diag)

    # Compensate for padding at the beginning of each axis: vox_index_padded = vox_index_original + pad_before
    # therefore world_coord = A_new @ (vox_idx_padded - pad_before) + T_new = (A_new @ vox_idx_padded) + (T_new - A_new @ pad_before)
    pad_before_vec = np.array([pad_before[0], pad_before[1], pad_before[2]], dtype=float)
    new_affine[:3, 3] = new_affine[:3, 3] - (new_affine[:3, :3] @ pad_before_vec)

    return resampled, new_affine

import numpy as np


def crop_roi_and_resample(
    input_file: Path,
    outdir: Path,
    expand: int,
    target_shape: tuple[int, int, int],
    target_spacing: float | None,
    saving_pt: bool = False
):
    assert input_file.exists(), f"Input file not found: {input_file}"

    ori_data, ori_affine = load_nifti(input_file)
    branches = separate_coronary(ori_data)
    
    for branch_type, branch_label in branches.items():
        print(f"{input_file.stem} - {branch_type}: {branch_label.shape}")

        # Compute axis-aligned cuboid ROI and expand bounds by given iterations
        label, affine = crop_expanded_roi(branch_label, ori_affine, iterations=expand)
        label, affine = make_affine_spacing_positive(label, affine)

        # Resample the cropped ROI to target shape (nearest to preserve binary)
        if target_spacing is None:
            label_resampled, affine_resampled = resample_to_shape(label, affine, target_shape)
        else:
            label_resampled, affine_resampled = resample_to_shape_and_spacing(label, affine, target_shape, target_spacing)

        base_name = input_file.stem.split('.')[0]

        save_nii(outdir, base_name, branch_type, label_resampled.astype(np.uint8), affine_resampled)
        if saving_pt:
            save_pt(outdir, base_name, branch_type, label_resampled.astype(np.uint8), affine_resampled)



def main(
    input_path: Annotated[Path, typer.Argument(help="Input file path or data directory")],
    outdir: Annotated[Path, typer.Argument(help="Output directory")],
    expand: Annotated[int, typer.Option(help="Expand ROI by this many voxels on each side")] = 2,
    target_shape: Annotated[tuple[int, int, int], typer.Option(help="Target shape (w,h,d)")] = (256, 256, 256),
    target_spacing: Annotated[float|None, typer.Option(help="Target spacing (mm) - used for spacing adjustment if needed")] = None,
    saving_pt: Annotated[bool, typer.Option(help="Save pt files")] = False,
):
    """
    Crop and resample coronary artery data by:
    1. Loading NIfTI file or files from input path, If the input is a directory, all nii files will be iteratively searched
    2. Separating LCA and RCA coronary branches
    3. Cropping and expanding ROIs for each branch
    4. flip data and adjust affine to make spacing positive
    5. Resampling to target shape assigned by `target_shape` (if `target_spacing` set, also resample to it by zoom and necessary padding, therefore the shape may be larger than `target_shape`)
    6. Saving results NIfTI files. output path as `outdir/<input_nii_path_relative_to_input_path>/<input_nii_name>_<branch_type>.nii.gz`
    
    Args:
        input_path: Path to input NIfTI file or directory containing NIfTI files
        outdir: Output directory for processed files
        expand: Number of voxels to expand ROI on each side (default: 5)
        target_shape: Target output shape in (d,w,h) format (default: (256, 256, 256))
        target_spacing: Target output spacing in mm (default: None, recommended: 0.5)
    """

    if input_path.is_dir():
        for p in input_path.rglob("*.nii.gz"):
            sub_outdir = outdir / p.parent.relative_to(input_path) / str(p.stem).split('.')[0]
            crop_roi_and_resample(p, sub_outdir, expand, target_shape, target_spacing, saving_pt)
    else:
        crop_roi_and_resample(input_path, outdir, expand, target_shape, target_spacing, saving_pt)


if __name__ == "__main__":
    typer.run(main)