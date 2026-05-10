from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import typer
from scipy import ndimage as ndi

from .affine_transforms import make_affine_spacing_positive
from .io import load_nifti, save_nii, save_pt


def separate_coronary(coronary: np.ndarray) -> dict[Literal["lca", "rca"], np.ndarray]:
    coronary = coronary.squeeze()
    assert coronary.ndim == 3, "Coronary np.ndarray after squeeze must be in shape (D, H, W)"

    outs = ndi.label(coronary.astype(np.int8))
    assert isinstance(outs, tuple) and len(outs) == 2
    labeled_array: np.ndarray = outs[0]
    num_features: int = outs[1]

    if num_features <= 1:
        raise ValueError("Coronary segmentation must have at least 2 components")

    component_sizes = np.bincount(labeled_array.ravel())[1:]
    largest_indices = np.argsort(component_sizes)[-2:][::-1] + 1

    region_0 = (labeled_array == largest_indices[0]).astype(np.bool_)
    region_1 = (labeled_array == largest_indices[1]).astype(np.bool_)

    center_0 = ndi.center_of_mass(region_0)
    center_1 = ndi.center_of_mass(region_1)

    if center_0[0] > center_1[0]:
        lca, rca = region_0, region_1
    else:
        lca, rca = region_1, region_0

    return {"lca": lca, "rca": rca}


def crop_expanded_roi(label: np.ndarray, affine: np.ndarray, iterations: int = 5) -> tuple[np.ndarray, np.ndarray]:
    coords = np.where(label)
    if coords[0].size == 0:
        return label.copy(), affine.copy()

    mins = [int(np.min(c)) for c in coords]
    maxs = [int(np.max(c)) + 1 for c in coords]
    shape = label.shape

    mins_exp = [max(0, mins[i] - iterations) for i in range(3)]
    maxs_exp = [min(shape[i], maxs[i] + iterations) for i in range(3)]

    slices = tuple(slice(mins_exp[i], maxs_exp[i]) for i in range(3))
    cropped = label[slices].copy()

    T = np.eye(4, dtype=float)
    T[:3, 3] = np.array((mins_exp[0], mins_exp[1], mins_exp[2]), dtype=float)
    return cropped, affine @ T


def resample_to_shape(
    label: np.ndarray,
    orig_affine: np.ndarray,
    target_shape: tuple[int, int, int],
    order: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    orig_shape = np.array(label.shape, dtype=float)
    target_shape_ = np.array(target_shape, dtype=float)

    zoom_factors = target_shape_ / orig_shape
    resampled = ndi.zoom(label.astype(np.uint8), zoom_factors.tolist(), order=order)
    resampled = (resampled > 0.5).astype(np.uint8)  # type: ignore

    scale_diag = np.ones(4, dtype=float)
    scale_diag[:3] = orig_shape / target_shape_
    new_affine = orig_affine @ np.diag(scale_diag)
    return resampled, new_affine


def resample_to_shape_and_spacing(
    data: np.ndarray,
    affine: np.ndarray,
    target_shape: tuple[int, int, int],
    target_spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    orig_spacing = np.abs(np.diag(affine))[:3]
    scale_factors = orig_spacing / target_spacing

    resampled = ndi.zoom(data, scale_factors, order=0)
    
    pad_before = [max(0, int(target_shape[i] - resampled.shape[i]) // 2) for i in range(3)]     # type: ignore
    pad_after = [max(0, int(target_shape[i] - resampled.shape[i] - pad_before[i])) for i in range(3)]   # type: ignore
    pad_width_tuple = tuple((pad_before[i], pad_after[i]) for i in range(3))
    resampled = np.pad(np.asarray(resampled), pad_width_tuple, mode="constant", constant_values=0)

    scale_diag = np.ones(4, dtype=float)
    scale_diag[:3] = 1.0 / scale_factors
    new_affine = affine @ np.diag(scale_diag)
    pad_before_vec = np.array([pad_before[0], pad_before[1], pad_before[2]], dtype=float)
    new_affine[:3, 3] = new_affine[:3, 3] - (new_affine[:3, :3] @ pad_before_vec)
    return resampled, new_affine


def process_single_input(
    input_file: Path,
    outdir: Path,
    expand: int,
    target_shape: tuple[int, int, int],
    target_spacing: float | None,
    saving_pt: bool = False,
) -> None:
    assert input_file.exists(), f"Input file not found: {input_file}"

    ori_data, ori_affine = load_nifti(input_file)
    branches = separate_coronary(ori_data)

    for branch_type, branch_label in branches.items():
        label, affine = crop_expanded_roi(branch_label, ori_affine, iterations=expand)
        label, affine = make_affine_spacing_positive(label, affine)

        if target_spacing is None:
            label_resampled, affine_resampled = resample_to_shape(label, affine, target_shape)
        else:
            label_resampled, affine_resampled = resample_to_shape_and_spacing(
                label, affine, target_shape, target_spacing
            )

        base_name = input_file.stem.split(".")[0]
        save_nii(outdir, base_name, branch_type, label_resampled.astype(np.uint8), affine_resampled)
        if saving_pt:
            save_pt(outdir, base_name, branch_type, label_resampled.astype(np.uint8), affine_resampled)


def process_input_path(
    input_path: Path,
    outdir: Path,
    expand: int,
    target_shape: tuple[int, int, int],
    target_spacing: float | None,
    workers: int | None = None,
    saving_pt: bool = False,
) -> None:
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
        # 并行处理目录下的多个 nii.gz 文件，默认使用系统 CPU 数量
        from tqdm import tqdm
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import os

        all_paths = list(input_path.rglob("*.nii.gz"))
        if len(all_paths) == 0:
            return

        max_workers = workers or (os.cpu_count() or 1)

        # 预先创建子目录，避免并发时目录创建冲突
        sub_outdirs = []
        for p in all_paths:
            sub_outdir = outdir / p.parent.relative_to(input_path) / str(p.stem).split(".")[0]
            sub_outdir.mkdir(parents=True, exist_ok=True)
            sub_outdirs.append((p, sub_outdir))

        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for p, sub_outdir in sub_outdirs:
                futures.append(ex.submit(process_single_input, p, sub_outdir, expand, target_shape, target_spacing, saving_pt))

            for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                # as_completed yields futures as they complete; check exceptions
                pass
    else:
        process_single_input(input_path, outdir, expand, target_shape, target_spacing, saving_pt)
