from pathlib import Path

import pytest
import nibabel as nib
import numpy as np
import torch

from sparse_view_dataset.cone_beam import ConeBeamParams
from sparse_view_dataset.projection import centerize_affine
from sparse_view_dataset.io import save_nii

# In this case, the input NIfTI file has a rsampled spacing of (0.7, 0.7, 0.7) mm
# This data should be produced by run `python main.py crop ./ori_data/asoca/coronary/ data/asoca_size128_spacing0-7 --target-spacing 0.7 --target-shape 128 128 128`
@pytest.fixture
def spacing() -> tuple[float, float, float]:
    return 0.7, 0.7, 0.7


@pytest.fixture
def input_nii_path() -> Path:
    # if you have produced the cropped data using the command above, you can test with it.
    # return Path("data/asoca_size128_spacing0-7/Diseased_1/Diseased_1_lca.nii.gz")
    
    # default to use the provided test data
    return Path(__file__).parent / "test_data" / "asoca_size128_spacing0-7" / "Diseased_1_lca.nii.gz"


@pytest.fixture
def output_dir() -> Path:
    return Path("output/test_cone_beam_projs")


def save_pngs(
    out_dir: str | Path,
    base_name: str,
    branch_type: str,
    data: np.ndarray,
) -> list[Path]:
    from PIL import Image
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_paths = []
    for i, slice_data in enumerate(data):
        # ODL projections are in (H, W) format, but we want to save as (W, H) for visualization
        slice_data = np.rot90(slice_data)
        slice_data_normalized = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8) * 255
        img = Image.fromarray(slice_data_normalized.astype(np.uint8))
        p_png = out_dir / f"{base_name}_{branch_type}_slice_{i:03d}.png"
        img.save(p_png)
        png_paths.append(p_png)
    return png_paths


def test_cone_beam_projs(
    input_nii_path: Path,
    output_dir: Path
) -> None:
    img = nib.loadsave.load(input_nii_path)
    assert isinstance(img, nib.nifti1.Nifti1Image), "Loaded image is not a NIfTI image."
    assert img.affine is not None, "Affine matrix is missing in the input NIfTI file."
    
    data = img.get_fdata()  # (W, H, D)
    assert data.ndim == 3, "Input image is not 3D."
    
    data_tensor = torch.from_numpy(data).float()[None] # (1, W, H, D)
    
    affine_centered = centerize_affine(img.affine, np.array(img.shape))
    
    cone_beam_params = ConeBeamParams.init_from(
        volume_size=img.shape,
        affine=affine_centered,
        num_proj=10,
        start_angle=0,
        proj_size=(128, 128),
        proj_range=np.pi,
        dde=400,
        dso=1400,
    )
    
    projector = cone_beam_params.get_projection()
    
    projs = projector(data_tensor).squeeze()
    assert projs.shape == (10, 128, 128), "Projection shape is not correct."
    
    output_dir = output_dir / "projections_from_nii"
    save_nii(output_dir, "test_cone_beam_projs", "projs", projs.numpy(), np.eye(4))
    save_pngs(output_dir, "test_cone_beam_projs", "projs", projs.numpy())


def test_cone_beam_from_spacing_data(
    spacing: tuple[float, float, float],
    input_nii_path: Path,
    output_dir: Path
) -> None:
    img = nib.loadsave.load(input_nii_path)
    assert isinstance(img, nib.nifti1.Nifti1Image), "Loaded image is not a NIfTI image."
    
    data = img.get_fdata()  # (W, H, D)
    assert data.ndim == 3, "Input image is not 3D."
    
    data_tensor = torch.from_numpy(data).float()[None] # (1, W, H, D)
    
    affine = np.diag(spacing + (1.0,))  # Create a simple affine with the given spacing and no translation
    
    affine_centered = centerize_affine(affine, np.array(img.shape))
    
    cone_beam_params = ConeBeamParams.init_from(
        volume_size=img.shape,
        affine=affine_centered,
        num_proj=10,
        start_angle=0,
        proj_size=(128, 128),
        proj_range=np.pi,
        dde=400,
        dso=1400,
    )
    
    projector = cone_beam_params.get_projection()
    
    projs = projector(data_tensor).squeeze()
    assert projs.shape == (10, 128, 128), "Projection shape is not correct."
    
    output_dir = output_dir / "projections_from_spacing"
    save_nii(output_dir, "test_cone_beam_projs", "projs", projs.numpy(), np.eye(4))
    save_pngs(output_dir, "test_cone_beam_projs", "projs", projs.numpy())
    

if __name__ == "__main__":
    pytest.main([__file__])
    