"""Integration test using real data from data/asoca_size128_spacing0-7 and ori_data/asoca.

Data files are located in tests/test_data/real_data_proj/:
  - Diseased_10_lca.nii.gz     (resampled coronary label)
  - Diseased_10.nii.gz         (original CT volume)

Output is saved to tests/output/test_real_data_proj/.
"""

from pathlib import Path

import numpy as np
import pytest

from sparse_view_dataset.cone_beam import ConeBeamParams
from sparse_view_dataset.projection import (
    Hu_to_mu,
    apply_affine,
    recenter_affine
)
from sparse_view_dataset.io import read_nii_data

TESTS_DIR = Path(__file__).parent
TEST_DATA_DIR = TESTS_DIR / "test_data" / "real_data_proj"
OUTPUT_DIR = TESTS_DIR / "output" / "test_real_data_proj"
TEST_CASE = "Diseased_10"
BRANCH = "lca"


@pytest.fixture(scope="module")
def test_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


@pytest.fixture(scope="module")
def test_data():
    """Load once, reuse across tests."""
    if not TEST_DATA_DIR.exists():
        pytest.skip(
            f"Test data not found at {TEST_DATA_DIR}. "
            f"Please download from Google Drive or run the crop pipeline first."
        )

    # —— Resampled coronary ——
    cor_file = TEST_DATA_DIR / f"{TEST_CASE}_{BRANCH}.nii.gz"
    resampled_cor_data, resample_cor_affine = read_nii_data(cor_file)

    # —— Original volume ——
    vol_file = TEST_DATA_DIR / f"{TEST_CASE}.nii.gz"
    ori_vol_data, ori_affine = read_nii_data(vol_file)

    # —— Preprocess ——
    ori_vol_data = Hu_to_mu(ori_vol_data)
    
    def get_new_world_center() -> np.ndarray:
        cor_shape = np.array(resampled_cor_data.shape)
        ori_shape = np.array(ori_vol_data.shape)
        cor_center_voxel = (cor_shape - 1) / 2
        ori_center_voxel = (ori_shape - 1) / 2
        cor_center_world = apply_affine(cor_center_voxel, resample_cor_affine)
        ori_center_world = apply_affine(ori_center_voxel, ori_affine)
        x_cor, y_cor, z_cor = cor_center_world
        x_ori, y_ori, z_ori = ori_center_world
        
        # 冠脉label在volume中的位置偏高，如果全使用冠脉中心，会导致DRR投影时图像上方超出volume范围，显示西欧爱过不佳
        return np.array((x_cor, y_cor, (z_ori + z_cor) / 2))

    new_world_center = get_new_world_center()

    return {
        "resampled_cor_data": resampled_cor_data,
        "resample_cor_affine_centered": recenter_affine(resample_cor_affine, new_world_center),
        "ori_vol_data": ori_vol_data,
        "ori_affine_centralized": recenter_affine(ori_affine, new_world_center),
    }


@pytest.fixture(scope="module")
def some_angles() -> tuple[np.ndarray, np.ndarray]:
    """Small set of (alpha, beta) in degrees, auto-converted to radians.
    Includes duplicate alphas to verify DiffDRR handles them correctly.
    """
    alphas_deg = np.array([-90, -60, -30, -15, 0,  0,  0,  0,  0,  0])
    betas_deg = np.array([ 0,   0,   0,   0,   0,  15, 30, 45, 60, 75])
    return np.deg2rad(alphas_deg), np.deg2rad(betas_deg)


class TestProjectionWithRealData:
    def test_construct_geometry(self, test_data, some_angles):
        """Verify that ConeBeamParams can be built from real data."""
        alphas, betas = some_angles
        params = ConeBeamParams.init_from_angles(
            volume_size=test_data["ori_vol_data"].shape,
            affine=test_data["ori_affine_centralized"],
            alphas=alphas,
            betas=betas,
            proj_size=(64, 64),
        )
        assert params.num_proj == 10
        proj = params.get_projection()
        assert len(proj.alphas) == 10
        assert len(proj.betas) == 10

    def test_forward_project(self, test_data, some_angles, test_output_dir):
        """Forward project real volume and verify output."""
        import torch

        alphas, betas = some_angles
        params = ConeBeamParams.init_from_angles(
            volume_size=test_data["ori_vol_data"].shape,
            affine=test_data["ori_affine_centralized"],
            alphas=alphas,
            betas=betas,
            proj_size=(64, 64),
        )
        projector = params.get_projection()

        vol_tensor = torch.from_numpy(test_data["ori_vol_data"].copy())[None].float()
        projs = projector(vol_tensor).squeeze()

        assert projs.shape == (10, 64, 64), f"Unexpected shape: {projs.shape}"
        assert projs.isfinite().all(), "Projection contains NaN/Inf"

        # Save for inspection
        out = {"projs": projs.numpy(), "meta": projector.to_dict()}
        torch.save(out, test_output_dir / "forward_project.pt")
        print(f"\n  Saved: {test_output_dir / 'forward_project.pt'}")
        print(f"  Proj range: [{projs.min():.4f}, {projs.max():.4f}]")

    def test_label_project(self, test_data, some_angles, test_output_dir):
        """Project coronary label volume."""
        import torch

        alphas, betas = some_angles
        params = ConeBeamParams.init_from_angles(
            volume_size=test_data["resampled_cor_data"].shape,
            affine=test_data["resample_cor_affine_centered"],
            alphas=alphas,
            betas=betas,
            proj_size=(64, 64),
        )
        projector = params.get_projection()

        data_tensor = torch.from_numpy(test_data["resampled_cor_data"].copy())[None].float()
        label_projs = projector(data_tensor).squeeze()

        assert label_projs.shape == (10, 64, 64)
        assert label_projs.isfinite().all()

        # Label projections should have higher values where coronary exists
        # (iodine has higher attenuation than water)
        assert label_projs.max() > 0, "Label projection should have positive values"

        torch.save(
            {"label_projs": label_projs.numpy()},
            test_output_dir / "label_project.pt",
        )
        print(f"\n  Saved: {test_output_dir / 'label_project.pt'}")

    def test_drr_generation(self, test_data, some_angles, test_output_dir):
        """Simulate DRR: combine CT + label projections with exponential attenuation."""
        import torch

        from sparse_view_dataset.constants import MU_IDODINE

        alphas, betas = some_angles

        # CT projection
        params_ct = ConeBeamParams.init_from_angles(
            volume_size=test_data["ori_vol_data"].shape,
            affine=test_data["ori_affine_centralized"],
            alphas=alphas,
            betas=betas,
            proj_size=(64, 64),
        )
        ct_proj = params_ct.get_projection()
        ct_tensor = torch.from_numpy(test_data["ori_vol_data"].copy())[None].float()
        ori_projs = ct_proj(ct_tensor).squeeze()

        # Label projection
        params_label = ConeBeamParams.init_from_angles(
            volume_size=test_data["resampled_cor_data"].shape,
            affine=test_data["resample_cor_affine_centered"],
            alphas=alphas,
            betas=betas,
            proj_size=(64, 64),
        )
        label_proj = params_label.get_projection()
        label_tensor = torch.from_numpy(test_data["resampled_cor_data"].copy())[None].float()
        label_projs = label_proj(label_tensor).squeeze()

        # DRR: exp(-(mu_water_path + iodine_path))
        xca_raw = torch.exp(-(ori_projs + label_projs * MU_IDODINE))
        xca_vis = torch.pow(xca_raw, 0.1)

        assert xca_vis.shape == (10, 64, 64)
        assert xca_vis.isfinite().all()
        assert (xca_vis > 0).all(), "DRR should be positive"

        torch.save(
            {
                "ori_projs": ori_projs.numpy(),
                "label_projs": label_projs.numpy(),
                "xca_raw": xca_raw.numpy(),
                "xca_vis": xca_vis.numpy(),
            },
            test_output_dir / "drr.pt",
        )
        print(f"\n  Saved: {test_output_dir / 'drr.pt'}")

    def test_render_mesh(self, test_data, some_angles, test_output_dir):
        """Render silhouette and depth from coronary mesh using real data."""
        import pyvista as pv
        import torch

        from sparse_view_dataset.mesh_utils import get_mesh_in_world, get_label_clouds_in_world
        from sparse_view_dataset.torch3d_render import Torch3DLabelRenderer

        alphas, betas = some_angles

        params = ConeBeamParams.init_from_angles(
            volume_size=test_data["resampled_cor_data"].shape,
            affine=test_data["resample_cor_affine_centered"],
            alphas=alphas,
            betas=betas,
            proj_size=(64, 64),
        )
        projector = params.get_projection()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build mesh & point clouds
        data_tensor = torch.from_numpy(test_data["resampled_cor_data"].copy()).to(device)
        mesh = get_mesh_in_world(data_tensor, affine=test_data["resample_cor_affine_centered"])
        point_clouds = {
            "bg_mask": get_label_clouds_in_world(
                data_tensor, affine=test_data["resample_cor_affine_centered"]
            ),
        }

        if mesh.n_points == 0 or mesh.n_cells == 0:
            pytest.skip("Empty mesh for this case - mesh extraction may need tuning")

        renderer = Torch3DLabelRenderer(projector, device)
        silhouette, depth, res_clouds = renderer.render(mesh, point_clouds)

        assert silhouette.shape == (10, 64, 64)
        assert depth.shape == (10, 64, 64)

        torch.save(
            {
                "silhouette": silhouette.cpu().numpy(),
                "depth": depth.cpu().numpy(),
                "bg_mask": res_clouds["bg_mask"].cpu().numpy(),
            },
            test_output_dir / "render.pt",
        )
        print(f"\n  Saved: {test_output_dir / 'render.pt'}")
        print(f"  Silhouette - pixels covered: {silhouette.sum().item():.0f} / {silhouette.numel()}")
        print(f"  Depth - min: {depth.min().item():.2f}, max: {depth.max().item():.2f}")

    def test_generate_visualization_gifs_real(self, some_angles, test_output_dir):
        """Generate visualization GIFs from projection and rendering results."""
        import torch

        from sparse_view_dataset.projection import _project_one_case_inner
        from sparse_view_dataset.projection_angles import ProjectionAngles

        alphas, betas = some_angles
        
        
        _project_one_case_inner(
            resampled_coronary_file=TEST_DATA_DIR / f"{TEST_CASE}_{BRANCH}.nii.gz",
            ori_volume_file=TEST_DATA_DIR / f"{TEST_CASE}.nii.gz",
            case_name=f"test_case_{TEST_CASE}_{BRANCH}",
            branch_type=BRANCH,
            proj_size=(64, 64),
            output_dir=test_output_dir,
            # some_angles 返回弧度，直接构造 ProjectionAngles（内部存储为弧度）
            angles=ProjectionAngles(np.column_stack((alphas, betas))),
            vis=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
