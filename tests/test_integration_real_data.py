"""Integration test using real data from data/asoca_size128_spacing0-7 and ori_data/asoca.

Output is saved to tests/output/test_real_data_proj/.
"""

from pathlib import Path

import numpy as np
import pytest

from sparse_view_dataset.cone_beam import ConeBeamParams
from sparse_view_dataset.projection import (
    Hu_to_mu,
    centerize_affine,
    centerize_ori_affine,
    make_affine_spacing_positive,
)
from sparse_view_dataset.io import read_nii_data

TESTS_DIR = Path(__file__).parent
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
    base_data = TESTS_DIR.parent / "data" / "asoca_size128_spacing0-7"
    ori_data = TESTS_DIR.parent / "ori_data" / "asoca"

    # —— Resampled coronary ——
    cor_file = base_data / TEST_CASE / f"{TEST_CASE}_{BRANCH}.nii.gz"
    resampled_cor_data, resample_cor_affine = read_nii_data(cor_file)

    # —— Original volume ——
    vol_file = ori_data / "volume" / f"{TEST_CASE}.nii.gz"
    ori_vol_data, ori_affine = read_nii_data(vol_file)

    # —— Preprocess ——
    ori_vol_data, ori_affine_positive = make_affine_spacing_positive(ori_vol_data, ori_affine)
    ori_vol_data = Hu_to_mu(ori_vol_data)

    resample_cor_affine_centered = centerize_affine(
        resample_cor_affine, np.array(resampled_cor_data.shape)
    )
    ori_affine_centralized = centerize_ori_affine(
        ori_affine_positive, resampled_cor_data.shape, resample_cor_affine
    )

    return {
        "resampled_cor_data": resampled_cor_data,
        "resample_cor_affine_centered": resample_cor_affine_centered,
        "ori_vol_data": ori_vol_data,
        "ori_affine_centralized": ori_affine_centralized,
    }


@pytest.fixture(scope="module")
def some_angles():
    """Small set of (alpha, beta) in degrees, auto-converted to radians."""
    alphas_deg = np.array([-90, -45, 0, 45, 90])
    betas_deg = np.array([-60, -30, 0, 30, 60])
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
        assert params.num_proj == 5
        proj = params.get_projection()
        assert proj.alphas_sorted is not None
        assert proj.betas_sorted is not None
        assert len(proj.alphas_sorted) == 5

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

        assert projs.shape == (5, 64, 64), f"Unexpected shape: {projs.shape}"
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

        assert label_projs.shape == (5, 64, 64)
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

        assert xca_vis.shape == (5, 64, 64)
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

        assert silhouette.shape == (5, 64, 64)
        assert depth.shape == (5, 64, 64)

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

    def test_visualize_results(self, test_data, some_angles, test_output_dir):
        """Generate visualization GIFs from projection and rendering results."""
        import torch

        from sparse_view_dataset.constants import MU_IDODINE
        from sparse_view_dataset.mesh_utils import get_mesh_in_world, get_label_clouds_in_world
        from sparse_view_dataset.torch3d_render import Torch3DLabelRenderer
        from sparse_view_dataset.visualize import plot_cloud_and_projs, save_gif

        alphas, betas = some_angles

        # ---- CT projection ----
        params_ct = ConeBeamParams.init_from_angles(
            volume_size=test_data["ori_vol_data"].shape,
            affine=test_data["ori_affine_centralized"],
            alphas=alphas, betas=betas,
            proj_size=(64, 64),
        )
        ct_proj = params_ct.get_projection()
        ct_tensor = torch.from_numpy(test_data["ori_vol_data"].copy())[None].float()
        ori_projs = ct_proj(ct_tensor).squeeze()

        # ---- Label projection ----
        params_label = ConeBeamParams.init_from_angles(
            volume_size=test_data["resampled_cor_data"].shape,
            affine=test_data["resample_cor_affine_centered"],
            alphas=alphas, betas=betas,
            proj_size=(64, 64),
        )
        label_proj = params_label.get_projection()
        label_tensor = torch.from_numpy(test_data["resampled_cor_data"].copy())[None].float()
        label_projs = label_proj(label_tensor).squeeze()

        xca_raw = torch.exp(-(ori_projs + label_projs * MU_IDODINE))
        xca_vis = torch.pow(xca_raw, 0.1)

        # ---- Render ----
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_tensor = torch.from_numpy(test_data["resampled_cor_data"].copy()).to(device)
        mesh = get_mesh_in_world(data_tensor, affine=test_data["resample_cor_affine_centered"])
        point_clouds = {
            "bg_mask": get_label_clouds_in_world(
                data_tensor, affine=test_data["resample_cor_affine_centered"]
            ),
        }
        if mesh.n_points == 0 or mesh.n_cells == 0:
            pytest.skip("Empty mesh for this case")

        renderer = Torch3DLabelRenderer(params_label.get_projection(), device)
        silhouette, depth, res_clouds = renderer.render(mesh, point_clouds)

        # ---- Save visualization GIFs ----
        vis_dir = test_output_dir / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # DRR projections GIF
        save_gif(vis_dir / "ori_projs.gif", ori_projs.transpose(-1, -2), origin="lower", cmap="gray")
        save_gif(vis_dir / "label_projs.gif", label_projs.transpose(-1, -2), origin="lower", cmap="gray")
        save_gif(vis_dir / "projs.gif", xca_vis.transpose(-1, -2), origin="lower", cmap="gray")

        # Depth and mask GIFs
        save_gif(vis_dir / "depth.gif", depth.transpose(-1, -2), origin="lower", cmap="viridis")
        save_gif(vis_dir / "mask_2d.gif", silhouette.transpose(-1, -2), origin="lower", cmap="gray")

        # Point cloud overlay GIF
        plot_cloud_and_projs(
            vis_dir / "bg_mask_and_projs.gif",
            res_clouds["bg_mask"],
            ori_projs,
        )
