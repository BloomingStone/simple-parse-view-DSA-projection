"""Tests for (alpha, beta) angle mode in cone beam projection and rendering."""

import numpy as np
import pytest
import torch

from sparse_view_dataset.cone_beam import ConeBeamParams


def R_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def R_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


@pytest.fixture
def volume_params():
    return {
        "volume_size": (32, 32, 32),
        "affine": np.array([
            [1, 0, 0, -16],
            [0, 1, 0, -16],
            [0, 0, 1, -16],
            [0, 0, 0,  1],
        ], dtype=float),
        "proj_size": (32, 32),
        "dde": 200.0,
        "dso": 400.0,
    }


@pytest.fixture
def sample_angles():
    return {
        "alphas": np.deg2rad(np.array([-28.6, -17.2, -5.7, 0.0, 5.7, 17.2, 28.6])),
        "betas": np.deg2rad(np.array([0.0, 2.9, 5.7, 8.6, 5.7, 2.9, 0.0])),
    }


class TestConeBeamParamsInitFromAngles:
    def test_basic_creation(self, volume_params, sample_angles):
        params = ConeBeamParams.init_from_angles(
            **volume_params,
            alphas=sample_angles["alphas"],
            betas=sample_angles["betas"],
        )
        assert params.alphas is not None
        assert params.betas is not None
        assert params.num_proj == 7

    def test_creation_without_betas(self, volume_params, sample_angles):
        params = ConeBeamParams.init_from_angles(
            **volume_params,
            alphas=sample_angles["alphas"],
            betas=None,
        )
        assert params.betas is not None
        assert np.all(params.betas == 0.0)

    def test_to_dict_contains_angles(self, volume_params, sample_angles):
        params = ConeBeamParams.init_from_angles(
            **volume_params,
            alphas=sample_angles["alphas"],
            betas=sample_angles["betas"],
        )
        d = params.to_dict()
        assert "alphas" in d
        assert "betas" in d


class TestConeBeamGeometrySourcePosition:
    """Verify that ODL geometry + src_shift_func matches the Euler Z-X formula."""

    def test_source_position_matches_euler(self, volume_params, sample_angles):
        params = ConeBeamParams.init_from_angles(
            **volume_params,
            alphas=sample_angles["alphas"],
            betas=sample_angles["betas"],
        )
        proj = params.get_projection()
        geometry = proj.geometry
        angles = geometry.angles
        src_pos = geometry.src_position(angles)

        # Compute expected source positions via Euler Z-X
        assert proj.alphas_sorted is not None and proj.betas_sorted is not None
        expected = [
            R_z(-a) @ R_x(-b) @ np.array([0.0, -params.dso, 0.0])
            for a, b in zip(proj.alphas_sorted, proj.betas_sorted)
        ]
        expected = np.array(expected)

        assert np.allclose(src_pos, expected,
                           atol=1e-5), f"Max error: {np.abs(src_pos - expected).max():.2e}"

    def test_zero_beta_matches_odl_rotation(self, volume_params):
        """When beta=0, source position should match pure Z rotation."""
        alphas = np.deg2rad(np.array([-28.6, -17.2, 0.0, 17.2, 28.6]))
        betas = np.zeros(5)
        params = ConeBeamParams.init_from_angles(
            **volume_params, alphas=alphas, betas=betas,
        )
        geometry = params.build_conebeam_geometry()[1]
        src_pos = geometry.src_position(geometry.angles)
        expected = np.array([
            R_z(a) @ np.array([0.0, -params.dso, 0.0])
            for a in geometry.angles
        ])
        assert np.allclose(src_pos, expected, atol=1e-5)


class TestProjectionConeBeamAlphaBeta:
    def test_alphas_sorted_property(self, volume_params, sample_angles):
        params = ConeBeamParams.init_from_angles(
            **volume_params,
            alphas=sample_angles["alphas"],
            betas=sample_angles["betas"],
        )
        proj = params.get_projection()
        assert proj.alphas_sorted is not None
        assert proj.betas_sorted is not None
        # ODL angles = -alphas_sorted (up to sorting)
        assert np.allclose(proj.geometry.angles, -proj.alphas_sorted)

    def test_forward_projection_shape(self, volume_params, sample_angles):
        params = ConeBeamParams.init_from_angles(
            **volume_params,
            alphas=sample_angles["alphas"],
            betas=sample_angles["betas"],
        )
        proj = params.get_projection()

        # Create simple volume with a sphere
        vol_data = np.zeros((32, 32, 32), dtype=np.float32)
        x, y, z = np.mgrid[-16:16:32j, -16:16:32j, -16:16:32j]
        vol_data[(x**2 + y**2 + z**2) < 64] = 1.0

        vol_tensor = torch.from_numpy(vol_data)[None].float()
        output = proj(vol_tensor)
        assert output.shape == (1, 7, 32, 32), f"Unexpected shape: {output.shape}"

    def test_to_dict(self, volume_params, sample_angles):
        params = ConeBeamParams.init_from_angles(
            **volume_params,
            alphas=sample_angles["alphas"],
            betas=sample_angles["betas"],
        )
        proj = params.get_projection()
        d = proj.to_dict()
        assert "alphas" in d
        assert "betas" in d
        assert "angles" in d
        assert len(d["alphas"]) == 7
        assert len(d["betas"]) == 7


class TestTorch3DLabelRendererAlphaBeta:
    @pytest.fixture
    def projection(self, volume_params, sample_angles):
        params = ConeBeamParams.init_from_angles(
            **volume_params,
            alphas=sample_angles["alphas"],
            betas=sample_angles["betas"],
        )
        return params.get_projection()

    def test_renderer_initialization(self, projection):
        import pyvista as pv  # noqa: F401

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from sparse_view_dataset.torch3d_render import Torch3DLabelRenderer

        renderer = Torch3DLabelRenderer(projection, device)
        assert renderer.width == 32
        assert renderer.height == 32
        assert renderer.fov > 0

    def test_render_cube(self, projection):
        import pyvista as pv
        from sparse_view_dataset.torch3d_render import Torch3DLabelRenderer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        renderer = Torch3DLabelRenderer(projection, device)

        # Simple cube mesh
        verts = torch.tensor([
            [-5, -5, -5], [5, -5, -5], [5, 5, -5], [-5, 5, -5],
            [-5, -5, 5], [5, -5, 5], [5, 5, 5], [-5, 5, 5],
        ], dtype=torch.float32, device=device)
        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
        ], dtype=torch.long, device=device)

        mesh_pv = pv.PolyData(
            verts.cpu().numpy(),
            np.hstack([np.full((len(faces), 1), 3), faces.cpu().numpy()])
        )
        point_clouds = {"test": verts}

        sil, depth, res_clouds = renderer.render(mesh_pv, point_clouds)
        assert sil.shape == (7, 32, 32)
        assert depth.shape == (7, 32, 32)
        assert "test" in res_clouds
        assert res_clouds["test"].shape == (7, 8, 3)

    def test_renderer_legacy_mode(self, volume_params):
        """Verify backward compatibility: no alphas → use ODL geometry path."""
        params = ConeBeamParams.init_from(
            **{k: v for k, v in volume_params.items() if k != "dde" and k != "dso"},
            num_proj=5,
            start_angle=0.0,
            proj_range=np.pi,
            dde=200.0,
            dso=400.0,
        )
        projection = params.get_projection()
        assert projection.alphas_sorted is None
        assert projection.betas_sorted is None

        import pyvista as pv
        from sparse_view_dataset.torch3d_render import Torch3DLabelRenderer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        renderer = Torch3DLabelRenderer(projection, device)

        verts = torch.tensor([
            [-5, -5, -5], [5, -5, -5], [5, 5, -5], [-5, 5, -5],
            [-5, -5, 5], [5, -5, 5], [5, 5, 5], [-5, 5, 5],
        ], dtype=torch.float32, device=device)
        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
        ], dtype=torch.long, device=device)

        mesh_pv = pv.PolyData(
            verts.cpu().numpy(),
            np.hstack([np.full((len(faces), 1), 3), faces.cpu().numpy()])
        )
        sil, depth, res_clouds = renderer.render(mesh_pv, {"test": verts})
        assert sil.shape == (5, 32, 32)
