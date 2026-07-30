"""Tests for angle configuration loading and parsing."""

import numpy as np
import pytest

from sparse_view_dataset.projection_angles import ProjectionAngles


class TestProjectionAngles:
    def test_from_alpha_beta_list(self):
        angles = ProjectionAngles.from_alpha_beta_list([(0.0, 0.0), (10.0, 5.0), (20.0, 10.0)])
        assert angles.num_angles == 3
        assert np.allclose(angles.alphas, np.deg2rad([0, 10, 20]))
        assert np.allclose(angles.betas, np.deg2rad([0, 5, 10]))

    def test_from_alpha_beta_arrays(self):
        alphas = np.array([0.0, 30.0, 60.0])
        betas = np.array([0.0, 10.0, 20.0])
        angles = ProjectionAngles.from_alpha_beta_arrays(alphas, betas)
        assert angles.num_angles == 3
        assert np.allclose(angles.alphas, np.deg2rad([0, 30, 60]))
        assert np.allclose(angles.betas, np.deg2rad([0, 10, 20]))

    def test_from_random(self):
        angles = ProjectionAngles.from_random(16, seed=42)
        assert angles.num_angles == 16
        # alpha range should be [-60, 60] degrees, beta range [-30, 30]
        assert angles.alphas.min() >= np.deg2rad(-60)
        assert angles.alphas.max() <= np.deg2rad(60)
        assert angles.betas.min() >= np.deg2rad(-30)
        assert angles.betas.max() <= np.deg2rad(30)

    def test_meta(self):
        alphas = np.array([0.0, 10.0])
        betas = np.array([0.0, 5.0])
        angles = ProjectionAngles.from_alpha_beta_arrays(alphas, betas)
        meta = angles.meta
        assert meta["num_angles"] == 2
        assert meta["alphas"] == [0.0, np.deg2rad(10.0)]
        assert meta["betas"] == [0.0, np.deg2rad(5.0)]

    def test_iteration(self):
        angles = ProjectionAngles.from_alpha_beta_list([(0.0, 0.0), (10.0, 5.0)])
        pairs = list(angles)
        assert len(pairs) == 2
        assert np.allclose(pairs[0], np.deg2rad([0, 0]))
        assert np.allclose(pairs[1], np.deg2rad([10, 5]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="must have the same shape"):
            ProjectionAngles.from_alpha_beta_arrays(
                np.array([0.0, 10.0]), np.array([0.0])
            )

    def test_invalid_list_raises(self):
        with pytest.raises(ValueError, match="angles must be a list"):
            ProjectionAngles.from_alpha_beta_list([(0.0, 0.0, 1.0)])    # type: ignore
