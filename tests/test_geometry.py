import numpy as np

from sparse_view_dataset.affine_transforms import apply_affine, recenter_affine


def test_apply_affine_single_point():
    affine = np.eye(4)
    affine[:3, 3] = [1, 2, 3]
    out = apply_affine(np.array([0.0, 0.0, 0.0]), affine)
    assert np.allclose(out, [1, 2, 3])


def test_recenter_affine_moves_new_center_to_origin():
    affine = np.eye(4)
    affine[:3, 3] = [10, 20, 30]
    recentered = recenter_affine(affine, np.array([5, 5, 5]))
    # new_center_world (5,5,5) should become (0,0,0) after recentering
    assert np.allclose(recentered[:3, 3], [5, 15, 25])


def test_recenter_affine_preserves_rotation():
    affine = np.eye(4)
    affine[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # 90° rotation
    affine[:3, 3] = [10, 20, 30]
    recentered = recenter_affine(affine, np.array([5, 5, 5]))
    # Rotation matrix should be unchanged
    assert np.allclose(recentered[:3, :3], affine[:3, :3])
    assert recentered.shape == (4, 4)
