import numpy as np

from sparse_view_dataset.affine_transforms import centerize_affine, centerize_ori_affine, apply_affine


def test_apply_affine_single_point():
    affine = np.eye(4)
    affine[:3, 3] = [1, 2, 3]
    out = apply_affine(np.array([0.0, 0.0, 0.0]), affine)
    assert np.allclose(out, [1, 2, 3])


def test_centerize_affine_moves_center_to_origin():
    affine = np.eye(4)
    centered = centerize_affine(affine, np.array([5, 5, 5]))
    assert np.allclose(centered[:3, 3], [-2, -2, -2])


def test_centerize_ori_affine_returns_expected_shape():
    ori_affine = np.eye(4)
    resampled_affine = np.eye(4)
    out = centerize_ori_affine(ori_affine, (5, 5, 5), resampled_affine)
    assert out.shape == (4, 4)
