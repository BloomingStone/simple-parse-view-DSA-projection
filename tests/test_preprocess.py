import numpy as np

from sparse_view_dataset.affine_transforms import make_affine_spacing_positive
from sparse_view_dataset.preprocess import crop_expanded_roi, resample_to_shape, separate_coronary


def test_separate_coronary_returns_two_branches():
    coronary = np.zeros((6, 6, 6), dtype=np.uint8)
    coronary[1:3, 1:3, 1:3] = 1
    coronary[4:6, 1:3, 1:3] = 1

    branches = separate_coronary(coronary)

    assert set(branches) == {"lca", "rca"}
    assert branches["lca"].shape == coronary.shape
    assert branches["rca"].shape == coronary.shape


def test_crop_expanded_roi_adjusts_affine_translation():
    label = np.zeros((10, 10, 10), dtype=np.uint8)
    label[3:5, 4:7, 2:6] = 1
    affine = np.eye(4)

    cropped, new_affine = crop_expanded_roi(label, affine, iterations=0)

    assert cropped.shape == (2, 3, 4)
    assert np.allclose(new_affine[:3, 3], [3, 4, 2])


def test_make_affine_spacing_positive_flips_data():
    data = np.zeros((3, 2, 2), dtype=np.uint8)
    data[0, 0, 0] = 1
    affine = np.eye(4)
    affine[0, 0] = -1

    flipped, new_affine = make_affine_spacing_positive(data, affine)

    assert flipped[2, 0, 0] == 1
    assert new_affine[0, 0] > 0


def test_resample_to_shape_keeps_target_shape():
    label = np.zeros((4, 4, 4), dtype=np.uint8)
    label[1:3, 1:3, 1:3] = 1
    affine = np.eye(4)

    resampled, new_affine = resample_to_shape(label, affine, (8, 8, 8))

    assert resampled.shape == (8, 8, 8)
    assert new_affine.shape == (4, 4)
