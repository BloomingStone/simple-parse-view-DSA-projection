from typing import TypeVar

import numpy as np

from torch import Tensor
from torch.nn import functional as F

ArrayLike = TypeVar("ArrayLike", bound=Tensor | np.ndarray)


def make_affine_spacing_positive(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

def apply_affine(points: ArrayLike, affine: np.ndarray) -> ArrayLike:
    import torch
    from torch.nn import functional as F_local

    is_numpy = isinstance(points, np.ndarray)
    is_single_point = points.ndim == 1

    if is_single_point:
        assert points.shape == (3,)
    else:
        assert points.ndim == 2 and points.shape[-1] == 3

    if is_numpy:
        pts = torch.from_numpy(points)
    else:
        pts = points

    aff = torch.from_numpy(affine)
    pts = pts.to(dtype=torch.float32)
    aff = aff.to(device=pts.device, dtype=pts.dtype)

    if is_single_point:
        pts = pts.unsqueeze(0)

    pts_h = F_local.pad(pts, (0, 1), value=1)
    out = pts_h @ aff.T
    out = out[:, :3]

    if is_single_point:
        out = out.squeeze(0)

    if is_numpy:
        return out.cpu().numpy()  # type: ignore[return-value]
    return out  # type: ignore[return-value]

def recenter_affine(affine: np.ndarray, new_center_world: np.ndarray) -> np.ndarray:
    """Re-center the affine so that the new center in world coordinates is at (0,0,0)."""
    new_affine = affine.copy()
    new_affine[:3, 3] -= new_center_world   # Shift the translation part to recenter
    return new_affine