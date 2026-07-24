from typing import TypeVar

import numpy as np

from torch import Tensor
from torch.nn import functional as F

ArrayLike = TypeVar("ArrayLike", bound=Tensor | np.ndarray)


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