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


def centerize_affine(affine: np.ndarray, voxels_shape: np.ndarray) -> np.ndarray:
    center = (voxels_shape - 1) / 2
    new_affine = affine.copy()
    new_affine[:3, 3] = -affine[:3, :3] @ center
    return new_affine


def centerize_ori_affine(
    ori_affine: np.ndarray,
    resampled_shape: tuple[int, ...],
    resampled_affine: np.ndarray,
) -> np.ndarray:
    r"""
    计算中心化后的 Affine 矩阵。
    该矩阵将原始图像的体素映射到一个以“重采样图像中心”为原点 (0,0,0) 的新空间。

    ### 简化数学推导:
    
    1.  **确定目标中心在世界坐标系的位置 ($c_r^W$):**
        设重采样图像中心在体素空间为 $c_r^{V_r} = (s_r - 1) / 2$。
        其在世界坐标系下的位置为：
        $$c_r^W = A_{V_r,W} \cdot c_r^{V_r} + T_r^W$$

    2.  **定义新坐标系 $P$ (Projected/Centralized Space):**
        我们希望新坐标系 $P$ 的原点就在 $c_r^W$ 处。
        对于任何世界坐标点 $x^W$，它在新坐标系下的表示为：
        $$x^P = x^W - c_r^W$$

    3.  **构造新的 Affine 矩阵:**
        原始映射关系为 $x^W = A_{V_{ori},W} \cdot x^{V_{ori}} + T_{ori}^W$。
        代入上式得到：
        $$x^P = (A_{V_{ori},W} \cdot x^{V_{ori}} + T_{ori}^W) - c_r^W$$
        $$x^P = A_{V_{ori},W} \cdot x^{V_{ori}} + (T_{ori}^W - c_r^W)$$
        
        因此，新矩阵的平移项为：
        $$T_{ori}^P = T_{ori}^W - c_r^W$$

    Args:
        ori_affine: 原始 4x4 Affine 矩阵 (Voxel to World)。
        resampled_shape: 重采样后的图像尺寸 (w, h, d)。
        resampled_affine: 重采样图像对应的 4x4 Affine 矩阵。

    Returns:
        np.ndarray: 中心化后的 4x4 Affine 矩阵。
    """
    resampled_shape_np = np.array(resampled_shape)
    assert resampled_shape_np.shape == (3,)
    c_r_Vr = (resampled_shape_np - 1) / 2.0

    A_Vr_W = resampled_affine[:3, :3]
    T_r_W = resampled_affine[:3, 3]
    c_r_W = A_Vr_W @ c_r_Vr + T_r_W

    A_Vori_W = ori_affine[:3, :3]
    T_ori_W = ori_affine[:3, 3]
    T_ori_P = T_ori_W - c_r_W

    affine_centralized = np.eye(4)
    affine_centralized[:3, :3] = A_Vori_W
    affine_centralized[:3, 3] = T_ori_P
    return affine_centralized
