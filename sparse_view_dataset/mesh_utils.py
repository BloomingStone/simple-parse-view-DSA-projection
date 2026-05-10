import numpy as np
import torch
from torch import Tensor
import pyvista as pv

from .affine_transforms import apply_affine

def get_mesh_in_voxel(label: Tensor) -> pv.PolyData:
    label_np = label.squeeze().cpu().numpy().astype(np.uint8)
    return pv.wrap(label_np).contour([1], method="marching_cubes").smooth_taubin().triangulate().clean()


def get_mesh_in_world(label: Tensor, affine: np.ndarray) -> pv.PolyData:
    mesh = get_mesh_in_voxel(label)
    mesh.points = apply_affine(mesh.points, affine)
    return mesh


def get_label_clouds_in_world(label: Tensor, affine: np.ndarray) -> Tensor:
    clouds = torch.stack(torch.where(label), dim=-1)
    return apply_affine(clouds, affine)