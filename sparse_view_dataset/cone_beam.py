from dataclasses import dataclass

import numpy as np
import odl
import torch.nn as nn
from torch import Tensor
from odl.contrib import torch as odl_torch

@dataclass
class ConeBeamParams:
    affine: np.ndarray
    nVoxels: np.ndarray
    sVoxels: np.ndarray
    min_pt_world: np.ndarray
    max_pt_world: np.ndarray
    nh: int
    nw: int
    sh: float
    sw: float
    dde: float
    dso: float
    num_proj: int
    start_angle: float
    end_angle: float
    proj_range: float

    @staticmethod
    def init_from(
        volume_size: tuple[int, ...],
        affine: np.ndarray,
        num_proj: int,
        start_angle: float,
        proj_size: tuple[int, int],
        proj_range: float = np.pi,
        dde: float = 400,
        dso: float = 1400,
    ) -> "ConeBeamParams":

        assert len(volume_size) == 3
        nVoxels = np.array(volume_size, dtype=int)
        A = affine[:3, :3].copy()
        T = affine[:3, 3].copy()
        spacing = np.linalg.norm(A, axis=0)
        sVoxel = spacing * nVoxels

        origin_world = T
        shape_world = A @ nVoxels + T
        min_pt_world = np.minimum(origin_world, shape_world)
        max_pt_world = np.maximum(origin_world, shape_world)

        end_angle = start_angle + proj_range
        dh = 512 * 0.3 / proj_size[0]
        dw = 512 * 0.3 / proj_size[1]
        nh = proj_size[0]
        nw = proj_size[1]
        sh = nh * dh
        sw = nw * dw

        params = ConeBeamParams(
            affine=affine,
            nVoxels=nVoxels,
            sVoxels=sVoxel,
            min_pt_world=min_pt_world,
            max_pt_world=max_pt_world,
            nh=nh,
            nw=nw,
            sh=sh,
            sw=sw,
            dde=dde,
            dso=dso,
            num_proj=num_proj,
            start_angle=start_angle,
            end_angle=end_angle,
            proj_range=proj_range,
        )
        return params


    def build_conebeam_geometry(self) -> tuple[odl.DiscretizedSpace, odl.tomo.ConeBeamGeometry, odl.tomo.RayTransform, odl.Operator]:
        reco_space = odl.uniform_discr(
            min_pt=[float(self.min_pt_world[0]), float(self.min_pt_world[1]), float(self.min_pt_world[2])],
            max_pt=[float(self.max_pt_world[0]), float(self.max_pt_world[1]), float(self.max_pt_world[2])],
            shape=[int(self.nVoxels[0]), int(self.nVoxels[1]), int(self.nVoxels[2])],
            dtype="float32",
        )

        angle_partition = odl.uniform_partition(min_pt=self.start_angle, max_pt=self.end_angle, shape=self.num_proj)
        detector_partition = odl.uniform_partition(
            min_pt=[-(self.sh / 2.0), -(self.sw / 2.0)],
            max_pt=[(self.sh / 2.0), (self.sw / 2.0)],
            shape=[self.nh, self.nw],
        )
        geometry = odl.tomo.ConeBeamGeometry(
            apart=angle_partition,
            dpart=detector_partition,
            src_radius=self.dso,
            det_radius=self.dde,
            axis=[0, 0, 1],
        )
        ray_trafo = odl.tomo.RayTransform(vol_space=reco_space, geometry=geometry, impl="astra_cuda")
        fbp_op = odl.tomo.fbp_op(ray_trafo=ray_trafo, filter_type="Ram-Lak", frequency_scaling=1.0)
        return reco_space, geometry, ray_trafo, fbp_op
    
    def get_projection(self) -> "ProjectionConeBeam":
        return ProjectionConeBeam(self)


class ProjectionConeBeam(nn.Module):
    def __init__(self, param: ConeBeamParams):
        super().__init__()
        self.param = param
        self.reco_space, self.geometry, self.ray_trafo, self.FBPOper = self.param.build_conebeam_geometry()
        self.trafo = odl_torch.OperatorModule(self.ray_trafo)

    def forward(self, x: Tensor) -> Tensor:
        return self.trafo(x)
