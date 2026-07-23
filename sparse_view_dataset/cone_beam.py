from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import odl
import torch
import torch.nn as nn
from torch import Tensor
from odl.contrib import torch as odl_torch


def _make_src_shift_func(alphas: np.ndarray, betas: np.ndarray, dso: float):
    """创建 ODL src_shift_func，将 (alpha, beta) 映射到 ODL 偏移量。

    ODL 的 src_position(angle) 在 angle = -alpha 时的基线位置为
    R_z(-alpha) @ (0, -dso, 0)。beta 角引入的额外位移表现为：

        shift_d = -dso * (1 - cos(beta))  绕 detector-to-source 方向
        shift_t = 0                        绕切线方向
        shift_r = dso * sin(beta)          绕旋转轴方向

    验证：R_z(-alpha) @ R_x(-beta) @ (0, -dso, 0) 与
          src_position_odl(-alpha) + shift 的结果一致。
    """
    neg_alphas = -np.asarray(alphas)
    betas_arr = np.asarray(betas)

    def shift_func(angle):
        angle = np.atleast_1d(np.asarray(angle, dtype=float))
        shifts = np.zeros((len(angle), 3), dtype=float)
        for i, a in enumerate(angle):
            idx = np.argmin(np.abs(neg_alphas - a))
            beta = betas_arr[idx]
            shifts[i, 0] = -dso * (1.0 - np.cos(beta))  # shift_d
            shifts[i, 2] = dso * np.sin(beta)            # shift_r
            # shift_t = 0 (默认)
        return shifts

    return shift_func


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
    alphas: Optional[np.ndarray] = None
    betas: Optional[np.ndarray] = None

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

    @staticmethod
    def init_from_angles(
        volume_size: tuple[int, ...],
        affine: np.ndarray,
        alphas: np.ndarray,
        betas: Optional[np.ndarray],
        proj_size: tuple[int, int],
        dde: float = 400,
        dso: float = 1400,
    ) -> "ConeBeamParams":
        """从 (alpha, beta) 角度对创建 ConeBeamParams。

        Parameters
        ----------
        volume_size : (D, H, W) 体素网格尺寸
        affine : 4x4 仿射矩阵
        alphas : (N,) RAO 角度（弧度），正值为从前向右
        betas : (N,) 或 None, CRA 角度（弧度），正值为从前向下
        proj_size : (H_det, W_det) 探测器像素数
        dde : detector 到 world origin 距离
        dso : source 到 world origin 距离
        """
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

        num_proj = len(alphas)
        dh = 512 * 0.3 / proj_size[0]
        dw = 512 * 0.3 / proj_size[1]
        nh = proj_size[0]
        nw = proj_size[1]
        sh = nh * dh
        sw = nw * dw

        if betas is None:
            betas = np.zeros(num_proj)

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
            start_angle=float(alphas[0]),
            end_angle=float(alphas[-1]),
            proj_range=float(alphas[-1] - alphas[0]),
            alphas=np.asarray(alphas),
            betas=np.asarray(betas),
        )
        return params

    def build_conebeam_geometry(self) -> tuple[odl.DiscretizedSpace, odl.tomo.ConeBeamGeometry, odl.tomo.RayTransform, odl.Operator]:
        reco_space = odl.uniform_discr(
            min_pt=[float(self.min_pt_world[0]), float(self.min_pt_world[1]), float(self.min_pt_world[2])],
            max_pt=[float(self.max_pt_world[0]), float(self.max_pt_world[1]), float(self.max_pt_world[2])],
            shape=[int(self.nVoxels[0]), int(self.nVoxels[1]), int(self.nVoxels[2])],
            dtype="float32",
        )

        if self.alphas is not None:
            # (alpha, beta) 模式：
            #   ODL angle = -alpha (RAO 为绕 Z 轴负向旋转)
            #   nonuniform_partition 需要单调递增的值
            odl_angles = -np.asarray(self.alphas, dtype=float)
            sort_idx = np.argsort(odl_angles)
            odl_angles_sorted = odl_angles[sort_idx]

            angle_partition = odl.nonuniform_partition(odl_angles_sorted)
            src_shift_func = _make_src_shift_func(self.alphas, self.betas, self.dso)
        else:
            # 原始模式：均匀角度
            angle_partition = odl.uniform_partition(min_pt=self.start_angle, max_pt=self.end_angle, shape=self.num_proj)
            src_shift_func = None

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
            src_shift_func=src_shift_func,
        )
        ray_trafo = odl.tomo.RayTransform(vol_space=reco_space, geometry=geometry, impl="astra_cuda")
        fbp_op = odl.tomo.fbp_op(ray_trafo=ray_trafo, filter_type="Ram-Lak", frequency_scaling=1.0)
        return reco_space, geometry, ray_trafo, fbp_op

    def get_projection(self) -> "ProjectionConeBeam":
        return ProjectionConeBeam(self)

    def to_dict(self) -> dict:
        d = {
            "affine": self.affine.tolist(),
            "nVoxels": self.nVoxels.tolist(),
            "sVoxels": self.sVoxels.tolist(),
            "min_pt_world": self.min_pt_world.tolist(),
            "max_pt_world": self.max_pt_world.tolist(),
            "nh": self.nh,
            "nw": self.nw,
            "sh": self.sh,
            "sw": self.sw,
            "dde": self.dde,
            "dso": self.dso,
            "num_proj": self.num_proj,
            "start_angle": self.start_angle,
            "end_angle": self.end_angle,
            "proj_range": self.proj_range,
        }
        if self.alphas is not None:
            d["alphas"] = self.alphas.tolist()
        if self.betas is not None:
            d["betas"] = self.betas.tolist()
        return d


class ProjectionConeBeam(nn.Module):
    def __init__(self, param: ConeBeamParams):
        super().__init__()
        self.param = param
        self.reco_space, self.geometry, self.ray_trafo, self.FBPOper = self.param.build_conebeam_geometry()

        # 存储与 ODL 几何角度顺序一致的 (alpha, beta)
        if param.alphas is not None:
            # (alpha, beta) 模式：使用 DiffDRR 投影
            self._use_diffdrr = True
            self._diffdrr = None  # lazy setup in first forward()

            odl_angles = np.asarray(self.geometry.angles, dtype=float)
            self.alphas_sorted = -odl_angles
            alphas_arr = np.asarray(param.alphas)
            betas_arr = np.asarray(param.betas)
            self.betas_sorted = np.array([
                betas_arr[np.argmin(np.abs(alphas_arr - a))]
                for a in self.alphas_sorted
            ])

            # DiffDRR PA 重定向矩阵 (RAS → DiffDRR camera)
            self._reorient = torch.tensor([
                [-1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ], dtype=torch.float32)

            # 初始平移: 源在 (0, -dso, 0)
            self._trans = torch.tensor([[0.0, -float(param.dso), 0.0]], dtype=torch.float32)
        else:
            # 原始模式：使用 ODL RayTransform
            self._use_diffdrr = False
            self.trafo = odl_torch.OperatorModule(self.ray_trafo)
            self.alphas_sorted = None
            self.betas_sorted = None

    def _setup_diffdrr(self, vol_np: np.ndarray, affine: np.ndarray, device: torch.device):
        """延迟初始化 DiffDRR。

        Parameters
        ----------
        vol_np : (D, H, W) numpy 数组，体数据
        affine : 4x4 仿射矩阵 (voxel → RAS)
        device : torch.device
        """
        from diffdrr.drr import DRR
        from torchio import LabelMap, ScalarImage, Subject

        vol_tensor = torch.from_numpy(vol_np).float().unsqueeze(0).to(device)
        aff_tensor = torch.from_numpy(affine).float().to(device)

        subject = Subject(
            volume=ScalarImage(tensor=vol_tensor, affine=aff_tensor),
            density=ScalarImage(tensor=vol_tensor, affine=aff_tensor),
            mask=LabelMap(tensor=torch.zeros_like(vol_tensor, dtype=torch.long), affine=aff_tensor),
            reorient=self._reorient.to(device),
        )

        # 探测器参数
        delx = float(self.param.sw / self.param.nw)  # pixel width
        dely = float(self.param.sh / self.param.nh)  # pixel height
        sdd = float(self.param.dso + self.param.dde)  # source-to-detector

        self._diffdrr = DRR(
            subject=subject,
            sdd=sdd,
            height=self.param.nh,
            width=self.param.nw,
            delx=delx,
            dely=dely,
            renderer="trilinear",
            patch_size=None,
        ).to(device).eval()

    def _forward_diffdrr(self, x: Tensor) -> Tensor:
        """使用 DiffDRR 进行前向投影。"""
        device = x.device
        vol_np = x.detach().cpu().numpy().squeeze()

        if self._diffdrr is None:
            self._setup_diffdrr(vol_np, self.param.affine, device)

        # 构造旋转参数: (N, 3) = [alpha, beta, 0] 弧度
        # DiffDRR 使用 ZXY intrinsic: 先绕 Z 转 alpha, 再绕 X 转 beta
        alphas = torch.from_numpy(self.alphas_sorted).float().to(device)
        betas = torch.from_numpy(self.betas_sorted).float().to(device)
        rots = torch.stack([alphas, betas, torch.zeros_like(alphas)], dim=-1)

        trans = self._trans.to(device)

        # DiffDRR 接收 -rots（参考 torch_drr.py 惯例）
        drr_img = self._diffdrr(
            -rots, trans,
            parameterization="euler_angles",
            convention="ZXY",
        )

        # drr_img shape: reshape=True 时输出 (N, 1, H, W)
        # squeeze channel dim → (N, H, W)
        # 输入带 batch 时补回 batch dim → (1, N, H, W)
        drr_img = drr_img.squeeze(1)
        if x.dim() == 4:
            drr_img = drr_img.unsqueeze(0)
        return drr_img

    def forward(self, x: Tensor) -> Tensor:
        if self._use_diffdrr:
            return self._forward_diffdrr(x)
        return self.trafo(x)

    def to_dict(self) -> dict:
        angles = self.geometry.angles
        R = np.asanyarray(self.geometry.rotation_matrix(angles))
        T = np.asanyarray(self.geometry.src_position(angles))
        d = {
            "param": self.param.to_dict(),
            "angles": angles.tolist(),
            "R": R.tolist(),
            "T": T.tolist(),
        }
        if self.alphas_sorted is not None:
            d["alphas"] = self.alphas_sorted.tolist()
            d["betas"] = self.betas_sorted.tolist()
        return d