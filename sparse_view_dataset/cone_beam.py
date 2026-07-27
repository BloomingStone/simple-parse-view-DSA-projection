from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def _compute_R_T(alphas: np.ndarray, betas: np.ndarray, dso: float):
    """从 (alpha, beta) 计算旋转矩阵 R 和源位置 T。

    角度约定遵循 DICOM XA Positioner Module (C.8.7.5)：
      - alpha (Positioner Primary Angle):  绕 Z (SI) 轴，从右向前为正
      - beta  (Positioner Secondary Angle): 绕旋转后 X (RL) 轴，从头侧向前为正
    参考 https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.7.5.html#sect_C.8.7.5.1.2

    R = R_z(alpha) @ R_x(beta)
    T = R @ (0, dso, 0)
    """
    cos_a, sin_a = np.cos(alphas), np.sin(alphas)
    cos_b, sin_b = np.cos(betas), np.sin(betas)
    zero = np.zeros_like(cos_a)

    # (N, 3, 3) R_c2w = R_z(alpha) @ R_x(beta)
    R = np.stack([
        np.stack([cos_a, -cos_b * sin_a,  sin_b * sin_a], axis=-1),
        np.stack([sin_a,  cos_a * cos_b, -cos_a * sin_b], axis=-1),
        np.stack([zero,   sin_b,           cos_b],        axis=-1),
    ], axis=-2)

    T = np.einsum("nij,j->ni", R, np.array([0.0, dso, 0.0]))
    return R, T


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
        """旧接口兼容：生成均匀角度并转换为 (alpha, beta) 模式。

        生成的 alpha 从 start_angle 到 start_angle + proj_range 均匀采样，
        beta 固定为 0。实际调用 init_from_angles。
        """
        alphas = np.linspace(start_angle, start_angle + proj_range, num_proj)
        betas = np.zeros(num_proj)
        return ConeBeamParams.init_from_angles(
            volume_size=volume_size, affine=affine,
            alphas=alphas, betas=betas,
            proj_size=proj_size, dde=dde, dso=dso,
        )

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

        角度约定遵循 DICOM XA Positioner Module (C.8.7.5)：
          - alpha (Positioner Primary Angle):  绕 Z (SI) 轴，从右向前为正
          - beta  (Positioner Secondary Angle): 绕旋转后 X (RL) 轴，从头侧向前为正
        参考 https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.7.5.html#sect_C.8.7.5.1.2

        Parameters
        ----------
        volume_size : (D, H, W) 体素网格尺寸
        affine : 4x4 仿射矩阵
        alphas : (N,) Positioner Primary Angle（弧度）
        betas : (N,) 或 None, Positioner Secondary Angle（弧度）
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

        return ConeBeamParams(
            affine=affine,
            nVoxels=nVoxels,
            sVoxels=sVoxel,
            min_pt_world=min_pt_world,
            max_pt_world=max_pt_world,
            nh=nh, nw=nw, sh=sh, sw=sw,
            dde=dde, dso=dso,
            num_proj=num_proj,
            alphas=np.asarray(alphas),
            betas=np.asarray(betas),
        )

    def get_projection(self) -> "ProjectionConeBeam":
        return ProjectionConeBeam(self)

    def to_dict(self) -> dict:
        d = {
            "affine": self.affine.tolist(),
            "nVoxels": self.nVoxels.tolist(),
            "sVoxels": self.sVoxels.tolist(),
            "min_pt_world": self.min_pt_world.tolist(),
            "max_pt_world": self.max_pt_world.tolist(),
            "nh": self.nh, "nw": self.nw,
            "sh": self.sh, "sw": self.sw,
            "dde": self.dde, "dso": self.dso,
            "num_proj": self.num_proj,
        }
        if self.alphas is not None:
            d["alphas"] = self.alphas.tolist()
        if self.betas is not None:
            d["betas"] = self.betas.tolist()
        return d


class ProjectionConeBeam(nn.Module):
    """锥束 X 射线投影算子，使用 DiffDRR（trilinear 渲染器）实现。

    接受 (alpha, beta) 角度对，alpha 为 Positioner Primary Angle（绕 Z 轴），
    beta 为 Positioner Secondary Angle（绕旋转后的 X 轴）。

    角度约定遵循 DICOM XA Positioner Module (C.8.7.5)：
      - alpha: 绕 Z (SI) 轴，从右向前为正
      - beta:  绕旋转后 X (RL) 轴，从头侧向前为正
    参考 https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.7.5.html#sect_C.8.7.5.1.2
    """

    def __init__(self, param: ConeBeamParams):
        super().__init__()
        self.param = param
        self._diffdrr = None  # lazy setup in first forward()

        if param.alphas is not None:
            self.alphas = np.asarray(param.alphas, dtype=float).copy()
            self.betas = np.asarray(param.betas, dtype=float).copy()
        else:
            self.alphas = np.array([])
            self.betas = np.array([])

        # DiffDRR AP 重定向矩阵 (RAS → DiffDRR camera, 射线源在患者前方)
        self._reorient = torch.tensor([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32)

        # 初始平移: 源在 (0, dso, 0)（AP 方向）
        self._trans = torch.tensor([[0.0, float(param.dso), 0.0]], dtype=torch.float32)

    def _setup_diffdrr(self, vol_np: np.ndarray, affine: np.ndarray, device: torch.device):
        """延迟初始化 DiffDRR。"""
        from diffdrr.drr import DRR
        from torchio import LabelMap, ScalarImage, Subject

        # torchio 期望 CPU tensor（内部会调用 .numpy()），
        # 构造完 DRR 后再整个移到目标 device
        cpu = torch.device("cpu")
        vol_tensor = torch.from_numpy(vol_np).float().unsqueeze(0).to(cpu)
        aff_tensor = torch.from_numpy(affine).float().to(cpu)

        subject = Subject(
            volume=ScalarImage(tensor=vol_tensor, affine=aff_tensor),
            density=ScalarImage(tensor=vol_tensor, affine=aff_tensor),
            mask=LabelMap(tensor=torch.zeros_like(vol_tensor, dtype=torch.long), affine=aff_tensor),
            reorient=self._reorient.to(cpu),
        )

        delx = float(self.param.sw / self.param.nw)
        dely = float(self.param.sh / self.param.nh)
        sdd = float(self.param.dso + self.param.dde)

        self._diffdrr = DRR(
            subject=subject, sdd=sdd,
            height=self.param.nh, width=self.param.nw,
            delx=delx, dely=dely,
            renderer="trilinear",
            patch_size=None,
        ).to(device).eval()

    def _forward_diffdrr(self, x: Tensor) -> Tensor:
        """使用 DiffDRR 进行前向投影。"""
        device = x.device
        vol_np = x.detach().cpu().numpy().squeeze()

        if self._diffdrr is None:
            self._setup_diffdrr(vol_np, self.param.affine, device)

        # (N, 3) = [alpha, beta, 0] -> DiffDRR: ZXY intrinsic
        alphas = torch.from_numpy(self.alphas).float().to(device)
        betas = torch.from_numpy(self.betas).float().to(device)
        rots = torch.stack([alphas, betas, torch.zeros_like(alphas)], dim=-1)
        trans = self._trans.to(device)

        assert self._diffdrr is not None
        drr_img = self._diffdrr(
            rots, trans,
            parameterization="euler_angles",
            convention="ZXY",
        )
        # drr_img: (N, 1, H, W) -> squeeze channel -> (N, H, W)
        drr_img = drr_img.squeeze(1)
        if x.dim() == 4:
            drr_img = drr_img.unsqueeze(0)
        return drr_img

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_diffdrr(x)

    def to_dict(self) -> dict:
        """导出元数据，包含从 (alpha, beta) 计算的 R/T。"""
        R, T = _compute_R_T(self.alphas, self.betas, self.param.dso)
        d = {
            "param": self.param.to_dict(),
            "alphas": self.alphas.tolist(),
            "betas": self.betas.tolist(),
            "R": R.tolist(),
            "T": T.tolist(),
        }
        return d