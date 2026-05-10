# ref: ct_geometry_projector.py of NERP
from dataclasses import dataclass
from typing import Iterable, Annotated, TypeVar
from pathlib import Path
import math
from functools import partial
from multiprocessing import Pool
import multiprocessing as mp

import numpy as np
from skimage.morphology import skeletonize
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import odl
from odl.contrib import torch as odl_torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer
)
from pytorch3d.structures import Meshes
import pyvista as pv
import nibabel as nib
from tqdm import tqdm
import typer
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib

from crop_resample import separate_coronary, make_affine_spacing_positive


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
    

def init_cone_beam_params(
    volume_size: tuple[int, ...],
    affine: np.ndarray,
    num_proj: int,
    start_angle: float,
    proj_size: tuple[int, int],
    proj_range: float = np.pi,
    dde: float = 400, # distance between origin and detector center (assume in x axis)
    dso: float = 1400, # distance between origin and source (assume in x axis)
) -> ConeBeamParams:
        '''
        image_size: [x, y, z], assume x = y for each slice image
        proj_size: [h, w]
        '''
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

        dh = 512 * 0.3 / proj_size[0]   # default proj size is 512 and dh = dw = 0.3
        dw = 512 * 0.3 / proj_size[1]
        nh = proj_size[0] # shape of sinogram is proj_size*proj_size
        nw = proj_size[1]
        sh = nh * dh
        sw = nw * dw

        return ConeBeamParams(
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
            proj_range=proj_range
        )


def build_conebeam_gemotry(param: ConeBeamParams) -> tuple[
    odl.DiscretizedSpace,
    odl.tomo.ConeBeamGeometry,
    odl.tomo.RayTransform,
    odl.Operator
]:
    # Reconstruction space:
    reco_space = odl.uniform_discr(
        min_pt=[float(param.min_pt_world[0]), float(param.min_pt_world[1]), float(param.min_pt_world[2])],
        max_pt=[float(param.max_pt_world[0]), float(param.max_pt_world[1]), float(param.max_pt_world[2])], 
        shape=[int(param.nVoxels[0]), int(param.nVoxels[1]), int(param.nVoxels[2])],
        dtype='float32'
    )
    
    angle_partition = odl.uniform_partition(
        min_pt=param.start_angle, 
        max_pt=param.end_angle,
        shape=param.num_proj
    )

    detector_partition = odl.uniform_partition(
        min_pt=[-(param.sh / 2.0), -(param.sw / 2.0)], 
        max_pt=[(param.sh / 2.0), (param.sw / 2.0)],
        shape=[param.nh, param.nw]
    )

    # Cone-beam geometry for 3D-2D projection
    geometry = odl.tomo.ConeBeamGeometry(
        apart=angle_partition, # partition of the angle interval
        dpart=detector_partition, # partition of the detector parameter interval
        src_radius=param.dso, # radius of the source circle
        det_radius=param.dde, # radius of the detector circle 
        axis=[0, 0, 1]
    ) # rotation axis is z-axis: (0, 0, 1)
    
    ray_trafo = odl.tomo.RayTransform(
        vol_space=reco_space, # domain of forward projector
        geometry=geometry, # geometry of the transform
        impl='astra_cuda'
    ) # implementation back-end for the transform: ASTRA toolbox, using CUDA, 2D or 3D
    
    FBPOper = odl.tomo.fbp_op(
        ray_trafo=ray_trafo, 
        filter_type='Ram-Lak',
        frequency_scaling=1.0
    )
    
    # Reconstruction space for imaging object, RayTransform operator, Filtered back-projection operator
    return reco_space, geometry, ray_trafo, FBPOper


class Projection_ConeBeam(nn.Module):
    def __init__(self, param: ConeBeamParams):
        super(Projection_ConeBeam, self).__init__()
        self.param = param
        
        # RayTransform operator
        self.reco_space, self.geometry, self.ray_trafo, self.FBPOper = build_conebeam_gemotry(self.param)
        
        # Wrap pytorch module
        self.trafo = odl_torch.OperatorModule(self.ray_trafo)

    def forward(self, x):
        return self.trafo(x)


def get_mesh_in_voxel(label: Tensor) -> pv.PolyData:
    label_np = label.squeeze().cpu().numpy().astype(np.uint8)
    mesh = pv.wrap(label_np)\
        .contour([1], method='marching_cubes')\
        .smooth_taubin()\
        .triangulate()\
        .clean()
    return mesh


def get_mesh_in_world(label: Tensor, affine: np.ndarray) -> pv.PolyData:
    mesh = get_mesh_in_voxel(label)
    mesh.points = apply_affine(mesh.points, affine)
    return mesh


def get_label_clouds_in_world(label: Tensor, affine: np.ndarray) -> Tensor:
    clouds = torch.stack(torch.where(label), dim=-1)
    clouds = apply_affine(clouds, affine)  # to world
    return clouds


ArrayLike = TypeVar("ArrayLike", bound=Tensor|np.ndarray)
def apply_affine(
    points: ArrayLike,
    affine: np.ndarray,
) -> ArrayLike:
    """
    points:
        (3,) or (N, 3), Tensor or numpy.ndarray
    affine:
        (4, 4), Tensor or numpy.ndarray

    return:
        same type as points
    """
    is_numpy = isinstance(points, np.ndarray)
    is_single_point = (points.ndim == 1)

    # -------- shape check --------
    if is_single_point:
        assert points.shape == (3,)
    else:
        assert points.ndim == 2 and points.shape[-1] == 3

    # -------- to tensor --------
    if is_numpy:
        pts = torch.from_numpy(points)
    else:
        pts = points

    aff = torch.from_numpy(affine)
    pts = pts.to(dtype=torch.float32)
    aff = aff.to(device=pts.device, dtype=pts.dtype)

    # (3,) -> (1, 3)
    if is_single_point:
        pts = pts.unsqueeze(0)

    # -------- affine transform --------
    pts_h = F.pad(pts, (0, 1), value=1)   # (N, 4)
    out = pts_h @ aff.T
    out = out[:, :3]

    # (1, 3) -> (3,)
    if is_single_point:
        out = out.squeeze(0)

    # -------- return same type --------
    if is_numpy:
        return out.cpu().numpy()  # type: ignore
    else:
        return out               # type: ignore


class Torch3DLabelRenderer:
    """
    使用 PyTorch3D 对 3D mesh / point cloud 做 cone-beam 风格的投影渲染。

    约定：
    1. 输入的 mesh / point cloud 都位于同一个 world coordinate system 中；
    2. 相机按 cone-beam 几何绕物体旋转，物体本身保持静止；
    3. 渲染输出的图像尺寸与 ODL / 投影数据的 detector 尺寸一致；
    4. 点云输出使用屏幕归一化坐标，并经过轴重排，以便和当前工程里的坐标约定对齐。
    """

    def __init__(self, projection: Projection_ConeBeam, device: torch.device):
        self.projection = projection
        self.param = projection.param
        self.geo = projection.geometry

        nw = self.param.nw
        nh = self.param.nh
        d_so = self.param.dso   # source 到 world origin 的距离
        d_do = self.param.dde   # detector center 到 world origin 的距离
        d_sd = d_so + d_do
        sh = self.param.sh      # detector height in world units

        # 这里用 detector 高度和 source-detector 距离估计视场角。
        # 该 FOV 主要用于 FoVPerspectiveCameras 的透视投影参数。
        self.fov = 2 * math.atan(sh / 2 / d_sd) / math.pi * 180
        self.width = int(nw)
        self.height = int(nh)

        # 设置了 znear 和 zfar 与形成的空间和 ODL 定义的体素空间大致匹配，这样投影矩阵将 头棱锥体视锥体（眼坐标系）
        # 映射至归一化设备空间立方体（NDC）时，所得点云形状近乎保持不变（由于 FOV 较小，近似为正交投影，因此使用NDC
        # 坐标的畸变相对较小，可忽略）
        # 参考 https://songho.ca/opengl/gl_projectionmatrix.html
        # 这做的目的是自然且准确地得到与投影结果匹配的旋转冠脉：在眼坐标系和NDC中，投影源固定在原点。投影源绕物体旋转在
        # 这两个坐标系下表现为物体本身旋转。最终所得点云的 x, y 坐标就是投影结果的屏幕坐标（经过归一化和轴重排），z 坐
        # 标则是与投影结果对应的归一化后深度值（经过相同的 near/far clipping 和归一化处理）。
        self.bound = self.param.sVoxels.max()
        self.znear = d_so - self.bound / 2
        self.zfar = d_so + self.bound / 2

        self.raster_settings = RasterizationSettings(
            image_size=(self.height, self.width),
            blur_radius=0.0,
            faces_per_pixel=1,
            max_faces_per_bin=50000,
            bin_size=0
        )
        self.device = device

        # PyTorch3D 默认相机坐标系：
        #   +Z 为前方，+Y 为上方
        # 而这里的工程约定希望：
        #   +Y 为前方，+Z 为上方，-X 为左方
        #
        # 因此需要一个固定的重定向旋转，把外部几何坐标系映射到 PyTorch3D 相机约定。
        self.reorient_rot = torch.tensor([
            [-1,  0,  0 ],
            [0,   0,  1 ],
            [0,   1,  0 ]],
            #^    ^   ^
            #|    |   |--column 3 (k) is the front direction of camera, align at +Y (0, 1, 0)
            #|    |--column 2 (j) is the up direction of camera, align at +Z (0, 0, 1)
            #|--column 1 (i) is the left direction of camera, align at -X (-1, 0, 0)
            dtype=torch.float32,
            device=self.device
        )

    @torch.no_grad()
    def render(
        self,
        mesh_pv: pv.PolyData,
        point_clouds: dict[str, torch.Tensor]
    ) -> tuple[Tensor, Tensor, dict[str, torch.Tensor]]:
        """
        渲染 mesh 的轮廓和深度图，并将多个点云投影到同一相机坐标体系下。

        Parameters
        ----------
        mesh_pv:
            输入三角网格，顶点坐标应已处于与 `projection.geometry` 一致的 world 坐标系中。
        point_clouds:
            一组点云，字典形式输入。每个点云的 shape 通常为 (N, 3)，
            坐标同样应处于 world 坐标系中。

        Returns
        -------
        silhouette:
            二值轮廓图，shape = (B, H, W)，其中 B 为投影角度数。
        depth:
            归一化后的深度图，shape = (B, H, W)。
            这里的深度值已经根据 `znear` / `bound` 做了线性归一化。
        res_clouds:
            每个点云对应的投影结果。
            当前返回的坐标经过了 screen normalization 和轴重排，
            用于和当前工程里的图像坐标系对齐。
        """

        # PyVista PolyData -> PyTorch3D Meshes
        verts = torch.from_numpy(np.array(mesh_pv.points)).float()
        faces_np = np.array(mesh_pv.faces.reshape(-1, 4)[:, 1:])  # 每个面前面的“4”表示四边形顶点数，这里跳过
        faces = torch.from_numpy(faces_np).long()
        mesh = Meshes([verts.to(self.device)], [faces.to(self.device)])

        # 当前投影角度序列
        angles = self.geo.angles  # (B,)

        # 旋转矩阵和源点位置
        # 这里的 R/T 组合用于构造 PyTorch3D 所需的 world-to-camera 外参。
        R = torch.from_numpy(self.geo.rotation_matrix(angles)).to(self.reorient_rot)  # (B, 3, 3), R_c2w
        R = R @ self.reorient_rot
        T = torch.from_numpy(self.geo.src_position(angles)).to(self.reorient_rot)     # (B, 3)

        # PyTorch3D 的相机外参是 world-to-camera 形式：
        #   X_cam = R * X_world + T
        # 若已知相机中心在 world 中的位置 C，则平移项为：
        #   T = -R^T * C
        T = -torch.einsum("bmn,bn->bm", (R.transpose(-2, -1), T))

        cameras = FoVPerspectiveCameras(
            device=self.device,
            fov=self.fov,
            R=R,
            T=T,
            zfar=self.zfar,
            znear=self.znear
        )

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self.raster_settings
        )

        # 光栅化
        fragments = rasterizer(mesh.extend(angles.shape[0]))

        # zbuf: 每个像素对应的可见表面深度
        depth = fragments.zbuf[..., 0]

        # pix_to_face >= 0 表示该像素被 mesh 覆盖
        silhouette = (fragments.pix_to_face[..., 0] >= 0).float()

        # 将图像旋转 90°，使输出方向与当前工程中的 ODL / detector 坐标约定一致
        depth = depth.rot90(-1, [-2, -1])
        silhouette = silhouette.rot90(-1, [-2, -1])

        # 深度归一化：
        # 1) depth > 0 的像素表示有效可见表面；
        # 2) 将其映射到大致 [0, 1] 尺度，便于后续比较或监督；
        # 3) depth < 0 通常对应无效区域 / 未命中区域，直接置零。
        depth[depth > 0] = (depth[depth > 0] - self.znear) / self.bound
        depth[depth < 0] = 0

        res_clouds = {}
        for key, cloud in point_clouds.items():
            # world -> screen pixel coordinates
            # screen space 的 xy 坐标是以图像左下角为原点,与投影结果一致。
            # 
            pts_screen = cameras.transform_points_screen(cloud, image_size=(self.height, self.width))
            x, y, z = pts_screen.unbind(-1)

            # pixel coordinates -> [0, 1] normalization
            # 这里是按图像尺寸做归一化，不是标准 图像空间。
            x = x / (self.width - 1)
            y = y / (self.height - 1)

            # 轴重排，使点云投影结果与当前工程中的图像坐标约定一致：
            # - x 保持为横向归一化坐标
            # - z 作为深度通道
            # - y 取反后作为纵向坐标
            pts_screen = torch.stack([x, z, 1 - y], dim=-1)
            res_clouds[key] = pts_screen

        return silhouette.cpu(), depth.cpu(), res_clouds

def plot_cloud_and_projs(
    gif_path: Path, 
    cloud: torch.Tensor, 
    projs: torch.Tensor,
) -> None:
    n_proj, h, w = projs.shape
    n_proj_, n_points, _ = cloud.shape
    assert n_proj == n_proj_
    
    # ---- 构建 XZ 平面的 StructuredGrid（只做一次）----
    x = np.linspace(0, 1, w)
    z = np.linspace(0, 1, h)
    x, z = np.meshgrid(x, z)
    y = np.ones_like(x)
    
    grid = pv.StructuredGrid(x, y, z)

    # 初始化标量
    img0 = projs[0].cpu().numpy()        # 防止上下颠倒
    grid["value"] = img0.flatten()
    
    # ---- 初始化点云 ----
    poly = pv.PolyData(cloud[0].cpu().numpy())
    
    # ---- Plotter ----
    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif(gif_path)
    
    plotter.add_mesh(
        grid,
        scalars="value",
        cmap="gray",
    )
    
    plotter.add_mesh(
        poly,
        color="red",
        point_size=3,
        render_points_as_spheres=True,
    )
    plotter.show_bounds(    #type: ignore
        grid='back',
        location='outer',
        all_edges=True,
    )
    plotter.camera_position = 'xz'
    plotter.camera.azimuth = - 20
    plotter.camera.elevation = 10
    plotter.show(auto_close=False)
    
    # ---- 逐帧更新 ----
    for i in range(n_proj):
        # 更新灰度图
        img = projs[i].cpu().numpy()
        grid["value"] = img.flatten()
        
        # 更新点云
        poly.points = cloud[i].cpu().numpy()
        
        plotter.write_frame()
        
    plotter.close()


def save_gif(
    output_path: Path,
    frames: torch.Tensor | np.ndarray,
    fps_gif: int = 10,
    **imshow_kwargs
) -> None:
    matplotlib.use("Agg")
    frames = frames.squeeze()
    if isinstance(frames, torch.Tensor):
        frames_np = frames.cpu().numpy()
    else:
        frames_np = frames

    if "vmin" not in imshow_kwargs or "vmax" not in imshow_kwargs:
        vmin, vmax = np.percentile(frames_np, [0.05, 99.5])
        imshow_kwargs.setdefault("vmin", vmin)
        imshow_kwargs.setdefault("vmax", vmax)

    h, w = frames_np.shape[1], frames_np.shape[2]

    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = plt.axes((0, 0, 1, 1))  # 填满整个 figure
    ax.axis("off")

    ims = []
    for i in range(frames_np.shape[0]):
        im = ax.imshow(frames_np[i], animated=True, **imshow_kwargs)
        ims.append([im])

    ani = animation.ArtistAnimation(
        fig,
        ims,
        interval=1000 / fps_gif,
        blit=True,
        repeat_delay=1000
    )

    writer = animation.PillowWriter(fps=fps_gif)
    ani.save(
        output_path,
        writer=writer,
        dpi=dpi,
        savefig_kwargs={
            "pad_inches": 0
        }
    )

    plt.close(fig)


class DataGenerator(nn.Module):
    def __init__(
        self, 
        ori_image_size: tuple[int, ...], 
        ori_affine: np.ndarray,
        resampled_cor_size: tuple[int, ...],
        resampled_cor_affine: np.ndarray,
        num_proj: int, 
        proj_size: tuple[int, int], 
        device: torch.device,
        start_angle: float = 0,
    ):
        super().__init__()
        self.ori_geo_param = init_cone_beam_params(
            volume_size=ori_image_size,
            affine=ori_affine,
            num_proj=num_proj,
            start_angle=start_angle,
            proj_size=proj_size,
        )
        self.resampled_cor_geo_param = init_cone_beam_params(
            volume_size=resampled_cor_size,
            affine=resampled_cor_affine,
            num_proj=num_proj,
            start_angle=start_angle,
            proj_size=proj_size,
        )
        self.num_proj = num_proj
        self.device = device
        self.affine = ori_affine
        
        self.ct_projector = Projection_ConeBeam(self.ori_geo_param)
        self.renderer = Torch3DLabelRenderer(
            Projection_ConeBeam(self.resampled_cor_geo_param), 
            device
        )
        
    def forward(
        self, 
        data: torch.Tensor, 
        mesh: pv.PolyData, 
        point_clouds: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        projs = 1 - self.ct_projector(data).squeeze()
        
        silhouette, depth, res_clouds = self.renderer.render(mesh, point_clouds)
        
        res = {
            'projs': projs.cpu(),
            'mask_2d': silhouette.cpu(),
            'depth': depth.cpu(),
        }
        
        for key in res_clouds.keys():
            assert key not in res
        
        res.update(res_clouds)
        return res


def centerize_affine(affine: np.ndarray, voxels_shape: np.ndarray) -> np.ndarray: 
    center = (voxels_shape -1) / 2 
    T = -affine[:3, :3] @ center 
    new_affine = affine.copy() 
    new_affine[:3, 3] = T 
    return new_affine

def centerize_ori_affine(
    ori_affine: np.ndarray,
    resampled_shape: tuple[int, ...],
    resampled_affine: np.ndarray
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
    # 1. 计算重采样图像在世界坐标系下的中心点
    resampled_shape_np = np.array(resampled_shape)
    assert resampled_shape_np.shape == (3,)
    c_r_Vr = (resampled_shape_np - 1) / 2.0
    
    A_Vr_W = resampled_affine[:3, :3]
    T_r_W = resampled_affine[:3, 3]
    c_r_W = A_Vr_W @ c_r_Vr + T_r_W
    
    # 2. 提取原始矩阵的线性部分和平移部分
    A_Vori_W = ori_affine[:3, :3]
    T_ori_W = ori_affine[:3, 3]
    
    # 3. 计算新的平移项：将世界坐标原点平移到 c_r_W
    # 新坐标 = 原始世界坐标 - 中心点世界坐标
    T_ori_P = T_ori_W - c_r_W
    
    # 4. 组装新 Affine
    affine_centralized = np.eye(4)
    affine_centralized[:3, :3] = A_Vori_W
    affine_centralized[:3, 3] = T_ori_P
    
    return affine_centralized


def read_nii_data(nii_file: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.loadsave.load(nii_file)
    assert isinstance(img, nib.nifti1.Nifti1Image)
    data = img.get_fdata()
    affine = img.affine
    assert affine is not None
    return data, affine


MU_WATER = 0.02     # mm^-1
MU_IDODINE = 0.25   # mm^-1


def parse_name_type(file_path: Path) -> tuple[str, str]:
    stem = file_path.stem.split('.')[0].lower()
    case_name = file_path.parent.stem
    if stem.endswith("_lca"):
        return case_name, "lca"
    if stem.endswith("_rca"):
        return case_name, "rca"
    raise ValueError(f"Cannot infer branch type from file name: {file_path}")

def density_simulation(
    ori_volume: np.ndarray,
    coronary_mask: np.ndarray,
) -> np.ndarray:
    res = ori_volume.copy()
    res = res / 1000.0 * MU_WATER + MU_WATER  # 将 HU 转换为衰减系数
    
    coronary_mask = coronary_mask.astype(np.bool_)
    
    # # HU 值在 v_min-v_max 之间的部分, 为心腔及被对比剂增强过的, 现在恢复为水衰减系数
    # masked_volume = ori_volume[coronary_mask>0]
    # v_min = np.quantile(masked_volume, 0.1 / 100)
    # v_max = np.quantile(masked_volume, 99.9 / 100)
    # threshold_mask = (ori_volume > v_min) & (ori_volume < v_max)
    
    res[(ori_volume > 0) & (ori_volume < 600)] = MU_WATER
    
    # 对冠状动脉部分, 设为碘化钠对比剂的衰减系数
    res[coronary_mask] = MU_IDODINE
    
    # 一些图像中会将 无效值标记为 -3096, 将这部分的衰减系数设置为 0
    res[ori_volume < -2000] = 0
    
    return res


def process_single_file(
    resampled_coronary_file: Path, 
    original_data_dir: Path,
    num_projs: Iterable[int], 
    proj_size: tuple[int, int], 
    output_dir: Path,
    vis_num_projs: Iterable[int] | None = None
):
    device = torch.device("cuda")
    
    # Find paths
    case_name, branch_type = parse_name_type(resampled_coronary_file)
    ori_coronary_file = original_data_dir / "coronary" / f"{case_name}.nii.gz"
    ori_volume_file = original_data_dir / "volume" / f"{case_name}.nii.gz"
    
    # read original data
    resampled_cor_data, resample_cor_affine = read_nii_data(resampled_coronary_file)
    ori_cor_data, ori_affine = read_nii_data(ori_coronary_file)
    ori_vol_data, ori_affine_ = read_nii_data(ori_volume_file)
    assert np.allclose(ori_affine, ori_affine_), f"Affine of coronary and volume do not match for case {case_name}"
    
    # separate coronary branches and select the branch of interest
    ori_cor_branches = separate_coronary(ori_cor_data)
    if branch_type not in ori_cor_branches:
        raise ValueError(f"Branch type {branch_type} not found in {ori_coronary_file}")
    branch_data = ori_cor_branches[branch_type]
    
    # ODL need positive spacing, so make affine spacing positive and adjust the data accordingly
    ori_cor_data, ori_affine_positive = make_affine_spacing_positive(ori_cor_data, ori_affine)
    ori_vol_data, _ = make_affine_spacing_positive(ori_vol_data, ori_affine)
    branch_data, _ = make_affine_spacing_positive(branch_data, ori_affine)
    
    # 将重采样图像的 affine 中心化，使得重采样label的中心点在世界坐标系中的位置为 (0, 0, 0)
    # 同时将 原始分辨率图像对齐到中心化后的重采样图像
    resample_cor_affine_centered = centerize_affine(resample_cor_affine, np.array(resampled_cor_data.shape))
    ori_affine_centralized = centerize_ori_affine(ori_affine_positive, resampled_cor_data.shape, resample_cor_affine)

    # Make coronary branch density as iodine contrast, and background density as water, to better simulate the projection image.
    density = density_simulation(ori_vol_data, branch_data)
    
    skeleton_np = skeletonize(resampled_cor_data)
    
    density_tensor = torch.from_numpy(density).to(device)
    resampled_cor_data_tensor = torch.from_numpy(resampled_cor_data).to(device)
    skeleton_tensor = torch.from_numpy(skeleton_np).to(device)
    point_clouds = {
        'bg_mask': get_label_clouds_in_world(resampled_cor_data_tensor, affine=resample_cor_affine_centered).to(device),
        'cl_mask': get_label_clouds_in_world(skeleton_tensor, affine=resample_cor_affine_centered).to(device)
    }
    mesh = get_mesh_in_world(resampled_cor_data_tensor, affine=resample_cor_affine_centered)

    # resampled_cor_data = resampled_cor_data[None].to(device)
    density_tensor = density_tensor[None].to(device)
    for n_proj in num_projs:
        data_generator = DataGenerator(
            ori_image_size=density_tensor.shape[-3:],
            ori_affine=ori_affine_centralized,
            resampled_cor_size=resampled_cor_data.shape[-3:],
            resampled_cor_affine=resample_cor_affine_centered,
            num_proj=n_proj,
            proj_size=proj_size,
            device=device
        )
        res = data_generator(density_tensor, mesh, point_clouds)
        
        sub_dir = output_dir / f"{n_proj:02d}_projs"
        sub_dir.mkdir(exist_ok=True, parents=True)
        
        torch.save(res, sub_dir / f"{case_name}.pt")
        
        if vis_num_projs is not None and n_proj in vis_num_projs:
            vis_dir = sub_dir / "vis" / f"{case_name}"
            vis_dir.mkdir(exist_ok=True, parents=True)
            projs = res["projs"]
            plot_cloud_and_projs(
                vis_dir/'bg_mask_and_projs.gif',
                res["bg_mask"],
                projs,
            )
            
            plot_cloud_and_projs(
                vis_dir/'cl_mask_and_depth.gif',
                res["cl_mask"],
                res["depth"],
            )
            save_gif(vis_dir/'projs.gif', projs.transpose(-1, -2), origin="lower", cmap='gray')
            save_gif(vis_dir/'depth.gif', res["depth"].transpose(-1, -2), origin="lower", cmap='gray')
            save_gif(vis_dir/'mask_2d.gif', res["mask_2d"].transpose(-1, -2), origin="lower", cmap='gray')

        torch.cuda.empty_cache()
    
    return res  # type: ignore return the last case for test


def test_process_single_file():
    data_dir = Path("data")
    nii_file = Path("data/asoca_size128_spacing0-7/Diseased_17/Diseased_17_lca.nii.gz")
    case_name = str(nii_file.relative_to(data_dir)).split('.')[0].replace('/', '_')
    proj_size = (512, 512)
    
    output_dir = Path("temp") / case_name
    output_dir.mkdir(exist_ok=True, parents=True)
    res = process_single_file(
        resampled_coronary_file=nii_file,
        original_data_dir=Path("./ori_data/asoca"),
        num_projs=(32, ),
        proj_size=proj_size,
        output_dir=output_dir,
        vis_num_projs=(32, )
    )
    
    def save_nii(key: str, value: torch.Tensor):
        nib.loadsave.save(
            nib.nifti1.Nifti1Image(value.cpu().numpy(), affine=np.eye(4)),
            output_dir / f"{key}.nii.gz"
        )
    
    def point_cloud_to_image(points: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        W, H = image_size
        B, N, _ = points.shape
        x, y, z = points.unbind(-1)
        ix = (x * (W - 1)).long().clamp(0, W - 1)
        iz = (z * (H - 1)).long().clamp(0, H - 1)
        ib = torch.arange(B).to(ix).view(B, 1).expand(B, N)
        res = torch.zeros(B, W, H).to(points)
        res[ib, ix, iz] = y
        return res
    
    clouds_keys = ["bg_mask", "cl_mask"]
    for k, v in res.items():
        if k in clouds_keys:
            save_nii(k, point_cloud_to_image(v, proj_size))
        else:
            save_nii(k, v)


def main(
    resample_coronary_dir: Annotated[Path, typer.Argument(help="Input directory containing resampled coronary nii files")],
    original_data_dir: Annotated[Path, typer.Argument(help="Input directory containing coronary dir and volume dir")],
    output_dir: Annotated[Path, typer.Argument(help="Output directory to save results")],
    proj_size: Annotated[tuple[int, int], typer.Option(help="Size of projection images")] = (512, 512),
    num_projs: Annotated[list[int], typer.Option(help="Number of projections to generate")] = [32, ],
    num_workers: Annotated[int, typer.Option(help="Number of workers to use")] = 4,
    vis_num_projs: Annotated[list[int]|None, typer.Option(help="Number of projections to visualize")] = None,
):
    """
    Main function for processing 3D medical images (.nii.gz) to generate 2D projections.

    Processing Pipeline:
        1. Loads all .nii.gz files from input directory
        2. For each file:
            - Loads 3D volume data and generates mesh
            - Generates multiple 2D projections (specified by num_projs)
            - Saves projections and derived data (depth maps, masks)
            - Optionally saves visualizations when num_projs=32
        3. Uses multiprocessing for parallel processing of files
    """
    if vis_num_projs is not None:
        for vis_proj in vis_num_projs:
            if vis_proj not in num_projs:
                raise ValueError(f"{vis_proj} not in {num_projs}")
    
    mp.set_start_method("spawn", force=True)
    
    nii_files = list(resample_coronary_dir.rglob("*.nii.gz"))
    if not nii_files:
        raise ValueError(f"No .nii.gz files found in {resample_coronary_dir}")
    
    worker = partial(
        process_single_file,
        original_data_dir=original_data_dir,
        num_projs=num_projs,
        proj_size=proj_size,
        output_dir=output_dir,
        vis_num_projs=vis_num_projs
    )
    
    print(f"Processing {len(nii_files)} files with {num_workers} workers...")
    with Pool(processes=num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(worker, nii_files),
            total=len(nii_files),
            desc="Processing files",
            ncols=80,
        ):
            pass

if __name__ == '__main__':
    typer.run(main)