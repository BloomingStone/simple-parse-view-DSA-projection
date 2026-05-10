import math

import numpy as np
import torch
from torch import Tensor
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer
)
from pytorch3d.structures import Meshes
import pyvista as pv

from .cone_beam import ProjectionConeBeam


class Torch3DLabelRenderer:
    """
    使用 PyTorch3D 对 3D mesh / point cloud 做 cone-beam 风格的投影渲染。

    约定：
    1. 输入的 mesh / point cloud 都位于同一个 world coordinate system 中；
    2. 相机按 cone-beam 几何绕物体旋转，物体本身保持静止；
    3. 渲染输出的图像尺寸与 ODL / 投影数据的 detector 尺寸一致；
    4. 点云输出使用屏幕归一化坐标，并经过轴重排，以便和当前工程里的坐标约定对齐。
    """

    def __init__(self, projection: ProjectionConeBeam, device: torch.device):
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
