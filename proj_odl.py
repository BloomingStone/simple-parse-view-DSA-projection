# ref: ct_geometry_projector.py of NERP
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

class Initialization_ConeBeam:
    def __init__(self, image_size, num_proj, start_angle, proj_size, affine):
        '''
        image_size: [x, y, z], assume x = y for each slice image
        proj_size: [h, w]
        '''
        self.param = {}
        
        self.image_size = image_size
        self.num_proj = num_proj
        self.proj_size = proj_size
        
        self.affine = affine
        self.spacing = np.abs(np.diag(self.affine[:3, :3]))

        # TODO 目前仅支持 spacing 为正数的情况，后续根据affine进行变换
        ## Imaging object (reconstruction objective) with object center as origin
        self.param['nx'] = image_size[0]
        self.param['ny'] = image_size[1]
        self.param['nz'] = image_size[2]
        self.param['sx'] = self.param['nx']*self.spacing[0]
        self.param['sy'] = self.param['ny']*self.spacing[1]
        self.param['sz'] = self.param['nz']*self.spacing[2]

        ## Projection view angles (ray directions)
        self.param['start_angle'] = start_angle
        self.param['end_angle'] = start_angle + np.pi
        self.param['nProj'] = num_proj

        ## Detector
        dh = 512 * 0.3 / proj_size[0]   # default proj size is 512 and dh = dw = 0.3
        dw = 512 * 0.3 / proj_size[1]
        self.param['nh'] = proj_size[0] # shape of sinogram is proj_size*proj_size
        self.param['nw'] = proj_size[1]
        self.param['sh'] = self.param['nh']*dh
        self.param['sw'] = self.param['nw']*dw
        self.param['dde'] = 400 # distance between origin and detector center (assume in x axis)
        self.param['dso'] = 1400 # distance between origin and source (assume in x axis)


def build_conebeam_gemotry(param):
    # Reconstruction space:
    reco_space = odl.uniform_discr(
        min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0, -param.param['sz'] / 2.0],
        max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0, param.param['sz'] / 2.0], 
        shape=[param.param['nx'], param.param['ny'], param.param['nz']],
        dtype='float32'
    )
    
    angle_partition = odl.uniform_partition(
        min_pt=param.param['start_angle'], 
        max_pt=param.param['end_angle'],
        shape=param.param['nProj']
    )

    detector_partition = odl.uniform_partition(
        min_pt=[-(param.param['sh'] / 2.0), -(param.param['sw'] / 2.0)], 
        max_pt=[(param.param['sh'] / 2.0), (param.param['sw'] / 2.0)],
        shape=[param.param['nh'], param.param['nw']]
    )

    # Cone-beam geometry for 3D-2D projection
    geometry = odl.tomo.ConeBeamGeometry(
        apart=angle_partition, # partition of the angle interval
        dpart=detector_partition, # partition of the detector parameter interval
        src_radius=param.param['dso'], # radius of the source circle
        det_radius=param.param['dde'], # radius of the detector circle 
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
    return reco_space, ray_trafo, FBPOper


class Projection_ConeBeam(nn.Module):
    def __init__(self, param):
        super(Projection_ConeBeam, self).__init__()
        self.param = param
        
        # RayTransform operator
        reco_space, ray_trafo, FBPOper = build_conebeam_gemotry(self.param)
        
        # Wrap pytorch module
        self.trafo = odl_torch.OperatorModule(ray_trafo)

    def forward(self, x):
        return self.trafo(x)


def get_geo(image_size, proj_size, num_proj, affine):
    start_angle = np.pi / float(num_proj * 2.0) * (num_proj - 1.0)

    # Initialize required parameters for image, view, detector
    geo_param = Initialization_ConeBeam(
        image_size=image_size, 
        num_proj=num_proj, 
        start_angle=start_angle,
        proj_size=proj_size,
        affine=affine
    )
    
    return geo_param


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
    def __init__(self, geo_param: Initialization_ConeBeam, device: torch.device):
        self.geo_param = geo_param
        nw = geo_param.param['nw']
        nh = geo_param.param['nh']
        d_so = geo_param.param['dso']   # distance of sorce to origin of world
        d_do = geo_param.param['dde']   # distance of detector to origin of world
        d_sd = d_so + d_do
        sh = geo_param.param['sh']  # size of height of detector
        
        self.fov = 2 * math.atan(sh / 2 / d_sd) / math.pi * 180
        self.width = int(nw)
        self.height = int(nh)
        
        self.bound = max(self.geo_param.param["sx"], self.geo_param.param["sy"])
        # self.zfar - self.znear = self.bound
        self.znear = d_so - self.bound/2
        self.zfar = d_so + self.bound/2
        
        self.raster_settings = RasterizationSettings(
            image_size=(self.height, self.width),
            blur_radius=0.0,
            faces_per_pixel=1,
            max_faces_per_bin=50000,
            bin_size=0
        )
        self.device = device
        
        self.geo = self._build_geometry(self.geo_param)
        
        # in pytorch3D, for the camera, z is front, y is up. need to reorient the camera
        # to front is Y and up is Z. 
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
    
    @staticmethod
    def _build_geometry(param: Initialization_ConeBeam) -> odl.tomo.ConeBeamGeometry:
        angle_partition = odl.uniform_partition(
            min_pt=param.param['start_angle'], 
            max_pt=param.param['end_angle'],
            shape=param.param['nProj']
        )

        detector_partition = odl.uniform_partition(
            min_pt=[-(param.param['sh'] / 2.0), -(param.param['sw'] / 2.0)], 
            max_pt=[(param.param['sh'] / 2.0), (param.param['sw'] / 2.0)],
            shape=[param.param['nh'], param.param['nw']]
        )

        # Cone-beam geometry for 3D-2D projection
        geometry = odl.tomo.ConeBeamGeometry(
            apart=angle_partition, # partition of the angle interval
            dpart=detector_partition, # partition of the detector parameter interval
            src_radius=param.param['dso'], # radius of the source circle
            det_radius=param.param['dde'], # radius of the detector circle 
            axis=[0, 0, 1]
        ) # rotation axis is z-axis: (0, 0, 1)
        return geometry
    
    
    @torch.no_grad()
    def render(
        self, 
        mesh_pv: pv.PolyData, 
        point_clouds: dict[str, torch.Tensor]
    ) -> tuple[Tensor, Tensor, dict[str, torch.Tensor]]:
        verts = torch.from_numpy(np.array(mesh_pv.points)).float()
        faces_np = np.array(mesh_pv.faces.reshape(-1, 4)[:, 1:])  # skip the first number which is the number of points per face
        faces = torch.from_numpy(faces_np).long()
        mesh = Meshes([verts.to(self.device)], [faces.to(self.device)])
        
        angles = self.geo.angles    # (B, )
        R = torch.from_numpy(self.geo.rotation_matrix(angles)).to(self.reorient_rot)  # (B, 3, 3), R_c2w
        R = R @ self.reorient_rot
        T = torch.from_numpy(self.geo.src_position(angles)).to(self.reorient_rot) # (B, 3)
        T = - torch.einsum("bmn, bn->bm", (R.transpose(-2, -1), T))
        
        # pytorch3D uses row vector so R_c2w for column vector don't need to be transposed. (v.T @ R_c2w).T = R_c2w.T @ v = R_c2w @ v
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
        
        # rasterize
        fragments = rasterizer(mesh.extend(angles.shape[0]))
        
        depth = fragments.zbuf[..., 0]
        silhouette = (fragments.pix_to_face[..., 0] >= 0).float()

        # rotate to match the odl coordinate system
        depth = depth.rot90(-1, [-2, -1])
        silhouette = silhouette.rot90(-1, [-2, -1])
        
        # normalize depth
        depth[depth > 0] = (depth[depth > 0] - self.znear) / self.bound
        depth[depth < 0] = 0
        
        res_clouds = {}
        for key, cloud in point_clouds.items():
            pts_screen = cameras.transform_points_screen(cloud, image_size=(self.height, self.width))
            x, y, z = pts_screen.unbind(-1)
            # normalize
            x = x / (self.width-1)
            y = y / (self.height-1)

            # rotate
            pts_screen = torch.stack([x, z, 1-y], dim=-1)
            res_clouds[key] = pts_screen
        
        return silhouette.cpu(), depth.cpu(), res_clouds


def plot_cloud_and_projs(gif_path: Path, cloud: torch.Tensor, projs: torch.Tensor):
    n_proj, h, w = projs.shape
    n_proj_, n_points, _ = cloud.shape
    assert n_proj == n_proj_
    
    # ---- 构建 XZ 平面的 StructuredGrid（只做一次）----
    x = np.linspace(0, 1, w)
    z = np.linspace(0, 1, h)
    x, z = np.meshgrid(x, z)
    y = np.zeros_like(x)
    
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
    )
    
    plotter.add_mesh(
        poly,
        color="red",
        point_size=1,
        render_points_as_spheres=True,
    )
    plotter.add_axes_at_origin(labels_off=True)    #type: ignore
    plotter.show_bounds(    #type: ignore
        bounds=[0, 1, 0, 1, 0, 1],
        grid='back',
        location='outer',
        all_edges=True,
    )
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
    fps_gif: int = 30,
    **imshow_kwargs
) -> None:
    matplotlib.use("Agg")
    frames = frames.squeeze()
    frames_np = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames

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
        image_size: tuple[int, ...], 
        num_proj: int, 
        proj_size: tuple[int, int], 
        affine: np.ndarray,
        device: torch.device
    ):
        super().__init__()
        self.geo_param = Initialization_ConeBeam(
            image_size=image_size, 
            num_proj=num_proj, 
            start_angle=0,
            proj_size=proj_size,
            affine=affine
        )
        self.num_proj = num_proj
        self.device = device
        self.affine = affine
        
        self.ct_projector = Projection_ConeBeam(self.geo_param)
        self.renderer = Torch3DLabelRenderer(self.geo_param, device)
        
    def forward(
        self, 
        data: torch.Tensor, 
        mesh: pv.PolyData, 
        point_clouds: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        projs = self.ct_projector(data).squeeze()
        
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


def process_single_file(
    nii_file: Path, 
    num_projs: Iterable[int], 
    proj_size: tuple[int, int], 
    output_dir: Path,
    vis_num_projs: list[int] | None = None
):
    device = torch.device("cuda")
    case_name = nii_file.stem.split('.')[0]
    
    nii_image = nib.loadsave.load(nii_file)
    assert isinstance(nii_image, nib.nifti1.Nifti1Image)
    data = torch.from_numpy(nii_image.get_fdata())    # [w, h, d]
    
    affine = nii_image.affine
    assert affine is not None
    affine = centerize_affine(affine, np.array(data.shape))
    
    skeleton_np = skeletonize(data.cpu().numpy())
    skeleton_tensor = torch.from_numpy(skeleton_np).to(device)
    point_clouds = {
        'bg_mask': get_label_clouds_in_world(data, affine=affine).to(device),
        'cl_mask': get_label_clouds_in_world(skeleton_tensor, affine=affine).to(device)
    }
    mesh = get_mesh_in_world(data, affine=affine)

    data = data[None].to(device)
    for n_proj in num_projs:
        data_generator = DataGenerator(
            image_size=data.shape[-3:],
            num_proj=n_proj,
            proj_size=proj_size,
            affine=affine,
            device=device
        )
        res = data_generator(data, mesh, point_clouds)
        
        sub_dir = output_dir / f"{n_proj:02d}_projs"
        sub_dir.mkdir(exist_ok=True, parents=True)
        
        torch.save(res, sub_dir / f"{case_name}.pt")
        
        if vis_num_projs is not None and n_proj in vis_num_projs:
            vis_dir = sub_dir / "vis" / f"{case_name}"
            vis_dir.mkdir(exist_ok=True, parents=True)
            projs = res["projs"]
            vmax = torch.quantile(projs, 0.995)
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
            save_gif(vis_dir/'projs.gif', projs.transpose(-1, -2), origin="lower", cmap='gray', vmax=vmax)
            save_gif(vis_dir/'depth.gif', res["depth"].transpose(-1, -2), origin="lower", cmap='gray')
            save_gif(vis_dir/'mask_2d.gif', res["mask_2d"].transpose(-1, -2), origin="lower", cmap='gray')

        torch.cuda.empty_cache()
    
    return res  # return the last case for test


def test_process_single_file():
    data_dir = Path("data")
    nii_file = Path("data/nii_size320_spacing0-4/Normal_01_lca.nii.gz")
    case_name = str(nii_file.relative_to(data_dir)).split('.')[0].replace('/', '_')
    num_projs = (2, 4, 8, 16, 32)
    proj_size = (512, 512)
    output_dir = Path("temp") / case_name
    output_dir.mkdir(exist_ok=True, parents=True)
    res = process_single_file(nii_file, num_projs, proj_size, output_dir, vis_num_projs=[32])
    
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
    input_dir: Annotated[Path, typer.Argument(help="Input directory containing .nii.gz files")],
    output_dir: Annotated[Path, typer.Argument(help="Output directory to save results")],
    proj_size: Annotated[tuple[int, int], typer.Option(help="Size of projection images")] = (512, 512),
    num_projs: Annotated[list[int], typer.Option(help="Number of projections to generate")] = [2, 4, 8, 16, 32],
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
        for vim_proj in vis_num_projs:
            if vim_proj not in num_projs:
                raise ValueError(f"{vim_proj} not in {num_projs}")
    
    mp.set_start_method("spawn", force=True)
    
    nii_files = list(input_dir.glob("*.nii.gz"))
    if not nii_files:
        raise ValueError(f"No .nii.gz files found in {input_dir}")
    
    worker = partial(
        process_single_file,
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