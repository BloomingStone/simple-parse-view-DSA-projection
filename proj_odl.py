# copy from ct_geometry_projector.py of NERP
from pathlib import Path
import math
from functools import partial
from multiprocessing import Pool
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import numpy as np
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

from saver import save_gif, save_deepthmap_gif


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
        self.reso = np.abs(np.diag(self.affine[:3, :3]))

        ## Imaging object (reconstruction objective) with object center as origin
        self.param['nx'] = image_size[0]
        self.param['ny'] = image_size[1]
        self.param['nz'] = image_size[2]
        self.param['sx'] = self.param['nx']*self.reso[0]
        self.param['sy'] = self.param['ny']*self.reso[1]
        self.param['sz'] = self.param['nz']*self.reso[2]

        ## Projection view angles (ray directions)
        self.param['start_angle'] = start_angle
        self.param['end_angle'] = start_angle + np.pi
        self.param['nProj'] = num_proj

        ## Detector
        self.param['nh'] = proj_size[0] # shape of sinogram is proj_size*proj_size
        self.param['nw'] = proj_size[1]
        self.param['sh'] = self.param['nh']*0.3
        self.param['sw'] = self.param['nw']*0.3
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


# Projector
class Projection_ConeBeam(nn.Module):
    def __init__(self, param):
        super(Projection_ConeBeam, self).__init__()
        self.param = param
        self.reso = param.reso
        
        # RayTransform operator
        reco_space, ray_trafo, FBPOper = build_conebeam_gemotry(self.param)
        
        # Wrap pytorch module
        self.trafo = odl_torch.OperatorModule(ray_trafo)
        
        self.back_projector = odl_torch.OperatorModule(ray_trafo.adjoint)

    def forward(self, x):
        return self.trafo(x)
    
    def back_projection(self, x):
        x = self.back_projector(x)
        return x


# FBP reconstruction
class FBP_ConeBeam(nn.Module):
    def __init__(self, param):
        super(FBP_ConeBeam, self).__init__()
        self.param = param
        self.reso = param.reso
        
        reco_space, ray_trafo, FBPOper = build_conebeam_gemotry(self.param)
        
        self.fbp = odl_torch.OperatorModule(FBPOper)

    def forward(self, x):
        x = self.fbp(x)
        return x

    def filter_function(self, x):
        raise NotImplementedError


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


class ConeBeam3DProjector():
    def __init__(self, geo_param: Initialization_ConeBeam):
        # Forward projection function
        self.forward_projector = Projection_ConeBeam(geo_param)

        # Filtered back-projection
        self.fbp = FBP_ConeBeam(geo_param)

    def forward_project(self, volume):
        '''
        Arguments:
            volume: torch tensor with input size (B, C, img_x, img_y, img_z)
        '''

        proj_data = self.forward_projector(volume)

        return proj_data

    def backward_project(self, projs):
        '''
        Arguments:
            projs: torch tensor with input size (B, num_proj, proj_size_h, proj_size_w)
        '''

        volume = self.fbp(projs)

        return volume
    
    def freeze(self):
        for param in self.forward_projector.parameters():
            param.requires_grad = False
        for param in self.fbp.parameters():
            param.requires_grad = False


def get_mesh_in_voxel(label: Tensor, device: torch.device, max_points: int=10000) -> pv.PolyData:
    label_big = F.interpolate(label.squeeze()[None, None].to(torch.float16).to(device), scale_factor=2, mode='nearest').cpu().numpy().squeeze()
    label_big = (label_big>0.5).astype(np.uint8)
    mesh = pv.wrap(label_big)\
        .contour([1], method='marching_cubes')\
        .smooth_taubin(
            n_iter=30, pass_band=0.001, normalize_coordinates=True)\
        .triangulate()\
        .decimate_pro(
            reduction=0.8,          # 减少 80% 三角面片
            preserve_topology=True, # 防止破洞
            feature_angle=30.0
        )\
        .triangulate()\
        .clean()
    mesh.points /= 2.0  # 因为上采样了2倍，所以点坐标要除以2
    return mesh

def get_mesh_in_world(label: Tensor, affine: np.ndarray, device: torch.device, max_points: int=10000) -> pv.PolyData:
    mesh = get_mesh_in_voxel(label, device, max_points)
    mesh.points = apply_affine(mesh.points, affine)
    return mesh

def apply_affine(points: Tensor | np.ndarray, affine: np.ndarray) -> np.ndarray:
    assert len(points.shape) == 2 and points.shape[-1] == 3
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)
    points = points.to(torch.float32)
    _affine = torch.from_numpy(affine).to(device=points.device, dtype=points.dtype)
    new_points = F.pad(points, (0, 1), "constant", 1)   # shape=(N, 4), [x, y, z, 1]
    new_points = new_points @ _affine.T
    new_points = new_points[:, :3]
    return new_points.cpu().numpy()


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    res = torch.eye(4)
    res[:3, :3] = torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))
    return res


class Torch3DLabelRenderer:
    def __init__(self, geo_param: Initialization_ConeBeam):
        self.geo_param = geo_param
        nw = geo_param.param['nw']
        nh = geo_param.param['nh']
        sdd = geo_param.param['dde'] + geo_param.param['dso']
        sh = geo_param.param['sh']  # size of height
        
        self.fov = 2 * math.atan(sh / 2 / sdd) / math.pi * 180
        self.width = int(nw)
        self.height = int(nh)
        self.zfar = sdd*1.2
        self.raster_settings = RasterizationSettings(
            image_size=(self.height, self.width),
            blur_radius=0.0,
            faces_per_pixel=1
        )
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            raise RuntimeError("No CUDA device available for Torch3DLabelRenderer")
        
        self.geo = self._build_geometry(self.geo_param)
        
        # Rotates the C-arm about the x-axis by 90 degrees
        self.reorient_rot = np.array([
            [1,  0,  0 ],
            [0,  0,  -1],
            [0,  1,  0 ]],
            dtype=np.float32
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
    
    def _calculate_R_T(self, angle: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        cauculate R T manully by angle
        """
        R_z_alpha = _axis_angle_rotation("Z", torch.tensor(angle))
        translate = torch.eye(4)
        sod = self.geo_param.param['dso']
        translate[:3, 3] = torch.tensor([0.0, sod, 0.0])
        reorient = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        
        # internal rotation (zxy) can translate to external rotation with oppisite order (YXZ), so here first rotate around X_world, then Y_world
        M_c2w_gt = R_z_alpha @ translate @ reorient
        
        R = M_c2w_gt[:3, :3]
        T = M_c2w_gt[:3, 3]
        return R, T

    @torch.no_grad()
    def render(self, mesh_pv: pv.PolyData):
        verts = torch.from_numpy(np.array(mesh_pv.points)).float()
        faces_np = np.array(mesh_pv.faces.reshape(-1, 4)[:, 1:])  # skip the first number which is the number of points per face
        faces = torch.from_numpy(faces_np).long()
        mesh = Meshes([verts.to(self.device)], [faces.to(self.device)])
        
        angles = self.geo.angles
        silhouette_final = torch.zeros(len(angles), self.height, self.width)
        depth_final = torch.zeros(len(angles), self.height, self.width)
        for i, angle in enumerate(angles):
            R = self.geo.rotation_matrix(angle) @ self.reorient_rot
            T = self.geo.src_position(angle)

            R = torch.from_numpy(R.squeeze())
            T = torch.from_numpy(T.squeeze())
            T = -T      # TODO why need to multiply -1?
            
            cameras = FoVPerspectiveCameras(
                device=self.device,
                fov=self.fov,
                R=R[None].to(self.device),  # pytorch3D uses row vector so R_c2w for column vector don't need to be transposed. (v.T @ R_c2w).T = R_c2w.T @ v = R_c2w @ v
                T=(-R.T @ T)[None].to(self.device), 
                zfar=self.zfar
            )
            
            rasterizer = MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            )
            
            # rasterize
            fragments = rasterizer(mesh)
            depth = fragments.zbuf[0, ..., 0]  # depth buffer
            # silhouette
            silhouette = (fragments.pix_to_face[..., 0] >= 0).float()
        
            silhouette_final[i] = silhouette.cpu()
            depth_final[i] = depth.cpu()

        return silhouette_final, depth_final


class DataGenerator(nn.Module):
    def __init__(self, image_size: tuple[int, ...], num_proj: int, proj_size: tuple[int, int], affine: np.ndarray):
        super().__init__()
        self.geo_param = geo_param = Initialization_ConeBeam(
            image_size=image_size, 
            num_proj=num_proj, 
            start_angle=0,
            proj_size=proj_size,
            affine=affine
        )
        
        self.ct_projector = ConeBeam3DProjector(geo_param)
        self.renderer = Torch3DLabelRenderer(geo_param)
        
    def forward(self, data: torch.Tensor, mesh: pv.PolyData) -> dict[str, Tensor]:
        # Perform projections
        projs = self.ct_projector.forward_project(data)
        silhouette, depth = self.renderer.render(mesh)
        
        return {
            'projs': projs,
            'silhouette': silhouette,
            'depth': depth
        }


def centerize_affine(affine: np.ndarray, voxels_shape: np.ndarray) -> np.ndarray: 
    center = (voxels_shape -1) / 2 
    T = -affine[:3, :3] @ center 
    new_affine = affine.copy() 
    new_affine[:3, 3] = T 
    return new_affine


def test():
    nii_file = "data/nii/Diseased_1_lca.nii.gz"
    nii_image = nib.loadsave.load(nii_file)
    assert isinstance(nii_image, nib.nifti1.Nifti1Image)
    nii_data_np = nii_image.get_fdata()
    affine = nii_image.affine
    assert affine is not None
    affine = centerize_affine(affine, np.array(nii_data_np.shape))
    
    nii_data = torch.from_numpy(nii_data_np)
    
    proj_size = (512, 512)
    num_proj = 32
    
    data_generator = DataGenerator(
        image_size=nii_data_np.shape[-3:],
        num_proj=num_proj,
        proj_size=proj_size,
        affine=affine
    )
    
    mesh = get_mesh_in_world(nii_data, device=torch.device("cpu"), affine=affine)
    
    res = data_generator(nii_data, mesh)
    
    test_dir = Path("temp")
    test_dir.mkdir(exist_ok=True)
    
    projs = res["projs"]
    vmax = torch.quantile(projs, 0.995)
    save_gif(test_dir/'projs.gif', projs.transpose(-1, -2), origin="lower", cmap='gray', vmax=vmax)
    save_deepthmap_gif(test_dir/'depth.gif', res["depth"])
    save_gif(test_dir/'silhouette.gif', res["silhouette"], cmap='gray')
    
    for n, v in res.items():
        nii_image = nib.nifti1.Nifti1Image(v.numpy().squeeze(), np.eye(4))
        nib.loadsave.save(nii_image, test_dir/f"{n}.nii.gz")
    
    torch.save(res, test_dir/'res.pt')
    
    return res


def process_single_file(nii_file, num_projs, proj_size, output_dir):
    case_name = nii_file.stem.split('.')[0]
    nii_image = nib.loadsave.load(nii_file)
    assert isinstance(nii_image, nib.nifti1.Nifti1Image)
    nii_data_np = nii_image.get_fdata()
    affine = nii_image.affine
    assert affine is not None
    affine = centerize_affine(affine, np.array(nii_data_np.shape))
    
    nii_data = torch.from_numpy(nii_data_np)[None]
    device = torch.device("cuda")
    mesh = get_mesh_in_world(nii_data, affine=affine, device=device)

    for n_proj in num_projs:
        data_generator = DataGenerator(
            image_size=nii_data_np.shape[-3:],
            num_proj=n_proj,
            proj_size=proj_size,
            affine=affine
        )
        res = data_generator(nii_data, mesh)
        
        sub_dir = output_dir / f"{n_proj:02d}_projs"
        sub_dir.mkdir(exist_ok=True, parents=True)
        
        res_np = {}
        for n, v in res.items():
            res_np[n] = v.cpu().numpy()
        
        # 保存结果文件路径和内容
        save_path = sub_dir / f"{case_name}.npz"
        np.savez_compressed(save_path, **res_np)
        
        # 可视化只在 32 投影时保存
        if n_proj == 32:
            vis_dir = sub_dir / "vis"
            vis_dir.mkdir(exist_ok=True, parents=True)
            projs = res["projs"]
            vmax = torch.quantile(projs, 0.995)
            save_gif(vis_dir/f'{case_name}_projs.gif', projs.transpose(-1, -2), origin="lower", cmap='gray', vmax=vmax)
            save_deepthmap_gif(vis_dir/f'{case_name}_depth.gif', res["depth"])

        torch.cuda.empty_cache()
    
    return res

if __name__ == '__main__':
    # input_dir=Path("data/nii")
    # output_dir=Path("data/nii_projs")
    
    input_dir=Path("data/nii_size320_spacing0-4")
    output_dir=Path("data/nii_size320_spacing0-4_projs")
    
    proj_size: tuple[int, int] = (512, 512)
    num_projs: tuple[int, ...] = (2, 4, 8, 16, 32)
    num_workers: int = 8
    
    nii_files = list(input_dir.glob("*.nii.gz"))
    if not nii_files:
        raise ValueError(f"No .nii.gz files found in {input_dir}")
    
    worker = partial(
        process_single_file,
        num_projs=num_projs,
        proj_size=proj_size,
        output_dir=output_dir,
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