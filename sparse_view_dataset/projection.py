from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable
import multiprocessing as mp
import signal
import traceback

import numpy as np
import pyvista as pv
from skimage.morphology import skeletonize
import torch
import torch.nn as nn
from tqdm import tqdm

from .constants import MU_IDODINE, MU_WATER
from .affine_transforms import centerize_affine, centerize_ori_affine, make_affine_spacing_positive
from .io import read_nii_data
from .preprocess import separate_coronary
from .cone_beam import ConeBeamParams
from .torch3d_render import Torch3DLabelRenderer
from .visualize import plot_cloud_and_projs, save_gif
from .mesh_utils import get_mesh_in_world, get_label_clouds_in_world


def density_simulation(ori_volume: np.ndarray, coronary_mask: np.ndarray) -> np.ndarray:
    res = ori_volume.copy()
    res = res / 1000.0 * MU_WATER + MU_WATER
    coronary_mask = coronary_mask.astype(np.bool_)
    res[(ori_volume > 0) & (ori_volume < 600)] = MU_WATER
    res[coronary_mask] = MU_IDODINE
    res[ori_volume < -2000] = 0
    return res


def parse_name_type(file_path: Path) -> tuple[str, str]:
    stem = file_path.stem.split(".")[0].lower()
    case_name = file_path.parent.stem
    if stem.endswith("_lca"):
        return case_name, "lca"
    if stem.endswith("_rca"):
        return case_name, "rca"
    raise ValueError(f"Cannot infer branch type from file name: {file_path}")


def get_mesh_and_clouds(resampled_cor_data: np.ndarray, resample_cor_affine: np.ndarray) -> tuple[pv.PolyData, dict[str, torch.Tensor]]:
    resampled_cor_data_tensor = torch.from_numpy(resampled_cor_data)
    skeleton_np = skeletonize(resampled_cor_data)
    skeleton_tensor = torch.from_numpy(skeleton_np)
    point_clouds = {
        "bg_mask": get_label_clouds_in_world(resampled_cor_data_tensor, affine=resample_cor_affine).cpu(),
        "cl_mask": get_label_clouds_in_world(skeleton_tensor, affine=resample_cor_affine).cpu(),
    }
    mesh = get_mesh_in_world(resampled_cor_data_tensor, affine=resample_cor_affine)
    return mesh, point_clouds


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
        self.ori_geo_param = ConeBeamParams.init_from(
            volume_size=ori_image_size,
            affine=ori_affine,
            num_proj=num_proj,
            start_angle=start_angle,
            proj_size=proj_size,
        )
        self.resampled_cor_geo_param = ConeBeamParams.init_from(
            volume_size=resampled_cor_size,
            affine=resampled_cor_affine,
            num_proj=num_proj,
            start_angle=start_angle,
            proj_size=proj_size,
        )
        self.num_proj = num_proj
        self.device = device
        self.affine = ori_affine
        
        self.ct_projector = self.ori_geo_param.get_projection()
        self.renderer = Torch3DLabelRenderer(
            self.resampled_cor_geo_param.get_projection(), 
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
        
        # Keep outputs on CPU to avoid CUDA IPC/sharing issues across processes.
        res.update({k: v.cpu() for k, v in res_clouds.items()})
        return res


def project_one_case(
    resampled_coronary_file: Path,
    original_data_dir: Path,
    num_projs: Iterable[int],
    proj_size: tuple[int, int],
    output_dir: Path,
    vis_num_projs: Iterable[int] | None = None,
) -> None:
    device = torch.device("cuda")
    
    # Find paths
    case_name, branch_type = parse_name_type(resampled_coronary_file)
    ori_coronary_file = original_data_dir / "coronary" / f"{case_name}.nii.gz"
    ori_volume_file = original_data_dir / "volume" / f"{case_name}.nii.gz"
    if not ori_coronary_file.exists():
        print(f"Original coronary file not found for case {case_name}, skipping (path: {ori_coronary_file}).")
        return
    if not ori_volume_file.exists():
        print(f"Original volume file not found for case {case_name}, skipping (path: {ori_volume_file}).")
        return
    
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
    if mesh.n_points == 0 or mesh.n_cells == 0:
        print(f"Empty mesh for {resampled_coronary_file}, skipping.")
        return


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
        
        res["case_name"] = case_name
        res["branch_type"] = branch_type
        torch.save(res, sub_dir / f"{case_name}_{branch_type}.pt")
        
        if vis_num_projs is not None and n_proj in vis_num_projs:
            vis_dir = sub_dir / "vis" / f"{case_name}_{branch_type}"
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
    
    return


def _project_one_case_safe(
    resampled_coronary_file: Path,
    original_data_dir: Path,
    num_projs: Iterable[int],
    proj_size: tuple[int, int],
    output_dir: Path,
    vis_num_projs: Iterable[int] | None = None,
) -> bool:
    try:
        project_one_case(
            resampled_coronary_file=resampled_coronary_file,
            original_data_dir=original_data_dir,
            num_projs=num_projs,
            proj_size=proj_size,
            output_dir=output_dir,
            vis_num_projs=vis_num_projs,
        )
        return True
    except Exception as e:
        print(f"Failed case {resampled_coronary_file}: {e}")
        print(traceback.format_exc())
        return False


def _pool_worker_init() -> None:
    # Let the parent process handle Ctrl+C; workers ignore SIGINT.
    signal.signal(signal.SIGINT, signal.SIG_IGN)



def process_resampled_directory(
    resample_coronary_dir: Path,
    original_data_dir: Path,
    output_dir: Path,
    proj_size: tuple[int, int] = (512, 512),
    num_projs: list[int] | tuple[int, ...] = (32,),
    num_workers: int = 4,
    vis_num_projs: list[int] | None = None,
) -> None:
    if vis_num_projs is not None:
        for vis_proj in vis_num_projs:
            if vis_proj not in num_projs:
                raise ValueError(f"{vis_proj} not in {num_projs}")

    all_nii_files = list(resample_coronary_dir.rglob("*.nii.gz"))
    nii_files = [
        p for p in all_nii_files
        if p.stem.split(".")[0].lower().endswith("_lca") or p.stem.split(".")[0].lower().endswith("_rca")
    ]
    if not nii_files:
        raise ValueError(
            f"No coronary branch files (*_lca.nii.gz / *_rca.nii.gz) found in {resample_coronary_dir}. "
            f"Found total .nii.gz files: {len(all_nii_files)}"
        )

    worker = partial(
        _project_one_case_safe,
        original_data_dir=original_data_dir,
        num_projs=num_projs,
        proj_size=proj_size,
        output_dir=output_dir,
        vis_num_projs=vis_num_projs,
    )

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=num_workers,
        initializer=_pool_worker_init,
        maxtasksperchild=1,
    )
    try:
        it = pool.imap_unordered(worker, nii_files, chunksize=1)
        for _ in tqdm(it, total=len(nii_files), desc="Processing files", ncols=80):
            pass
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, terminating workers...")
        pool.terminate()
        pool.join()
        raise
    except Exception:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()
