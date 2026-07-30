from functools import partial
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
from .affine_transforms import apply_affine, recenter_affine
from .io import read_nii_data
from .cone_beam import ConeBeamParams
from .torch3d_render import Torch3DLabelRenderer
from .visualize import plot_cloud_and_projs, save_gif
from .mesh_utils import get_mesh_in_world, get_label_clouds_in_world
from .projection_angles import ProjectionAngles


_WORKER_DEVICE: torch.device | None = None



def density_simulation(ori_volume: np.ndarray, coronary_mask: np.ndarray) -> np.ndarray:
    res = ori_volume.copy()
    res = res / 1000.0 * MU_WATER + MU_WATER
    coronary_mask = coronary_mask.astype(np.bool_)
    res[(ori_volume > 0) & (ori_volume < 600)] = MU_WATER
    res[coronary_mask] = MU_IDODINE
    res[ori_volume < -1000] = 0
    return res

def Hu_to_mu(hu_volume: np.ndarray) -> np.ndarray:
    invalid_mask = (hu_volume < -1000)  # anything below -1000 HU is considered invalid and set to 0 attenuation
    brone_area = (hu_volume > 600)  # anything above 600 HU is considered bone
    mu = hu_volume / 1000.0 * MU_WATER + MU_WATER
    mu[invalid_mask] = 0
    mu[brone_area] *= 1.5
    return mu


def parse_name_type(file_path: Path) -> tuple[str, str]:
    stem = file_path.stem.split(".")[0].lower()
    case_name = file_path.parent.stem
    if stem.endswith("_lca"):
        return case_name, "lca"
    if stem.endswith("_rca"):
        return case_name, "rca"
    raise ValueError(f"Cannot infer branch type from file name: {file_path}")


def _build_device_schedule(num_workers: int, devices: Iterable[int]) -> list[int]:
    devices = [int(device_id) for device_id in devices]
    if not devices:
        raise ValueError("devices must contain at least one CUDA device id")

    workers_per_device = num_workers // len(devices)
    if workers_per_device < 1:
        raise ValueError(
            f"num_workers={num_workers} is too small for {len(devices)} devices; "
            f"need at least {len(devices)} workers"
        )

    return [device_id for device_id in devices for _ in range(workers_per_device)]


def _get_worker_device() -> torch.device:
    if _WORKER_DEVICE is None:
        return torch.device("cuda")
    return _WORKER_DEVICE


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


def project_one_case(
    resampled_coronary_file: Path,
    original_data_dir: Path,
    angles: ProjectionAngles,
    proj_size: tuple[int, int],
    output_dir: Path,
    device: torch.device|None = None,
    vis: bool = False,
    dde: float = 400.0,
    dso: float = 1400.0,
    det_spacing: float = 0.3,
) -> None:
    device = device or _get_worker_device()

    # Find paths
    case_name, branch_type = parse_name_type(resampled_coronary_file)
    ori_volume_file = original_data_dir / "volume" / f"{case_name}.nii.gz"
    if not ori_volume_file.exists():
        print(f"Original volume file not found for case {case_name}, skipping (path: {ori_volume_file}).")
        return
    
    _project_one_case_inner(
        resampled_coronary_file=resampled_coronary_file,
        ori_volume_file=ori_volume_file,
        case_name=case_name,
        branch_type=branch_type,
        angles=angles,
        proj_size=proj_size,
        output_dir=output_dir,
        device=device,
        vis=vis,
        dde=dde, dso=dso, det_spacing=det_spacing,
    )

def _project_one_case_inner(
    resampled_coronary_file: Path,
    ori_volume_file: Path,
    case_name: str,
    branch_type: str,
    angles: ProjectionAngles,
    proj_size: tuple[int, int],
    output_dir: Path,
    device: torch.device,
    vis: bool = False,
    dde: float = 400.0,
    dso: float = 1400.0,
    det_spacing: float = 0.3,
) -> None:
    # read resampled coronary data: 用于提供roi信息
    # 目前计算骨架和点云时都使用 resampled_cor_affine_centered 来进行坐标变换，以保证和渲染器的坐标系一致
    # 此处理流程继承自之前版本的实现，后续可以考虑使用 ori_cor_data 进行处理，效果理论上一样。
    resampled_cor_data, resample_cor_affine = read_nii_data(resampled_coronary_file)

    # read original data
    ori_vol_data, ori_affine = read_nii_data(ori_volume_file)

    # \mu = \mu_w ( 1 + HU/1000 )
    # \int \mu dl = \int \mu_w ( 1 + HU/1000 ) dl = \int \mu_w dl + \int \mu_w HU/1000 dl  = \mu_w L + \mu_w / 1000 \int HU dl
    # \mu_w L is a constant offset that depends on the total length of the ray in the volume, and does not affect the relative contrast.
    # Therefore, Hu is not suitable for projection and rendering, convert to linear attenuation coefficient (mu) using a simple water-based model
    ori_vol_data = Hu_to_mu(ori_vol_data)

    def get_new_world_center() -> np.ndarray:
        ori_shape = np.array(ori_vol_data.shape)
        # 冠脉label的重心（非零体素平均坐标），比几何中心更准确
        cor_indices = np.nonzero(resampled_cor_data)
        cor_center_voxel = np.array([np.mean(idx) for idx in cor_indices])
        ori_center_voxel = (ori_shape - 1) / 2
        cor_center_world = apply_affine(cor_center_voxel, resample_cor_affine)
        ori_center_world = apply_affine(ori_center_voxel, ori_affine)
        x_cor, y_cor, z_cor = cor_center_world
        x_ori, y_ori, z_ori = ori_center_world
        
        # 冠脉label在volume中的位置偏高，如果全使用冠脉中心，会导致DRR投影时图像上方超出volume范围，显示西欧爱过不佳
        return np.array((x_cor, y_cor, (z_ori + z_cor) / 2))
    
    new_world_center = get_new_world_center()

    resample_cor_affine_centered = recenter_affine(resample_cor_affine, new_world_center)
    ori_affine_centralized = recenter_affine(ori_affine, new_world_center)
    
    skeleton_np = skeletonize(resampled_cor_data)

    ori_vol_data_tensor = torch.from_numpy(ori_vol_data.copy())[None].to(device)
    resampled_cor_data_tensor = torch.from_numpy(resampled_cor_data.copy())[None].to(device)
    skeleton_tensor = torch.from_numpy(skeleton_np).to(device)
    point_clouds = {
        'bg_mask': get_label_clouds_in_world(resampled_cor_data_tensor, affine=resample_cor_affine_centered).to(device),
        'cl_mask': get_label_clouds_in_world(skeleton_tensor, affine=resample_cor_affine_centered).to(device)
    }
    mesh = get_mesh_in_world(resampled_cor_data_tensor, affine=resample_cor_affine_centered)
    if mesh.n_points == 0 or mesh.n_cells == 0:
        print(f"Empty mesh for {resampled_coronary_file}, skipping.")
        return

    alphas = angles.alphas
    betas = angles.betas

    ori_geo_param = ConeBeamParams.init_from_angles(
        volume_size=ori_vol_data.shape,
        affine=ori_affine_centralized,
        alphas=alphas,
        betas=betas,
        proj_size=proj_size,
        dde=dde, dso=dso, det_spacing=det_spacing,
    )
    resampled_cor_geo_param = ConeBeamParams.init_from_angles(
        volume_size=resampled_cor_data.shape,
        affine=resample_cor_affine_centered,
        alphas=alphas,
        betas=betas,
        proj_size=proj_size,
        dde=dde, dso=dso, det_spacing=det_spacing,
    )
    original_ct_projector = ori_geo_param.get_projection()
    resampled_cor_ct_projector = resampled_cor_geo_param.get_projection()
    mesh_renderer = Torch3DLabelRenderer(resampled_cor_ct_projector, device)

    ori_projs = original_ct_projector(ori_vol_data_tensor).squeeze()
    label_projs = resampled_cor_ct_projector(resampled_cor_data_tensor).squeeze()

    xca_raw = torch.exp(- (ori_projs + label_projs * MU_IDODINE))
    xca_vis = torch.pow(xca_raw, 0.1)  # gamma correction for better visualization

    silhouette, depth, res_clouds = mesh_renderer.render(mesh, point_clouds)

    res = {
        "ori_projs": ori_projs.cpu(),   # 保存原始体积（吸收率）的投影结果
        "xca_raw": xca_raw.cpu(),       # 原始数据叠加冠脉标签后 通过 exp(-x) 转化为相对强度
        "projs": xca_vis.cpu(),         # xca_raw 经过 gamma 校正后的可视化结果
        "label_projs": label_projs.cpu(),
        "mask_2d": silhouette.cpu(),
        "depth": depth.cpu(),
        "ori_projs_meta": original_ct_projector.to_dict(),
        "label_projs_meta": resampled_cor_ct_projector.to_dict(),
        "angles_meta": angles.meta,
    }

    for key in res_clouds.keys():
        assert key not in res

    res.update({k: v.cpu() for k, v in res_clouds.items()})

    output_dir.mkdir(exist_ok=True, parents=True)

    res["case_name"] = case_name
    res["branch_type"] = branch_type
    torch.save(res, output_dir / f"{case_name}_{branch_type}.pt")

    if not vis:
        return
    
    vis_dir = output_dir / "vis" / f"{case_name}_{branch_type}"
    vis_dir.mkdir(exist_ok=True, parents=True)
    bg_mask = res["bg_mask"]
    cl_mask = res["cl_mask"]

    plot_cloud_and_projs(
        vis_dir / 'bg_mask_and_projs.gif',
        bg_mask,
        xca_vis,
    )

    plot_cloud_and_projs(
        vis_dir / 'cl_mask_and_depth.gif',
        cl_mask,
        depth,
    )
    save_gif(vis_dir / 'ori_projs.gif', ori_projs, cmap='gray')
    save_gif(vis_dir / 'label_projs.gif', label_projs, cmap='gray')
    save_gif(vis_dir / 'projs.gif', xca_vis, cmap='gray')
    save_gif(vis_dir / 'xca_raw.gif', xca_raw, cmap='gray')
    save_gif(vis_dir / 'depth.gif', depth)
    save_gif(vis_dir / 'mask_2d.gif', silhouette, cmap='gray')

    torch.cuda.empty_cache()

    return


def _project_one_case_safe(
    resampled_coronary_file: Path,
    original_data_dir: Path,
    angles: ProjectionAngles,
    proj_size: tuple[int, int],
    output_dir: Path,
    device: torch.device|None = None,
    vis: bool = False,
    dde: float = 400.0,
    dso: float = 1400.0,
    det_spacing: float = 0.3,
) -> tuple[bool, Path, str]:
    try:
        project_one_case(
            resampled_coronary_file=resampled_coronary_file,
            original_data_dir=original_data_dir,
            angles=angles,
            proj_size=proj_size,
            output_dir=output_dir,
            device=device,
            vis=vis,
            dde=dde, dso=dso, det_spacing=det_spacing,
        )
        return True, resampled_coronary_file, ""
    except Exception:
        return False, resampled_coronary_file, traceback.format_exc()


def _append_failed_case_log(output_dir: Path, case_path: Path, tb: str) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "failed_cases.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"==== {case_path} ====\n")
            f.write(tb)
            if not tb.endswith("\n"):
                f.write("\n")
            f.write("\n")
    except Exception:
        print(f"Failed to write failed_cases.log for {case_path}", flush=True)


def _pool_worker_init(device_schedule: list[int]) -> None:
    # Let the parent process handle Ctrl+C; workers ignore SIGINT.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but a GPU device list was provided")

    proc = mp.current_process()
    worker_index = proc._identity[0] - 1 if proc._identity else 0
    device_id = device_schedule[worker_index % len(device_schedule)]
    torch.cuda.set_device(device_id)

    global _WORKER_DEVICE
    _WORKER_DEVICE = torch.device(f"cuda:{device_id}")



def process_resampled_directory(
    resample_coronary_dir: Path,
    original_data_dir: Path,
    output_dir: Path,
    proj_size: tuple[int, int],
    angles: ProjectionAngles,
    num_workers: int = 4,
    devices: list[int] | tuple[int, ...] = (0,),
    vis: bool = False,
    dde: float = 400.0,
    dso: float = 1400.0,
    det_spacing: float = 0.3,
) -> None:
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

    device_schedule = _build_device_schedule(num_workers, devices)

    worker = partial(
        _project_one_case_safe,
        original_data_dir=original_data_dir,
        angles=angles,
        proj_size=proj_size,
        output_dir=output_dir,
        vis=vis,
        dde=dde, dso=dso, det_spacing=det_spacing,
    )

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(
        processes=len(device_schedule),
        initializer=_pool_worker_init,
        initargs=(device_schedule,),
        maxtasksperchild=1,
    )
    try:
        it = pool.imap_unordered(worker, nii_files, chunksize=1)
        for succeeded, case_path, tb in tqdm(it, total=len(nii_files), desc="Processing files", ncols=80):
            if not succeeded:
                print(f"Failed case {case_path}", flush=True)
                print(tb, flush=True)
                _append_failed_case_log(output_dir, case_path, tb)
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
