from pathlib import Path

import numpy as np
import typer
from .constants import (
    DEFAULT_EXPAND,
    DEFAULT_TARGET_SHAPE,
    DEFAULT_TARGET_SPACING,
    DEFAULT_PROJ_SIZE,
)


app = typer.Typer(help="Sparse coronary projection dataset pipeline.")


def _parse_alpha_beta_pairs(pairs_str: str) -> tuple[np.ndarray, np.ndarray]:
    """解析 "--alpha-betas '0,0;0.1,0.05;0.2,0.1'" 格式的字符串。

    返回 (alphas, betas) 两个 numpy 数组。
    """
    pairs = pairs_str.split(";")
    alphas, betas = [], []
    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue
        parts = pair.split(",")
        if len(parts) != 2:
            raise ValueError(f"无效的角度对: '{pair}'，格式应为 'alpha,beta'")
        alphas.append(float(parts[0]))
        betas.append(float(parts[1]))
    if not alphas:
        raise ValueError("未解析到任何角度对")
    return np.deg2rad(np.array(alphas)), np.deg2rad(np.array(betas))


@app.command("crop")
def crop(
    input_path: Path = typer.Argument(..., help="Input file path or data directory"),
    outdir: Path = typer.Argument(..., help="Output directory"),
    expand: int = typer.Option(DEFAULT_EXPAND, help="Expand ROI by this many voxels on each side"),
    target_shape: tuple[int, int, int] = typer.Option(DEFAULT_TARGET_SHAPE, help="Target shape (w,h,d)"),
    target_spacing: float | None = typer.Option(DEFAULT_TARGET_SPACING, help="Target spacing (mm) - used for spacing adjustment if needed"),
    workers: int | None = typer.Option(None, help="Number of workers for parallel processing (default: None, will use cpu_count)"),
    saving_pt: bool = typer.Option(False, help="Save pt files"),
):
    from .preprocess import process_input_path

    process_input_path(input_path, outdir, expand, target_shape, target_spacing, workers, saving_pt)


@app.command("project")
def project(
    resample_coronary_dir: Path = typer.Argument(..., help="Input directory containing resampled coronary nii files"),
    original_data_dir: Path = typer.Argument(..., help="Input directory containing coronary dir and volume dir"),
    output_dir: Path = typer.Argument(..., help="Output directory to save results"),
    proj_size: tuple[int, int] = typer.Option(DEFAULT_PROJ_SIZE, help="Size of projection images"),
    angle_csv: list[Path] = typer.Option(None, "--angle-csv", help="CSV file(s) with alpha,beta columns (can be repeated)"),
    alpha_betas: str | None = typer.Option(None, "--alpha-betas", help="Alpha/beta pairs as 'alpha1,beta1;alpha2,beta2;...'"),
    num_random: int | None = typer.Option(None, "--num-random", help="Generate N random (alpha,beta) pairs for testing"),
    num_workers: int = typer.Option(4, help="Number of workers to use"),
    devices: list[int] = typer.Option([0], "--device", "-d", help="CUDA device ids to use; workers are split evenly across them"),
    num_of_vis: int = typer.Option(0, "--num-of-vis", help="Number of first angles to visualize (0 = no visualization)"),
):
    from .projection import process_resampled_directory, make_angle_configs

    # 解析角度配置
    alpha_betas_list = None
    if alpha_betas is not None:
        a, b = _parse_alpha_beta_pairs(alpha_betas)
        alpha_betas_list = [(a, b)]

    angle_configs = make_angle_configs(
        csv_paths=angle_csv if angle_csv else None,
        alpha_betas_list=alpha_betas_list,
        num_random=num_random,
    )

    process_resampled_directory(
        resample_coronary_dir=resample_coronary_dir,
        original_data_dir=original_data_dir,
        output_dir=output_dir,
        proj_size=proj_size,
        angle_configs=angle_configs,
        num_workers=num_workers,
        devices=devices,
        num_of_vis=num_of_vis,
    )
