from pathlib import Path

import typer
from .constants import (
    DEFAULT_EXPAND,
    DEFAULT_TARGET_SHAPE,
    DEFAULT_TARGET_SPACING,
    DEFAULT_PROJ_SIZE,
    DEFAULT_NUM_PROJS
)


app = typer.Typer(help="Sparse coronary projection dataset pipeline.")


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
    num_projs: list[int] = typer.Option(DEFAULT_NUM_PROJS, help="Number of projections to generate"),
    num_workers: int = typer.Option(4, help="Number of workers to use"),
    vis_num_projs: list[int] | None = typer.Option(None, help="Number of projections to visualize"),
):
    from .projection import process_resampled_directory

    process_resampled_directory(
        resample_coronary_dir=resample_coronary_dir,
        original_data_dir=original_data_dir,
        output_dir=output_dir,
        proj_size=proj_size,
        num_projs=num_projs,
        num_workers=num_workers,
        vis_num_projs=vis_num_projs,
    )
