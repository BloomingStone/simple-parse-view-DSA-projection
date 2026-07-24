from pathlib import Path

from cyclopts import App, Parameter
from .constants import (
    DEFAULT_EXPAND,
    DEFAULT_TARGET_SHAPE,
    DEFAULT_TARGET_SPACING,
    DEFAULT_PROJ_SIZE,
)
from .projection_angles import ProjectionAngles

app = App(
    name="sparse_view_dataset", 
    help="Sparse View Dataset Processing CLI",
    default_parameter=Parameter(short_alias=True)
)

@app.command
def crop(
    input_path: Path,
    outdir: Path,
    expand: int = DEFAULT_EXPAND,
    target_shape: tuple[int, int, int] = DEFAULT_TARGET_SHAPE,
    target_spacing: float | None = DEFAULT_TARGET_SPACING,
    workers: int | None = None,
    saving_pt: bool = False,
):
    """
    Crop the input data to the region of interest (ROI) and optionally adjust spacing and shape.
    Args:
        input_path (Path): Input file path or data directory.
        outdir (Path): Output directory to save cropped data.
        expand (int): Expand ROI by this many voxels on each side.
        target_shape (tuple[int, int, int]): Target shape (w,h,d) for the cropped data.
        target_spacing (float | None): Target spacing (mm) - used for spacing adjustment if needed.
        workers (int | None): Number of workers for parallel processing (default: None, will use cpu_count).
        saving_pt (bool): Whether to save pt files.
    """
    from .preprocess import process_input_path

    process_input_path(input_path, outdir, expand, target_shape, target_spacing, workers, saving_pt)


@app.command
def project(
    resample_coronary_dir: Path,
    original_data_dir: Path,
    output_dir: Path,
    proj_size: tuple[int, int] = DEFAULT_PROJ_SIZE,
    angle_csv: Path | None = None,
    alpha_betas: list[tuple[float, float]] | None = None,
    num_random: int | None = None,
    workers: int = 2,
    devices: list[int] = [0],
    vis: bool = False,
):
    """
    Generate projection images from resampled coronary data using specified angles.
    Choose one of the following methods to specify angles:
    1. Provide a CSV file with alpha,beta columns (angle_csv).
    2. Provide a list of (alpha, beta) pairs (alpha_betas).
    3. Generate N random (alpha, beta) pairs (num_random).
    
    Args:
        resample_coronary_dir (Path): Directory containing resampled coronary nii files.
        original_data_dir (Path): Directory containing original coronary and volume data.
        output_dir (Path): Directory to save projection results.
        proj_size (tuple[int, int]): Size of the projection images.
        angle_csv (Path | None): CSV file with alpha,beta columns for angles.
        alpha_betas (list[tuple[float, float]] | None): usage: `--alpha-betas 30 45 --alpha-betas 1.2 4.7`.
        num_random (int | None): Generate N random (alpha,beta) pairs for testing.
        num_workers (int): Number of workers to use for processing.
        devices (list[int]): CUDA device ids to use; workers are split evenly across them.
        do_vis (bool): Whether to generate visualization images.
    """
    
    match (angle_csv, alpha_betas, num_random):
        case (Path() as csv_path, None, None):
            angles = ProjectionAngles.from_csv(csv_path)
        case (None, list() as ab_list, None):
            angles = ProjectionAngles.from_alpha_beta_list(ab_list)
        case (None, None, int() as n_random):
            angles = ProjectionAngles.from_random(n_random)
        case _:
            raise ValueError("Must provide exactly one of angle_csv, alpha_betas, or num_random.")

    
    from .projection import process_resampled_directory
    process_resampled_directory(
        resample_coronary_dir=resample_coronary_dir,
        original_data_dir=original_data_dir,
        output_dir=output_dir,
        proj_size=proj_size,
        angles=angles,
        num_workers=workers,
        devices=devices,
        vis=vis
    )
