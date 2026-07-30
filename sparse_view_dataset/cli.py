import json
from pathlib import Path

from cyclopts import App, Parameter
from .constants import (
    DEFAULT_EXPAND,
    DEFAULT_TARGET_SHAPE,
    DEFAULT_TARGET_SPACING,
    DEFAULT_PROJ_SIZE,
    DEFAULT_DDE,
    DEFAULT_DSO,
    DEFAULT_DET_SPACING,
)
from .projection_angles import ProjectionAngles

app = App(
    name="sparse_view_dataset", 
    help="Sparse View Dataset Processing CLI",
    default_parameter=Parameter(short_alias=True)
)


def _parse_ref_dicom_or_json(
    ref_path: Path,
) -> dict:
    """从 DICOM 或 JSON 文件中读取投影相关参数。

    支持的字段（JSON 键名 / DICOM 标签）：
      - Rows / (0028,0010)        → proj_size[0]
      - Columns / (0028,0011)     → proj_size[1]
      - PositionerPrimaryAngle / (0018,1510)  → alpha (度)
      - PositionerSecondaryAngle / (0018,1511) → beta (度)
      - DistanceSourceToDetector / (0018,1110) → SDD (mm)
      - DistanceSourceToPatient / (0018,1111)  → SOD (mm)
      - ImagerPixelSpacing / (0018,1164)       → det_spacing (mm/pixel)

    Returns:
        dict: 包含以下键（部分可能缺失）：
            proj_size, alpha, beta, dde, dso, det_spacing
    """
    ext = ref_path.suffix.lower()
    info: dict = {}

    if ext == ".json":
        with open(ref_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        def _get(key: str):
            v = data.get(key)
            if v is None:
                return None
            if isinstance(v, (list, tuple)):
                return float(v[0])
            return float(v)

        rows = _get("Rows")
        cols = _get("Columns")
        if rows is not None and cols is not None:
            info["proj_size"] = (int(rows), int(cols))
        alpha = _get("PositionerPrimaryAngle")
        beta = _get("PositionerSecondaryAngle")
        if alpha is not None and beta is not None:
            info["alpha"] = alpha
            info["beta"] = beta
        sdd = _get("DistanceSourceToDetector")
        sod = _get("DistanceSourceToPatient")
        if sdd is not None and sod is not None:
            info["dso"] = sod
            info["dde"] = sdd - sod
        spacing = data.get("ImagerPixelSpacing")
        if spacing is not None and len(spacing) > 0:
            info["det_spacing"] = float(spacing[0])

    elif ext == ".dcm":
        try:
            import pydicom
        except ImportError:
            raise ImportError("pydicom is required to read .dcm files. Install it with: pip install pydicom")

        ds = pydicom.dcmread(str(ref_path))

        if hasattr(ds, "Rows") and hasattr(ds, "Columns"):
            info["proj_size"] = (int(ds.Rows), int(ds.Columns))

        if hasattr(ds, "PositionerPrimaryAngle") and hasattr(ds, "PositionerSecondaryAngle"):
            info["alpha"] = float(ds.PositionerPrimaryAngle)
            info["beta"] = float(ds.PositionerSecondaryAngle)

        if hasattr(ds, "DistanceSourceToDetector") and hasattr(ds, "DistanceSourceToPatient"):
            sdd = float(ds.DistanceSourceToDetector)
            sod = float(ds.DistanceSourceToPatient)
            info["dso"] = sod
            info["dde"] = sdd - sod

        if hasattr(ds, "ImagerPixelSpacing") and ds.ImagerPixelSpacing:
            info["det_spacing"] = float(ds.ImagerPixelSpacing[0])

    else:
        raise ValueError(f"Unsupported reference file format: {ext}. Use .json or .dcm")

    return info


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
    Crop the input data to the region of interest (ROI) and optionally adjust spacing or shape.
    
    Note: target_spacing and target_shape are mutually exclusive.
    - If target_spacing is set, the data is resampled to that spacing (target_shape is ignored).
    - Otherwise, the data is resampled to target_shape.
    
    Args:
        input_path (Path): Input file path or data directory.
        outdir (Path): Output directory to save cropped data.
        expand (int): Expand ROI by this many voxels on each side.
        target_shape (tuple[int, int, int]): Target shape (w,h,d) for the cropped data.
                      Only used when target_spacing is None.
        target_spacing (float | None): Target spacing (mm). When set, target_shape is ignored
                                       and output is resampled to this spacing only.
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
    alpha_betas: list[tuple[float, float]] | None = None,
    num_random: int | None = None,
    workers: int = 2,
    devices: list[int] = [0],
    vis: bool = False,
    dde: float = DEFAULT_DDE,
    dso: float = DEFAULT_DSO,
    det_spacing: float = DEFAULT_DET_SPACING,
):
    """
    Generate projection images from resampled coronary data using specified angles.
    Choose one of the following methods to specify angles:
    1. Provide a list of (alpha, beta) pairs (alpha_betas).
    2. Generate N random (alpha, beta) pairs (num_random).
    
    Args:
        resample_coronary_dir (Path): Directory containing resampled coronary nii files.
        original_data_dir (Path): Directory containing original coronary and volume data.
        output_dir (Path): Directory to save projection results.
        proj_size (tuple[int, int]): Size of the projection images.
        alpha_betas (list[tuple[float, float]] | None): usage: `--alpha-betas 30 45 --alpha-betas 1.2 4.7`.
        num_random (int | None): Generate N random (alpha,beta) pairs for testing.
        workers (int): Number of workers to use for processing.
        devices (list[int]): CUDA device ids to use; workers are split evenly across them.
        vis (bool): Whether to generate visualization images.
        dde (float): Detector to world origin distance (mm, default: 400).
        dso (float): Source to world origin distance (mm, default: 1400).
        det_spacing (float): Detector pixel spacing (mm/pixel, default: 0.3).
    """
    
    if alpha_betas is not None:
        angles = ProjectionAngles.from_alpha_beta_list(alpha_betas)
    elif num_random is not None:
        angles = ProjectionAngles.from_random(num_random)
    else:
        raise ValueError("Must provide one of alpha_betas or num_random.")

    
    from .projection import process_resampled_directory
    process_resampled_directory(
        resample_coronary_dir=resample_coronary_dir,
        original_data_dir=original_data_dir,
        output_dir=output_dir,
        proj_size=proj_size,
        angles=angles,
        num_workers=workers,
        devices=devices,
        vis=vis,
        dde=dde, dso=dso, det_spacing=det_spacing,
    )


@app.command(name="project-once")
def project_once(
    coronary_path: Path,
    volume_path: Path,
    output_dir: Path,
    ref_dicom_or_json: Path | None = None,
    proj_size: tuple[int, int] | None = None,
    alpha_betas: list[tuple[float, float]] | None = None,
    dde: float | None = None,
    dso: float | None = None,
    det_spacing: float | None = None,
    vis: bool = False,
):
    """
    Generate projection for a single case with explicit file paths.
    Parameters can be provided from a reference DICOM/JSON file, overridden via CLI args, or specified directly.
    
    When --ref-dicom-or-json is given, the command reads proj_size, alpha/beta, dde/dso and det_spacing
    from the DICOM/JSON metadata. Any additional CLI parameters (--proj-size, --alpha-betas, --dde, etc.)
    will override the values read from the reference file.

    Args:
        coronary_path (Path): Path to the resampled coronary NIfTI file (e.g. .../Diseased_17_lca.nii.gz).
        volume_path (Path): Path to the original CT volume NIfTI file.
        output_dir (Path): Output directory for the projection result.
        ref_dicom_or_json (Path | None): Reference DICOM (.dcm) or JSON file containing projection parameters.
        proj_size (tuple[int, int] | None): Override projection image size (rows, cols).
        alpha_betas (list[tuple[float, float]] | None): Override angle(s) in degrees, e.g. `--alpha-betas 39.2 0.1`.
        dde (float | None): Override detector-to-origin distance (mm).
        dso (float | None): Override source-to-origin distance (mm).
        det_spacing (float | None): Override detector pixel spacing (mm/pixel).
        vis (bool): Whether to generate visualization images.
    """
    import torch
    import numpy as np

    from .projection import _project_one_case_inner, parse_name_type

    # --- 1. Parse case info from coronary_path ---
    case_name, branch_type = parse_name_type(coronary_path)

    # --- 2. Resolve projection parameters ---
    # Start with defaults
    final_proj_size = DEFAULT_PROJ_SIZE
    final_alpha_betas: list[tuple[float, float]] | None = None
    final_dde = DEFAULT_DDE
    final_dso = DEFAULT_DSO
    final_det_spacing = DEFAULT_DET_SPACING

    # Read from ref file if provided
    if ref_dicom_or_json is not None:
        ref_info = _parse_ref_dicom_or_json(ref_dicom_or_json)
        if "proj_size" in ref_info and proj_size is None:
            final_proj_size = ref_info["proj_size"]
        if "alpha" in ref_info and "beta" in ref_info and alpha_betas is None:
            final_alpha_betas = [(ref_info["alpha"], ref_info["beta"])]
        if "dde" in ref_info and dde is None:
            final_dde = ref_info["dde"]
        if "dso" in ref_info and dso is None:
            final_dso = ref_info["dso"]
        if "det_spacing" in ref_info and det_spacing is None:
            final_det_spacing = ref_info["det_spacing"]

    # CLI overrides
    if proj_size is not None:
        final_proj_size = proj_size
    if alpha_betas is not None:
        final_alpha_betas = alpha_betas
    if dde is not None:
        final_dde = dde
    if dso is not None:
        final_dso = dso
    if det_spacing is not None:
        final_det_spacing = det_spacing

    if final_alpha_betas is None:
        raise ValueError(
            "Angles must be specified either via --ref-dicom-or-json, "
            "--alpha-betas, or a combination of both."
        )

    angles = ProjectionAngles.from_alpha_beta_list(final_alpha_betas)

    # --- 3. Run projection ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Projecting case {case_name} ({branch_type}) with parameters:")
    print(f"  proj_size: {final_proj_size}")
    print(f"  angles (deg): {[(np.rad2deg(a), np.rad2deg(b)) for a, b in angles.alpha_beta]}")
    print(f"  dde: {final_dde}, dso: {final_dso}, det_spacing: {final_det_spacing}")
    print(f"  output_dir: {output_dir}")
    
    _project_one_case_inner(
        resampled_coronary_file=coronary_path,
        ori_volume_file=volume_path,
        case_name=case_name,
        branch_type=branch_type,
        angles=angles,
        proj_size=final_proj_size,
        output_dir=output_dir,
        device=device,
        vis=vis,
        dde=final_dde,
        dso=final_dso,
        det_spacing=final_det_spacing,
    )
    
    print("done")
