from pathlib import Path

from sparse_view_dataset.projection import _append_failed_case_log, _project_one_case_safe


def test_append_failed_case_log_creates_and_appends(tmp_path: Path):
    case_path = tmp_path / "case_01.nii.gz"

    _append_failed_case_log(tmp_path, case_path, "traceback one")
    _append_failed_case_log(tmp_path, case_path, "traceback two")

    log_path = tmp_path / "failed_cases.log"
    content = log_path.read_text(encoding="utf-8")

    assert content.count(f"==== {case_path} ====") == 2
    assert "traceback one" in content
    assert "traceback two" in content


def test_project_one_case_safe_returns_failure_tuple(monkeypatch, tmp_path: Path):
    case_path = tmp_path / "case_02_lca.nii.gz"

    def raise_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("sparse_view_dataset.projection.project_one_case", raise_error)

    succeeded, returned_path, tb = _project_one_case_safe(
        resampled_coronary_file=case_path,
        original_data_dir=tmp_path,
        num_projs=(32,),
        proj_size=(512, 512),
        output_dir=tmp_path,
        vis_num_projs=None,
    )

    assert succeeded is False
    assert returned_path == case_path
    assert "RuntimeError: boom" in tb