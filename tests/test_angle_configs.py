"""Tests for angle configuration loading and parsing."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from sparse_view_dataset.projection import load_angle_csv, make_angle_configs
from sparse_view_dataset.cli import _parse_alpha_beta_pairs


class TestLoadAngleCsv:
    def test_with_header(self):
        """CSV with a header line should be parsed correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("alpha,beta\n")
            for i in range(5):
                f.write(f"{i * 10},{i * 5}\n")  # degrees
            csv_path = Path(f.name)

        try:
            alphas, betas = load_angle_csv(csv_path)
            assert len(alphas) == 5
            assert np.allclose(alphas, np.deg2rad([0, 10, 20, 30, 40]))
            assert np.allclose(betas, np.deg2rad([0, 5, 10, 15, 20]))
        finally:
            csv_path.unlink()

    def test_without_header(self):
        """CSV without a header should also be parsed correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            for i in range(3):
                f.write(f"{i * 30},{i * 10}\n")  # degrees
            csv_path = Path(f.name)

        try:
            alphas, betas = load_angle_csv(csv_path)
            assert len(alphas) == 3
            assert np.allclose(alphas, np.deg2rad([0, 30, 60]))
            assert np.allclose(betas, np.deg2rad([0, 10, 20]))
        finally:
            csv_path.unlink()

    def test_single_row(self):
        """Single-row CSV should produce length-1 arrays."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("30,15\n")  # degrees
            csv_path = Path(f.name)

        try:
            alphas, betas = load_angle_csv(csv_path)
            assert len(alphas) == 1
            assert np.isclose(alphas[0], np.deg2rad(30))
            assert np.isclose(betas[0], np.deg2rad(15))
        finally:
            csv_path.unlink()


class TestMakeAngleConfigs:
    def test_from_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("0,0\n10,5\n")  # degrees
            csv_path = Path(f.name)

        try:
            configs = make_angle_configs(csv_paths=[csv_path])
            assert len(configs) == 1
            assert "name" in configs[0]
            assert "alphas" in configs[0]
            assert "betas" in configs[0]
            assert len(configs[0]["alphas"]) == 2
        finally:
            csv_path.unlink()

    def test_from_direct_pairs(self):
        alphas = np.array([0.0, 0.2, 0.4])
        betas = np.array([0.0, 0.1, 0.0])
        configs = make_angle_configs(alpha_betas_list=[(alphas, betas)])
        assert len(configs) == 1
        assert "custom" in configs[0]["name"]
        assert np.allclose(configs[0]["alphas"], alphas)
        assert np.allclose(configs[0]["betas"], betas)

    def test_from_random(self):
        configs = make_angle_configs(num_random=16)
        assert len(configs) == 1
        assert "random" in configs[0]["name"]
        assert len(configs[0]["alphas"]) == 16
        assert len(configs[0]["betas"]) == 16

    def test_multiple_configs(self):
        alphas_a = np.array([0.0, 0.5])
        betas_a = np.array([0.0, 0.1])
        alphas_b = np.array([0.2, 0.7])
        betas_b = np.array([0.05, 0.15])
        configs = make_angle_configs(
            alpha_betas_list=[(alphas_a, betas_a), (alphas_b, betas_b)],
            names=["config_a", "config_b"],
        )
        assert len(configs) == 2
        assert configs[0]["name"] == "config_a"
        assert configs[1]["name"] == "config_b"

    def test_no_input_raises(self):
        with pytest.raises(ValueError, match="至少需要提供一种角度配置"):
            make_angle_configs()


class TestParseAlphaBetaPairs:
    def test_basic(self):
        a, b = _parse_alpha_beta_pairs("0,0;30,6;60,-3")
        assert np.allclose(a, np.deg2rad([0, 30, 60]))
        assert np.allclose(b, np.deg2rad([0, 6, -3]))

    def test_single_pair(self):
        a, b = _parse_alpha_beta_pairs("20,10")
        assert np.allclose(a, np.deg2rad([20]))
        assert np.allclose(b, np.deg2rad([10]))

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="无效的角度对"):
            _parse_alpha_beta_pairs("0.5")  # missing beta

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="未解析到任何角度"):
            _parse_alpha_beta_pairs("")
