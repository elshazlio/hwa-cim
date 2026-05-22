"""Tests for Phase 4.5 surrogate Monte Carlo path (AgDR-0005)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hwa_cim.maestro_pex import HARDWARE_PROFILES, hardware_profile_metrics_extra
from hwa_cim.noise import NoiseProfileCSV
from hwa_cim.surrogate_mc import (
    PHASE_LABEL,
    PROFILE_DISPLAY_NAME,
    PROFILE_KIND,
    find_all_signal_xy_pairs,
    parse_viva_sweep_header,
    process_pvt_pex_wide_corners,
    run_surrogate_mc,
    sample_wide_viva_csv,
    summarize_sweep_points,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_CAP_TINY = _FIXTURES / "surrogate_cap_tiny.csv"
_PHASE45_SUMMARY = _FIXTURES / "surrogate_phase45_summary.csv"
_NOPEX_CORNERS = _FIXTURES / "pvt_corners_tiny_nopex.csv"
_PEX_CORNERS = _FIXTURES / "pvt_corners_tiny_pex.csv"

_REPO = Path(__file__).resolve().parents[1]
_CAP_FULL = _REPO / "stuff_from_cadence" / "manual_mc_2_var_cap_1.csv"
_DVTH_FULL = _REPO / "stuff_from_cadence" / "manual_mc_4_var_1.csv"


def test_parse_viva_sweep_header() -> None:
    col = '"/OA_Charge (umc_mc_d_c1_vp=-0.067,umc_mc_d_cox_vp=0) X"'
    parsed = parse_viva_sweep_header(col)
    assert parsed["umc_mc_d_c1_vp"] == pytest.approx(-0.067)
    assert parsed["umc_mc_d_cox_vp"] == pytest.approx(0.0)


def test_find_all_signal_xy_pairs_cap_tiny() -> None:
    import pandas as pd

    cols = list(pd.read_csv(_CAP_TINY, nrows=0).columns)
    pairs = find_all_signal_xy_pairs(cols, "/OA_Charge")
    assert len(pairs) == 2


def test_sample_wide_csv_cap_tiny() -> None:
    df = sample_wide_viva_csv(
        _CAP_TINY,
        "/OA_Charge",
        200.25,
        variable_group="mom_cap_grid",
        marker="tiny",
    )
    assert len(df) == 2
    spread = float(df["sampled_v"].max() - df["sampled_v"].min())
    assert spread == pytest.approx(0.024, rel=0.05)


@pytest.mark.skipif(not _CAP_FULL.is_file(), reason="Cadence export not in workspace")
def test_sample_wide_csv_cap_full_spread() -> None:
    df = sample_wide_viva_csv(
        _CAP_FULL,
        "/OA_Charge",
        200.25,
        variable_group="mom_cap_grid",
    )
    assert len(df) == 9
    spread = float(df["sampled_v"].max() - df["sampled_v"].min())
    assert spread == pytest.approx(0.023806, rel=0.02)


@pytest.mark.skipif(not _DVTH_FULL.is_file(), reason="Cadence export not in workspace")
def test_sample_wide_csv_dvth0_full_spread() -> None:
    df = sample_wide_viva_csv(
        _DVTH_FULL,
        "/OA_Charge",
        200.25,
        variable_group="dvth0_grid",
    )
    assert len(df) == 81
    spread = float(df["sampled_v"].max() - df["sampled_v"].min())
    assert spread == pytest.approx(0.001162, rel=0.05)


def test_summarize_sweep_profile_kind() -> None:
    df = sample_wide_viva_csv(_CAP_TINY, "/OA_Charge", 200.25, marker="tiny")
    summary = summarize_sweep_points(
        df,
        variable_group="mom_cap_grid",
        marker="tiny",
        signal="/OA_Charge",
        sample_time_ns=200.25,
    )
    assert summary["phase_label"] == PHASE_LABEL
    assert summary["profile_kind"] == PROFILE_KIND
    assert summary["profile_display_name"] == PROFILE_DISPLAY_NAME
    assert summary["profile_is_foundry_certified"] is False


def test_phase45_summary_rejected_by_noise_profile_csv() -> None:
    with pytest.raises(ValueError, match="missing"):
        NoiseProfileCSV.load(_PHASE45_SUMMARY)


def test_hardware_profiles_surrogate_mc_metadata() -> None:
    info = HARDWARE_PROFILES["surrogate_mc"]
    assert info.phase_label == "Phase 4.5"
    assert "Gaussian" in info.profile_display_name
    assert info.profile_is_foundry_certified is False
    extra = hardware_profile_metrics_extra("surrogate_mc")
    assert extra["phase_label"] == "Phase 4.5"
    assert extra["profile_kind"] == PROFILE_KIND
    assert extra["profile_is_foundry_certified"] is False


def test_pvt_corner_wide_csv_sampling() -> None:
    corner_df = process_pvt_pex_wide_corners(
        _NOPEX_CORNERS,
        _PEX_CORNERS,
        sample_time_ns=200.25,
    )
    assert len(corner_df) == 2
    nom = corner_df[corner_df["corner"] == "nominal"].iloc[0]
    assert nom["nopex_v"] == pytest.approx(0.827, rel=0.01)
    assert nom["pex_v"] == pytest.approx(0.755, rel=0.01)
    assert nom["relative_gain"] == pytest.approx(0.755 / 0.827, rel=0.02)


def test_surrogate_mc_cli_writes_outputs(tmp_path: Path) -> None:
    metrics = run_surrogate_mc(
        _CAP_TINY,
        tmp_path / "out",
        sample_time_ns=200.25,
        variable_group="mom_cap_grid",
        marker="cli_test",
    )
    assert (tmp_path / "out" / "surrogate_mc_points.csv").is_file()
    assert (tmp_path / "out" / "surrogate_mc_summary.csv").is_file()
    assert (tmp_path / "out" / "surrogate_mc_metrics.json").is_file()
    assert metrics["phase_label"] == PHASE_LABEL
    mjson = json.loads((tmp_path / "out" / "surrogate_mc_metrics.json").read_text())
    assert mjson["profile_is_foundry_certified"] is False
