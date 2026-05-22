"""Tests for Maestro/VIVA PEX OA_Charge path (AgDR-0004)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hwa_cim.maestro_pex import (
    MaestroRunSpec,
    build_calibration_yaml,
    find_signal_xy_columns,
    load_viva_waveform,
    process_run_pair,
    sample_at_time_ns,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_NOPEX = _FIXTURES / "maestro_nopex_tiny.csv"
_PEX = _FIXTURES / "maestro_pex_tiny.csv"


def test_find_signal_columns_ignores_read_out() -> None:
    cols = list(
        __import__("pandas").read_csv(_NOPEX, nrows=0).columns
    )
    x_col, y_col = find_signal_xy_columns(cols, "/OA_Charge")
    assert "OA_Charge" in x_col and x_col.endswith(" X")
    assert "Read_Out" not in x_col and "Read_Out" not in y_col


def test_sample_at_time_ns_interpolation() -> None:
    t = np.array([0.0, 100e-9, 200e-9, 300e-9])
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert sample_at_time_ns(t, v, 200.0) == pytest.approx(3.0)
    assert sample_at_time_ns(t, v, 150.0) == pytest.approx(2.5)


def test_process_run_pair_delta_and_gain() -> None:
    spec = MaestroRunSpec(
        marker="tiny",
        nopex_csv=_NOPEX,
        pex_csv=_PEX,
        signal="/OA_Charge",
        sample_time_ns=200.0,
    )
    row = process_run_pair(spec)
    assert row["nopex_v"] == pytest.approx(3.0)
    assert row["pex_v"] == pytest.approx(3.3)
    assert row["delta_v"] == pytest.approx(0.3)
    assert row["relative_gain"] == pytest.approx(1.1)


def test_read_out_not_used_for_sampling() -> None:
    """Read_Out Y differs wildly; OA_Charge path must not pick those columns."""
    tx, vy = load_viva_waveform(_NOPEX, "/OA_Charge")
    assert vy[2] == pytest.approx(3.0)


def test_build_calibration_yaml_scales_gains() -> None:
    row = process_run_pair(
        MaestroRunSpec(marker="t", nopex_csv=_NOPEX, pex_csv=_PEX, sample_time_ns=200.0)
    )
    payload = build_calibration_yaml([row])
    assert payload["relative_gain_mean"] == pytest.approx(1.1)
    assert payload["mac"]["g_eff_sparse"] > 0.6


def test_maestro_pex_cli_dry_run(tmp_path: Path) -> None:
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        f"""
runs:
  - marker: tiny
    nopex_csv: {_NOPEX}
    pex_csv: {_PEX}
    signal: /OA_Charge
    sample_time_ns: 200.0
""",
        encoding="utf-8",
    )
    from hwa_cim.maestro_pex import run_maestro_pex

    out = tmp_path / "out"
    result = run_maestro_pex(manifest_path=manifest, out_dir=out, repo_root=tmp_path)
    assert (out / "maestro_pex_summary.csv").is_file()
    assert (out / "maestro_pex_metrics.json").is_file()
    assert result["metrics"]["profile_is_statistical"] is False
