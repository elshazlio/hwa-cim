"""Tests for Cadence-informed surrogate stress (AgDR-0007)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hwa_cim.cadence_stress import load_cadence_stress_profile
from hwa_cim.layers import NoisyQuantLinear
from hwa_cim.maestro_pex import HARDWARE_PROFILES, hardware_profile_metrics_extra
from hwa_cim.noise import NoiseProfileCSV, additive_relative_output_noise
from hwa_gui.wizard.actions import resolve_hardware_profile, resolve_hwa_train

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_PHASE45 = _FIXTURES / "surrogate_phase45_summary.csv"
_PHASE5 = _FIXTURES / "noise_profile_example.csv"


def test_load_cadence_stress_sigma_rel() -> None:
    prof = load_cadence_stress_profile(_PHASE45)
    expected = 0.009715 / 0.755016
    assert prof.surrogate_sigma_rel == pytest.approx(expected, rel=1e-4)
    assert prof.phase_label == "Phase 4.5"
    assert prof.profile_kind == "cadence_informed_surrogate_stress"


def test_load_rejects_phase5_csv() -> None:
    with pytest.raises(ValueError, match="Phase 5"):
        load_cadence_stress_profile(_PHASE5)


def test_load_rejects_malformed_summary(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_cadence_stress_profile(bad)


def test_additive_relative_output_noise_changes_tensor() -> None:
    torch.manual_seed(0)
    y = torch.randn(4, 8)
    y2 = additive_relative_output_noise(y, 0.01)
    assert y2.shape == y.shape
    assert not torch.allclose(y, y2)


def test_cadence_stress_layer_only_in_training() -> None:
    layer = NoisyQuantLinear(16, 8, noise_mode="cadence_stress", surrogate_sigma_rel=0.02)
    x = torch.randn(2, 16)
    layer.eval()
    with torch.no_grad():
        y_eval = layer(x)
    layer.train()
    torch.manual_seed(0)
    y_train = layer(x)
    assert not torch.allclose(y_eval, y_train)


def test_hardware_profile_cadence_metadata() -> None:
    info = HARDWARE_PROFILES["cadence_surrogate_stress"]
    assert info.phase_label == "Phase 4.5"
    assert info.profile_is_foundry_certified is False
    assert "Cadence" in info.banner
    extra = hardware_profile_metrics_extra("cadence_surrogate_stress")
    assert extra["hardware_profile_mode"] == "cadence_surrogate_stress"


def test_resolve_cadence_surrogate_stress(tmp_path: Path) -> None:
    summary = tmp_path / "surrogate_mc_summary.csv"
    summary.write_text(_PHASE45.read_text(encoding="utf-8"), encoding="utf-8")
    noise_mode, cal, np_path, sur, err = resolve_hardware_profile(
        "cadence_surrogate_stress",
        repo=tmp_path,
        surrogate_summary=summary,
    )
    assert err is None
    assert noise_mode == "cadence_stress"
    assert sur == summary
    assert np_path is None


def test_resolve_hwa_train_cadence_missing_summary(tmp_path: Path) -> None:
    _, _, _, err = resolve_hwa_train(
        "cadence_surrogate_stress",
        repo=tmp_path,
        data_dir=tmp_path,
        out_dir=tmp_path / "out",
        epochs=1,
        batch_size=32,
        lr=1e-3,
        gamma=0.02,
        alpha=3.0,
        seed=0,
        device="cpu",
        eval_noisy_seeds=2,
    )
    assert err is not None


def test_run_noisy_eval_cadence_stress_writes_provenance(tmp_path: Path) -> None:
    pytest.importorskip("torchvision")
    summary = tmp_path / "summary.csv"
    summary.write_text(_PHASE45.read_text(encoding="utf-8"), encoding="utf-8")
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": _tiny_baseline_state()}, ckpt)

    from hwa_cim.evaluate import run_noisy_eval

    out = tmp_path / "noisy.json"
    payload = run_noisy_eval(
        checkpoint=ckpt,
        data_dir=Path("data"),
        seeds=2,
        device="cpu",
        out=out,
        noise_mode="cadence_stress",
        surrogate_summary=summary,
    )
    assert payload["noise_mode"] == "cadence_stress"
    assert "surrogate_sigma_rel" in payload
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["phase_label"] == "Phase 4.5"
    assert saved["profile_is_foundry_certified"] is False


def _tiny_baseline_state() -> dict:
    from hwa_cim.models import MicroMLP

    m = MicroMLP()
    return m.state_dict()


def test_phase45_still_rejected_by_noise_profile_csv() -> None:
    with pytest.raises(ValueError, match="missing"):
        NoiseProfileCSV.load(_PHASE45)
