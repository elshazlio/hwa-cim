"""Wizard job thunks and HWA profile mapping (mirrors Run page 3a)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hwa_cim.maestro_pex import DEFAULT_PEX_CALIBRATION

from hwa_gui.wizard.state import DEFAULT_PATHS

DEFAULT_SURROGATE_SUMMARY = "results/surrogate_mc/cap_sweep/surrogate_mc_summary.csv"


def baseline_artifacts_present(repo: Path, baseline_ckpt: str | None = None) -> bool:
    ck = repo / (baseline_ckpt or DEFAULT_PATHS["baseline_ckpt"])
    return ck.is_file()


def hwa_artifacts_present(repo: Path, hwa_ckpt: str | None = None) -> bool:
    ck = repo / (hwa_ckpt or DEFAULT_PATHS["hwa_ckpt"])
    metrics = ck.parent / "metrics.json"
    return ck.is_file() and metrics.is_file()


def noisy_eval_present(repo: Path, noisy_json: str | None = None) -> bool:
    p = repo / (noisy_json or DEFAULT_PATHS["noisy_json"])
    return p.is_file()


def load_metrics_json(repo: Path, rel_path: str) -> dict[str, Any] | None:
    p = repo / rel_path
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_noisy_eval(repo: Path, rel_path: str) -> dict[str, Any] | None:
    return load_metrics_json(repo, rel_path)


def resolve_hardware_profile(
    profile_mode: str,
    *,
    repo: Path,
    cal_yaml: str = "config/calibration.yaml",
    noise_profile: Path | None = None,
    surrogate_summary: Path | None = None,
) -> tuple[str, Path | None, Path | None, Path | None, str | None]:
    """
    Map GUI hardware profile → train/eval knobs.

    Returns (noise_mode, calibration_path, noise_profile_path, surrogate_summary_path, error).
    """
    np_path = noise_profile
    sur_path = surrogate_summary
    cal_path: Path | None

    if profile_mode == "monte_carlo_csv":
        return "csv", None, None, None, (
            "Phase 5 Monte Carlo CSV is future work in the guided demo. "
            "Use Advanced lab → Run or Hardware profiles when you have a CSV."
        )

    if profile_mode == "pex_corner_proxy":
        if not np_path:
            np_path = repo / "results/maestro_pex/corner_proxy_noise_profile.csv"
        if not np_path.is_file():
            return "csv", None, None, None, (
                "PEX corner proxy needs corner_proxy_noise_profile.csv — "
                "generate corners in Advanced lab → Hardware profiles first."
            )
        return "csv", repo / cal_yaml if cal_yaml else None, np_path, None, None

    if profile_mode == "maestro_pex":
        cal = repo / DEFAULT_PEX_CALIBRATION
        if not cal.is_file() and cal_yaml:
            cal = repo / cal_yaml
        return "synthetic", cal if cal.is_file() else None, None, None, None

    if profile_mode == "cadence_surrogate_stress":
        if not sur_path:
            sur_path = repo / DEFAULT_SURROGATE_SUMMARY
        if not sur_path.is_file():
            return "cadence_stress", None, None, None, (
                f"Cadence-informed stress needs Phase 4.5 summary at {sur_path.relative_to(repo)} — "
                "run **Hardware profiles → Phase 4.5 Surrogate MC** or **Refresh Phase 4.5 plots**."
            )
        return "cadence_stress", None, None, sur_path, None

    if profile_mode in ("synthetic", "surrogate_mc"):
        cal = repo / cal_yaml if cal_yaml else None
        return "synthetic", cal if cal and cal.is_file() else None, None, None, None

    return "synthetic", None, None, None, f"Unknown profile mode: {profile_mode}"


def resolve_hwa_train(
    profile_mode: str,
    *,
    repo: Path,
    data_dir: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    gamma: float,
    alpha: float,
    seed: int,
    device: str,
    eval_noisy_seeds: int,
    cal_yaml: str = "config/calibration.yaml",
    noise_profile: Path | None = None,
    surrogate_summary: Path | None = None,
) -> tuple[str, Path | None, Path | None, str | None]:
    """
    Returns (noise_mode, calibration_path, noise_profile_path, error_message).
    error_message is set when the run must not start.
    """
    noise_mode, cal_path, np_path, sur_path, err = resolve_hardware_profile(
        profile_mode,
        repo=repo,
        cal_yaml=cal_yaml,
        noise_profile=noise_profile,
        surrogate_summary=surrogate_summary,
    )
    if err:
        return noise_mode, cal_path, np_path, err
    return noise_mode, cal_path, np_path, None


def resolve_noisy_eval(
    profile_mode: str,
    *,
    repo: Path,
    surrogate_summary: Path | None = None,
) -> tuple[str, Path | None, str | None]:
    """Returns (noise_mode, surrogate_summary_path, error_message)."""
    noise_mode, _, _, sur_path, err = resolve_hardware_profile(
        profile_mode,
        repo=repo,
        surrogate_summary=surrogate_summary,
    )
    if err and profile_mode == "cadence_surrogate_stress":
        return noise_mode, sur_path, err
    if profile_mode == "cadence_surrogate_stress":
        return "cadence_stress", sur_path, None
    return "synthetic", None, None


def make_baseline_thunk(
    *,
    repo: Path,
    data_dir: Path,
    out_dir: Path,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> Callable[[], None]:
    def _go() -> None:
        from hwa_cim.train_baseline import run_baseline

        run_baseline(
            data_dir=data_dir,
            out_dir=out_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            device=device,
        )

    return _go


def make_noisy_eval_thunk(
    *,
    repo: Path,
    checkpoint: Path,
    data_dir: Path,
    gamma: float = 0.02,
    seeds: int = 10,
    device: str = "cpu",
    out_json: Path | None = None,
    profile_mode: str = "synthetic",
    surrogate_summary: Path | None = None,
) -> tuple[Callable[[], None] | None, str | None]:
    noise_mode, sur_path, err = resolve_noisy_eval(
        profile_mode, repo=repo, surrogate_summary=surrogate_summary
    )
    if err:
        return None, err

    def _go() -> None:
        from hwa_cim.evaluate import run_noisy_eval

        run_noisy_eval(
            checkpoint=checkpoint,
            data_dir=data_dir,
            gamma=gamma,
            seeds=seeds,
            device=device,
            out=out_json,
            noise_mode=noise_mode,
            surrogate_summary=sur_path,
        )

    return _go, None


def make_hwa_train_thunk(
    *,
    repo: Path,
    data_dir: Path,
    out_dir: Path,
    profile_mode: str,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    gamma: float = 0.02,
    alpha: float = 3.0,
    seed: int = 42,
    device: str = "cpu",
    eval_noisy_seeds: int = 10,
    cal_yaml: str = "config/calibration.yaml",
    noise_profile: Path | None = None,
    surrogate_summary: Path | None = None,
) -> tuple[Callable[[], None] | None, str | None]:
    noise_mode, cal_path, np_path, err = resolve_hwa_train(
        profile_mode,
        repo=repo,
        data_dir=data_dir,
        out_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        alpha=alpha,
        seed=seed,
        device=device,
        eval_noisy_seeds=eval_noisy_seeds,
        cal_yaml=cal_yaml,
        noise_profile=noise_profile,
        surrogate_summary=surrogate_summary,
    )
    if err:
        return None, err

    _, _, sur_path, _ = resolve_hardware_profile(
        profile_mode,
        repo=repo,
        cal_yaml=cal_yaml,
        noise_profile=noise_profile,
        surrogate_summary=surrogate_summary,
    )

    def _go() -> None:
        from hwa_cim.train_hwa import run_hwa_train

        run_hwa_train(
            data_dir=data_dir,
            out_dir=out_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            gamma=gamma,
            alpha=alpha,
            seed=seed,
            device=device,
            noise_mode=noise_mode,
            noise_profile=np_path,
            surrogate_summary=sur_path,
            calibration_yaml=cal_path,
            eval_noisy_seeds=eval_noisy_seeds,
            hardware_profile_mode=profile_mode,
        )

    return _go, None
