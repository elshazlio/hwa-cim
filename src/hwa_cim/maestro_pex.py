"""
Maestro / VIVA PEX waveform path: deterministic /OA_Charge calibration (not Monte Carlo).

See AgDR-0004. Ignores Read_Out_* probes because the inference testbench does not engage read mode.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from hwa_cim.config import DEFAULT_CALIBRATION_YAML, load_mac_calibration
from hwa_cim.utils_io import save_json

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = _REPO_ROOT / "config" / "maestro_pex.yaml"
DEFAULT_PEX_CALIBRATION = _REPO_ROOT / "config" / "calibration_pex.yaml"
_MIN_NOPEX_V_FOR_GAIN = 1e-6
_REASONABLE_GAIN = (0.25, 4.0)

READ_OUT_PATTERN = re.compile(r"read[_\s-]*out", re.IGNORECASE)


@dataclass(frozen=True)
class HardwareProfileInfo:
    """GUI / metrics metadata for a hardware profile mode."""

    mode: str
    badge: str
    banner: str
    profile_signal: Optional[str] = None
    profile_is_statistical: bool = False
    profile_warning: str = ""
    phase_label: str = ""
    profile_display_name: str = ""
    profile_kind: str = ""
    profile_is_foundry_certified: bool = False
    sigma_source: str = ""


HARDWARE_PROFILES: dict[str, HardwareProfileInfo] = {
    "synthetic": HardwareProfileInfo(
        mode="synthetic",
        badge="Synthetic",
        banner="Simulation-only synthetic noise. No Cadence profile is being used.",
        profile_is_statistical=False,
        profile_warning="AFM-style synthetic weight noise only",
    ),
    "maestro_pex": HardwareProfileInfo(
        mode="maestro_pex",
        badge="PEX calibrated",
        banner="Deterministic PEX calibration from OA_Charge. Not statistical Monte Carlo noise.",
        profile_signal="/OA_Charge",
        profile_is_statistical=False,
        profile_warning="Deterministic PEX calibration only; not Monte Carlo",
    ),
    "pex_corner_proxy": HardwareProfileInfo(
        mode="pex_corner_proxy",
        badge="Corner proxy",
        banner="Corner-derived proxy profile. Useful for stress testing, not MC mismatch statistics.",
        profile_signal="/OA_Charge",
        profile_is_statistical=False,
        profile_warning="Corner-derived sigma proxy; not Monte Carlo",
    ),
    "monte_carlo_csv": HardwareProfileInfo(
        mode="monte_carlo_csv",
        badge="Monte Carlo",
        banner="Statistical Monte Carlo profile. This is the thesis-grade Phase 5 CSV path.",
        profile_is_statistical=True,
        profile_warning="Statistical MC noise profile from Cadence export",
        phase_label="Phase 5",
        profile_display_name="Foundry / statistical Monte Carlo CSV",
        profile_kind="foundry_statistical_mc",
        profile_is_foundry_certified=True,
    ),
    "surrogate_mc": HardwareProfileInfo(
        mode="surrogate_mc",
        badge="Phase 4.5 Surrogate MC",
        banner=(
            "Surrogate Monte Carlo with user-defined Gaussian parametric variation. "
            "Not UMC-certified Monte Carlo; summaries are artifact/plot inputs until "
            "code-indexed profiles exist."
        ),
        profile_signal="/OA_Charge",
        profile_is_statistical=True,
        profile_warning=(
            "Surrogate MC with user-defined Gaussian parametric variation; "
            "not UMC-certified Monte Carlo"
        ),
        phase_label="Phase 4.5",
        profile_display_name=(
            "Surrogate Monte Carlo with user-defined Gaussian parametric variation"
        ),
        profile_kind="user_defined_gaussian_parametric_surrogate_mc",
        profile_is_foundry_certified=False,
        sigma_source="corner_delta_div_3",
    ),
}


def hardware_profile_metrics_extra(mode: str) -> dict[str, Any]:
    """Fields to merge into HWA ``metrics.json``."""
    info = HARDWARE_PROFILES.get(mode, HARDWARE_PROFILES["synthetic"])
    extra: dict[str, Any] = {
        "hardware_profile_mode": info.mode,
        "hardware_profile_badge": info.badge,
        "profile_signal": info.profile_signal,
        "profile_is_statistical": info.profile_is_statistical,
        "profile_warning": info.profile_warning,
    }
    if info.phase_label:
        extra["phase_label"] = info.phase_label
    if info.profile_display_name:
        extra["profile_display_name"] = info.profile_display_name
    if info.profile_kind:
        extra["profile_kind"] = info.profile_kind
    extra["profile_is_foundry_certified"] = info.profile_is_foundry_certified
    if info.sigma_source:
        extra["sigma_source"] = info.sigma_source
    return extra


def _normalize_signal(signal: str) -> str:
    s = signal.strip()
    if not s.startswith("/"):
        s = "/" + s
    return s


def _signal_key(signal: str) -> str:
    """Match token inside VIVA column headers (e.g. OA_Charge)."""
    return _normalize_signal(signal).lstrip("/")


def find_signal_xy_columns(columns: list[str], signal: str) -> tuple[str, str]:
    """
    Return (x_col, y_col) for a VIVA signal.

    Skips Read_Out_* columns. Requires headers containing ``{signal_key}`` and ending in `` X`` / `` Y``.
    """
    key = _signal_key(signal)
    x_col: Optional[str] = None
    y_col: Optional[str] = None
    for col in columns:
        if READ_OUT_PATTERN.search(col):
            continue
        if key not in col:
            continue
        if col.rstrip().endswith(" X"):
            x_col = col
        elif col.rstrip().endswith(" Y"):
            y_col = col
    if x_col is None or y_col is None:
        raise ValueError(
            f"Could not find X/Y columns for signal {signal!r} (key={key!r}); "
            f"got columns sample: {columns[:6]}..."
        )
    return x_col, y_col


def load_viva_waveform(csv_path: Path, signal: str) -> tuple[np.ndarray, np.ndarray]:
    """Load time (seconds) and value arrays for ``signal`` from a VIVA export."""
    df = pd.read_csv(csv_path)
    x_col, y_col = find_signal_xy_columns(list(df.columns), signal)
    x = df[x_col].astype(float).to_numpy()
    y = df[y_col].astype(float).to_numpy()
    return x, y


def sample_at_time_ns(
    time_s: np.ndarray,
    values: np.ndarray,
    sample_time_ns: float,
) -> float:
    """Linear interpolation at ``sample_time_ns`` (X axis in CSV is seconds)."""
    t_s = float(sample_time_ns) * 1e-9
    if time_s.size == 0:
        raise ValueError("empty waveform")
    if t_s <= time_s[0]:
        return float(values[0])
    if t_s >= time_s[-1]:
        return float(values[-1])
    return float(np.interp(t_s, time_s, values))


def waveform_delta_metrics(
    time_s: np.ndarray,
    pex_v: np.ndarray,
    nopex_v: np.ndarray,
) -> dict[str, float]:
    """Max and RMS of PEX − no-PEX over the shared time axis."""
    if pex_v.shape != nopex_v.shape:
        n = min(pex_v.size, nopex_v.size)
        pex_v = pex_v[:n]
        nopex_v = nopex_v[:n]
        time_s = time_s[:n]
    delta = pex_v - nopex_v
    return {
        "delta_max_v": float(np.max(np.abs(delta))),
        "delta_rms_v": float(np.sqrt(np.mean(delta**2))),
    }


@dataclass
class MaestroRunSpec:
    marker: str
    nopex_csv: Path
    pex_csv: Path
    signal: str = "/OA_Charge"
    sample_time_ns: float = 200.0
    input_code: Optional[int] = None
    weight_population: Optional[int] = None
    ideal_output_v: Optional[float] = None


@dataclass
class MaestroCornerSpec:
    marker: str
    corner: str
    pex_csv: Path
    signal: str = "/OA_Charge"
    sample_time_ns: float = 200.0


@dataclass
class MaestroManifest:
    runs: list[MaestroRunSpec] = field(default_factory=list)
    corner_runs: list[MaestroCornerSpec] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path, repo_root: Path | None = None) -> "MaestroManifest":
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("PyYAML required for Maestro manifest") from e

        root = repo_root or _REPO_ROOT
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        runs: list[MaestroRunSpec] = []
        for item in raw.get("runs") or []:
            runs.append(
                MaestroRunSpec(
                    marker=str(item["marker"]),
                    nopex_csv=_resolve_path(item["nopex_csv"], root),
                    pex_csv=_resolve_path(item["pex_csv"], root),
                    signal=str(item.get("signal", "/OA_Charge")),
                    sample_time_ns=float(item.get("sample_time_ns", 200.0)),
                    input_code=_optional_int(item.get("input_code")),
                    weight_population=_optional_int(item.get("weight_population")),
                    ideal_output_v=_optional_float(item.get("ideal_output_v")),
                )
            )
        corners: list[MaestroCornerSpec] = []
        for item in raw.get("corner_runs") or []:
            corners.append(
                MaestroCornerSpec(
                    marker=str(item["marker"]),
                    corner=str(item["corner"]),
                    pex_csv=_resolve_path(item["pex_csv"], root),
                    signal=str(item.get("signal", "/OA_Charge")),
                    sample_time_ns=float(item.get("sample_time_ns", 200.0)),
                )
            )
        return cls(runs=runs, corner_runs=corners)


def _resolve_path(p: str | Path, repo_root: Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _optional_int(v: Any) -> Optional[int]:
    if v is None or (isinstance(v, str) and v.lower() in ("null", "")):
        return None
    return int(v)


def _optional_float(v: Any) -> Optional[float]:
    if v is None or (isinstance(v, str) and v.lower() in ("null", "")):
        return None
    return float(v)


def process_run_pair(spec: MaestroRunSpec) -> dict[str, Any]:
    """Sample no-PEX and PEX waveforms; compute deltas at ``sample_time_ns``."""
    signal = _normalize_signal(spec.signal)
    tx_n, vy_n = load_viva_waveform(spec.nopex_csv, signal)
    tx_p, vy_p = load_viva_waveform(spec.pex_csv, signal)

    nopex_v = sample_at_time_ns(tx_n, vy_n, spec.sample_time_ns)
    pex_v = sample_at_time_ns(tx_p, vy_p, spec.sample_time_ns)
    delta_v = pex_v - nopex_v
    rel_gain = pex_v / nopex_v if abs(nopex_v) > _MIN_NOPEX_V_FOR_GAIN else float("nan")
    calibration_warnings: list[str] = []
    if abs(nopex_v) < _MIN_NOPEX_V_FOR_GAIN:
        calibration_warnings.append(
            f"|nopex_v|={abs(nopex_v):.3e} V at {spec.sample_time_ns} ns — "
            "pick a sample time away from zero-crossings for gain calibration."
        )
    elif not (_REASONABLE_GAIN[0] <= rel_gain <= _REASONABLE_GAIN[1]):
        calibration_warnings.append(
            f"relative_gain={rel_gain:.4g} outside {_REASONABLE_GAIN}; "
            "calibration scale may be clamped."
        )

    wf_metrics = waveform_delta_metrics(tx_n, vy_p, vy_n)

    row: dict[str, Any] = {
        "marker": spec.marker,
        "signal": signal,
        "sample_time_ns": spec.sample_time_ns,
        "nopex_csv": str(spec.nopex_csv),
        "pex_csv": str(spec.pex_csv),
        "nopex_v": nopex_v,
        "pex_v": pex_v,
        "delta_v": delta_v,
        "relative_gain": rel_gain,
        **wf_metrics,
        "calibration_warnings": calibration_warnings,
    }
    if spec.input_code is not None:
        row["input_code"] = spec.input_code
    if spec.weight_population is not None:
        row["weight_population"] = spec.weight_population
    if spec.ideal_output_v is not None:
        row["ideal_output_v"] = spec.ideal_output_v
        row["g_eff_measured"] = pex_v / spec.ideal_output_v if spec.ideal_output_v else float("nan")
    return row


def process_corner_runs(corners: list[MaestroCornerSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample /OA_Charge per corner; return (per-corner rows, proxy profile by marker).

    Proxy profile uses mean/std across corners — labeled corner-derived, not MC.
    """
    rows: list[dict[str, Any]] = []
    for spec in corners:
        signal = _normalize_signal(spec.signal)
        tx, vy = load_viva_waveform(spec.pex_csv, signal)
        v = sample_at_time_ns(tx, vy, spec.sample_time_ns)
        rows.append(
            {
                "marker": spec.marker,
                "corner": spec.corner,
                "signal": signal,
                "sample_time_ns": spec.sample_time_ns,
                "pex_csv": str(spec.pex_csv),
                "pex_v": v,
            }
        )
    corner_df = pd.DataFrame(rows)
    if corner_df.empty:
        return corner_df, corner_df

    proxy_rows: list[dict[str, Any]] = []
    for marker, grp in corner_df.groupby("marker"):
        vals = grp["pex_v"].astype(float)
        proxy_rows.append(
            {
                "marker": marker,
                "signal": grp["signal"].iloc[0],
                "sample_time_ns": grp["sample_time_ns"].iloc[0],
                "n_corners": len(grp),
                "pex_v_mean": float(vals.mean()),
                "pex_v_std": float(vals.std(ddof=0)) if len(vals) > 1 else 0.0,
                "sigma_proxy_v": float(vals.std(ddof=0)) if len(vals) > 1 else 0.0,
                "profile_kind": "corner_derived_sigma_proxy",
            }
        )
    return corner_df, pd.DataFrame(proxy_rows)


def build_calibration_yaml(
    summary_rows: list[dict[str, Any]],
    base_calibration: Path | None = None,
) -> dict[str, Any]:
    """
    Build calibration dict from Maestro summary.

    Scales schematic ``g_eff_sparse`` / ``g_eff_dense`` by mean ``relative_gain`` when finite.
    """
    mac = load_mac_calibration(base_calibration or DEFAULT_CALIBRATION_YAML)
    gains = [
        r["relative_gain"]
        for r in summary_rows
        if np.isfinite(r.get("relative_gain", float("nan")))
        and _REASONABLE_GAIN[0] <= r["relative_gain"] <= _REASONABLE_GAIN[1]
    ]
    scale = float(np.mean(gains)) if gains else 1.0
    cal_warnings = []
    for r in summary_rows:
        cal_warnings.extend(r.get("calibration_warnings") or [])
    if not gains:
        cal_warnings.append("No finite relative_gain in reasonable range; MAC scale left at 1.0.")

    payload: dict[str, Any] = {
        "source": "maestro_pex",
        "profile_warning": HARDWARE_PROFILES["maestro_pex"].profile_warning,
        "profile_is_statistical": False,
        "profile_signal": "/OA_Charge",
        "relative_gain_mean": scale,
        "mac": {
            "g_eff_sparse": mac.g_eff_sparse * scale,
            "g_eff_dense": mac.g_eff_dense * scale,
            "offset_dense_v": mac.offset_dense_v,
            "population_sparse_max": mac.population_sparse_max,
            "population_dense_min": mac.population_dense_min,
        },
        "ladder": {"integrated_operating_point": mac.integrated_operating_point},
        "maestro_pex_runs": summary_rows,
        "calibration_warnings": cal_warnings,
    }
    return payload


def write_calibration_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("PyYAML required to write calibration YAML") from e
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def write_maestro_pex_figures(
    manifest: MaestroManifest,
    summary_df: pd.DataFrame,
    figures_dir: Path,
) -> list[str]:
    """Save thesis-style PNGs for overlay, delta, and sampled summary."""
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    if manifest.runs:
        spec = manifest.runs[0]
        signal = _normalize_signal(spec.signal)
        tx_n, vy_n = load_viva_waveform(spec.nopex_csv, signal)
        tx_p, vy_p = load_viva_waveform(spec.pex_csv, signal)
        t_ns = tx_n * 1e9

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t_ns, vy_n, label="no-PEX", color="#94a3b8", linewidth=1.8)
        ax.plot(tx_p * 1e9, vy_p, label="PEX", color="#38bdf8", linewidth=1.8)
        ax.axvline(spec.sample_time_ns, color="#fbbf24", linestyle="--", linewidth=1.2, label="sample")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(f"{signal} — no-PEX vs PEX")
        ax.grid(True, alpha=0.3)
        ax.legend()
        p_overlay = figures_dir / "oa_charge_overlay.png"
        fig.tight_layout()
        fig.savefig(p_overlay, dpi=200)
        plt.close(fig)
        written.append(str(p_overlay))

        n = min(vy_n.size, vy_p.size, tx_n.size)
        delta = vy_p[:n] - vy_n[:n]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(tx_n[:n] * 1e9, delta, color="#38bdf8", linewidth=1.8)
        ax.axhline(0.0, color="#64748b", linewidth=0.8)
        ax.axvline(spec.sample_time_ns, color="#fbbf24", linestyle="--", linewidth=1.2, label="sample")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Δ voltage (V)")
        ax.set_title(f"{signal} — PEX minus no-PEX")
        ax.grid(True, alpha=0.3)
        ax.legend()
        p_delta = figures_dir / "oa_charge_delta.png"
        fig.tight_layout()
        fig.savefig(p_delta, dpi=200)
        plt.close(fig)
        written.append(str(p_delta))

    if not summary_df.empty and "nopex_v" in summary_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = summary_df["marker"].astype(str)
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, summary_df["nopex_v"], width=w, label="no-PEX", color="#94a3b8")
        ax.bar(x + w / 2, summary_df["pex_v"], width=w, label="PEX", color="#38bdf8")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Voltage (V)")
        ax.set_title("Sampled OA_Charge at manifest time")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        p_sum = figures_dir / "oa_charge_summary.png"
        fig.tight_layout()
        fig.savefig(p_sum, dpi=200)
        plt.close(fig)
        written.append(str(p_sum))

    return written


def export_corner_proxy_csv(proxy_df: pd.DataFrame, out_path: Path) -> None:
    """
    Minimal CSV for stress testing (not Phase 5 MC schema).

    Columns: marker, ideal_output, mean_output, sigma with corner spread semantics.
    """
    if proxy_df.empty:
        return
    rows = []
    for _, r in proxy_df.iterrows():
        mean_v = float(r["pex_v_mean"])
        sigma_v = float(r["sigma_proxy_v"])
        rows.append(
            {
                "marker": r["marker"],
                "input_code": 0,
                "ideal_output": mean_v,
                "mean_output": mean_v,
                "sigma": sigma_v,
                "profile_kind": r.get("profile_kind", "corner_derived_sigma_proxy"),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def run_maestro_pex(
    manifest_path: Path = DEFAULT_MANIFEST,
    out_dir: Path = Path("results/maestro_pex"),
    write_calibration: Path | None = None,
    write_figures: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Execute manifest: summary CSV, metrics JSON, optional calibration + corner exports."""
    root = repo_root or _REPO_ROOT
    manifest = MaestroManifest.load(manifest_path, repo_root=root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [process_run_pair(spec) for spec in manifest.runs]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "maestro_pex_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    corner_detail = pd.DataFrame()
    corner_proxy = pd.DataFrame()
    if manifest.corner_runs:
        corner_detail, corner_proxy = process_corner_runs(manifest.corner_runs)
        corner_detail.to_csv(out_dir / "maestro_pex_corners.csv", index=False)
        if not corner_proxy.empty:
            corner_proxy.to_csv(out_dir / "maestro_pex_corner_proxy.csv", index=False)
            export_corner_proxy_csv(corner_proxy, out_dir / "corner_proxy_noise_profile.csv")

    agg_delta_max = float(summary_df["delta_max_v"].max()) if not summary_df.empty else 0.0
    agg_delta_rms = float(summary_df["delta_rms_v"].mean()) if not summary_df.empty else 0.0

    metrics: dict[str, Any] = {
        "manifest": str(manifest_path.resolve()),
        "signal": "/OA_Charge",
        "sample_time_ns": manifest.runs[0].sample_time_ns if manifest.runs else None,
        "n_runs": len(manifest.runs),
        "n_corner_runs": len(manifest.corner_runs),
        "summary_csv": str(summary_path),
        "delta_max_v": agg_delta_max,
        "delta_rms_v": agg_delta_rms,
        "profile_is_statistical": False,
        "profile_warning": "Deterministic PEX calibration from Maestro/VIVA; not Monte Carlo",
        "read_out_signals_ignored": True,
        "warnings": [
            "Read_Out_* columns are ignored (inference testbench; read mode / sense amp inactive).",
            "This path does not replace Phase 5 --noise-mode csv Monte Carlo training.",
        ],
    }
    if summary_rows:
        metrics["summary_rows"] = summary_rows

    metrics_path = out_dir / "maestro_pex_metrics.json"
    save_json(metrics_path, metrics)

    cal_path_written: Optional[str] = None
    if write_calibration is not None and summary_rows:
        cal_payload = build_calibration_yaml(summary_rows)
        write_calibration_yaml(Path(write_calibration), cal_payload)
        cal_path_written = str(Path(write_calibration).resolve())
        metrics["calibration_yaml"] = cal_path_written

    figure_paths: list[str] = []
    if write_figures and manifest.runs:
        figure_paths = write_maestro_pex_figures(
            manifest, summary_df, out_dir / "figures"
        )
        metrics["figures_dir"] = str((out_dir / "figures").resolve())
        metrics["figure_paths"] = figure_paths

    return {
        "metrics": metrics,
        "summary_csv": str(summary_path),
        "metrics_json": str(metrics_path),
        "calibration_yaml": cal_path_written,
        "figure_paths": figure_paths,
        "corner_proxy_csv": str(out_dir / "corner_proxy_noise_profile.csv")
        if (out_dir / "corner_proxy_noise_profile.csv").is_file()
        else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Maestro/VIVA PEX OA_Charge calibration reports")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="YAML manifest (default: config/maestro_pex.yaml)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("results/maestro_pex"))
    ap.add_argument(
        "--write-calibration",
        type=Path,
        default=None,
        help="Write scaled MAC calibration YAML (e.g. config/calibration_pex.yaml)",
    )
    ap.add_argument(
        "--no-write-figures",
        action="store_true",
        help="Skip PNG exports under out-dir/figures/",
    )
    args = ap.parse_args()
    result = run_maestro_pex(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        write_calibration=args.write_calibration,
        write_figures=not args.no_write_figures,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
