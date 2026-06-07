"""
Phase 4.5 — Surrogate Monte Carlo with user-defined Gaussian parametric variation.

Parses wide VIVA exports from manual PDK delta-parameter sweeps (not foundry MC).
See AgDR-0005 and background_info/Surrogate_MC_Phase5_Readiness_Report.md.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from hwa_cim.maestro_pex import (
    READ_OUT_PATTERN,
    _normalize_signal,
    _signal_key,
    sample_at_time_ns,
)
from hwa_cim.utils_io import save_json

_REPO_ROOT = Path(__file__).resolve().parents[2]

PHASE_LABEL = "Phase 4.5"
PROFILE_DISPLAY_NAME = (
    "Surrogate Monte Carlo with user-defined Gaussian parametric variation"
)
PROFILE_KIND = "user_defined_gaussian_parametric_surrogate_mc"
PROFILE_WARNING = (
    "Surrogate MC with user-defined Gaussian parametric variation; "
    "not UMC-certified Monte Carlo"
)
SIGMA_SOURCE = "corner_delta_div_3"

# Parenthesized sweep vars: umc_mc_d_c1_vp=0.067
_SWEEP_VAR_PATTERN = re.compile(
    r"(umc_mc_[a-zA-Z0-9_]+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

# Corner labels from modelFiles / row-2 shorthand
_CORNER_LABEL_MAP = {
    "nom": "nominal",
    "ff": "ff",
    "ss": "ss",
    "fnsp": "fnsp",
}


@dataclass(frozen=True)
class SweepColumnPair:
    """One waveform column pair from a wide VIVA export."""

    x_col: str
    y_col: str
    sweep_vars: dict[str, float]
    header_label: str


def parse_viva_sweep_header(col: str) -> dict[str, float]:
    """Extract ``umc_mc_*`` sweep assignments from a VIVA column header."""
    return {m.group(1): float(m.group(2)) for m in _SWEEP_VAR_PATTERN.finditer(col)}


def _header_paren_segment(col: str) -> str:
    """Text inside first ``(...)`` in a VIVA column name."""
    start = col.find("(")
    end = col.rfind(")")
    if start < 0 or end <= start:
        return ""
    return col[start + 1 : end]


def _row_hint_from_pair(df: pd.DataFrame, pair: SweepColumnPair) -> Optional[str]:
    """Corner shorthand from VIVA row 0 under the X column, if non-numeric."""
    if pair.x_col not in df.columns or len(df) == 0:
        return None
    val = df.iloc[0][pair.x_col]
    if pd.isna(val):
        return None
    key = str(val).strip().lower()
    if key.replace(".", "", 1).isdigit():
        return None
    return key


def infer_corner_label(header_segment: str, row2_value: Optional[str] = None) -> str:
    """Map VIVA header or row-2 shorthand to corner name."""
    if row2_value is not None:
        key = str(row2_value).strip().lower()
        if key in _CORNER_LABEL_MAP:
            return _CORNER_LABEL_MAP[key]
        if key and not key.replace(".", "", 1).isdigit():
            return key
    seg = header_segment.lower()
    if "modelfiles=nom" in seg or ",nom," in seg:
        return "nominal"
    if ":ff_" in seg or "ff_sp" in seg:
        return "ff"
    if ":ss_" in seg or "ss_sp" in seg or "ss_65" in seg:
        return "ss"
    if "fnsp" in seg:
        return "fnsp"
    return header_segment[:48] if header_segment else "unknown"


def find_all_signal_xy_pairs(columns: list[str], signal: str) -> list[SweepColumnPair]:
    """
    Return all (X, Y) pairs for ``signal`` in a wide VIVA CSV header row.

    Skips ``Read_Out_*``. Pairs X/Y columns that share the same header stem.
    """
    key = _signal_key(signal)
    x_by_stem: dict[str, str] = {}
    y_by_stem: dict[str, str] = {}
    meta_by_stem: dict[str, dict[str, float]] = {}
    pair_order: list[str] = []

    for col in columns:
        if READ_OUT_PATTERN.search(col):
            continue
        if key not in col:
            continue
        stem = col.rstrip()
        if stem.endswith(" Y"):
            base = stem[:-2]
            y_by_stem[base] = col
        elif stem.endswith(" X"):
            base = stem[:-2]
            x_by_stem[base] = col
            meta_by_stem[base] = parse_viva_sweep_header(col)
            if base not in pair_order:
                pair_order.append(base)

    pairs: list[SweepColumnPair] = []
    for base in pair_order:
        if base not in y_by_stem or base not in x_by_stem:
            continue
        pairs.append(
            SweepColumnPair(
                x_col=x_by_stem[base],
                y_col=y_by_stem[base],
                sweep_vars=meta_by_stem.get(base, {}),
                header_label=base,
            )
        )
    if not pairs:
        raise ValueError(
            f"No X/Y pairs for signal {signal!r} (key={key!r}); "
            f"columns sample: {columns[:4]}..."
        )
    return pairs


def _coerce_numeric_waveform(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-numeric rows (e.g. metadata ``nom`` row) and return float arrays."""
    sub = df[[x_col, y_col]].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        raise ValueError(f"No numeric samples for columns {x_col!r} / {y_col!r}")
    return (
        sub[x_col].astype(float).to_numpy(),
        sub[y_col].astype(float).to_numpy(),
    )


def sample_wide_viva_csv(
    csv_path: Path,
    signal: str,
    sample_time_ns: float = 200.25,
    *,
    variable_group: str = "sweep",
    marker: Optional[str] = None,
) -> pd.DataFrame:
    """
    Sample ``signal`` at ``sample_time_ns`` for every sweep column pair in a wide CSV.

    Returns a DataFrame with sweep variables plus ``sampled_v``.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    signal = _normalize_signal(signal)
    marker = marker or csv_path.stem
    pairs = find_all_signal_xy_pairs(list(df.columns), signal)

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        tx, vy = _coerce_numeric_waveform(df, pair.x_col, pair.y_col)
        v = sample_at_time_ns(tx, vy, sample_time_ns)
        row: dict[str, Any] = {
            "marker": marker,
            "signal": signal,
            "sample_time_ns": sample_time_ns,
            "variable_group": variable_group,
            "sampled_v": v,
            "header_label": pair.header_label,
        }
        row.update(pair.sweep_vars)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_sweep_points(
    points_df: pd.DataFrame,
    *,
    variable_group: str,
    marker: str,
    signal: str,
    sample_time_ns: float,
) -> dict[str, Any]:
    """Aggregate sampled voltages into a Phase 4.5 summary record."""
    vals = points_df["sampled_v"].astype(float)
    vmin = float(vals.min())
    vmax = float(vals.max())
    return {
        "marker": marker,
        "signal": signal,
        "sample_time_ns": sample_time_ns,
        "variable_group": variable_group,
        "n_points": int(len(vals)),
        "mean_output": float(vals.mean()),
        "sigma_output": float(vals.std(ddof=0)) if len(vals) > 1 else 0.0,
        "min_output": vmin,
        "max_output": vmax,
        "spread_output": vmax - vmin,
        "phase_label": PHASE_LABEL,
        "profile_display_name": PROFILE_DISPLAY_NAME,
        "profile_kind": PROFILE_KIND,
        "profile_is_statistical": True,
        "profile_is_foundry_certified": False,
        "sigma_source": SIGMA_SOURCE,
        "profile_warning": PROFILE_WARNING,
    }


def export_phase45_summary_csv(summaries: list[dict[str, Any]], out_path: Path) -> None:
    """Write Phase 4.5 summary table (not Phase 5 ``NoiseProfileCSV`` schema)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out_path, index=False)


def process_pvt_pex_wide_corners(
    nopex_csv: Path,
    pex_csv: Path,
    signal: str = "/OA_Charge",
    sample_time_ns: float = 200.25,
) -> pd.DataFrame:
    """
    Align no-PEX and PEX wide 4-corner VIVA exports by corner label.

    Returns per-corner ``nopex_v``, ``pex_v``, ``delta_v``, ``relative_gain``.
    """
    signal = _normalize_signal(signal)
    nopex_df = pd.read_csv(nopex_csv)
    pex_df = pd.read_csv(pex_csv)
    nopex_cols = list(nopex_df.columns)
    pex_cols = list(pex_df.columns)

    def _corner_table(raw: pd.DataFrame, columns: list[str]) -> dict[str, float]:
        pairs = find_all_signal_xy_pairs(columns, signal)
        out: dict[str, float] = {}
        for pair in pairs:
            tx, vy = _coerce_numeric_waveform(raw, pair.x_col, pair.y_col)
            v = sample_at_time_ns(tx, vy, sample_time_ns)
            seg = _header_paren_segment(pair.x_col)
            label = infer_corner_label(seg, _row_hint_from_pair(raw, pair))
            out[label] = v
        return out

    nopex_v = _corner_table(nopex_df, nopex_cols)
    pex_v = _corner_table(pex_df, pex_cols)

    rows: list[dict[str, Any]] = []
    for corner in sorted(set(nopex_v) | set(pex_v)):
        nv = nopex_v.get(corner, float("nan"))
        pv = pex_v.get(corner, float("nan"))
        rel = pv / nv if abs(nv) > 1e-12 else float("nan")
        rows.append(
            {
                "corner": corner,
                "signal": signal,
                "sample_time_ns": sample_time_ns,
                "nopex_v": nv,
                "pex_v": pv,
                "delta_v": pv - nv,
                "relative_gain": rel,
                "profile_kind": "pvt_pex_corner_deterministic",
                "phase_label": PHASE_LABEL,
                "profile_warning": (
                    "Deterministic PVT/post-layout corner comparison; not Monte Carlo sigma"
                ),
            }
        )
    return pd.DataFrame(rows)


def run_surrogate_mc(
    csv_path: Path,
    out_dir: Path,
    *,
    signal: str = "/OA_Charge",
    sample_time_ns: float = 200.25,
    variable_group: str = "sweep",
    marker: Optional[str] = None,
    write_pvt_corners: bool = False,
    nopex_corner_csv: Optional[Path] = None,
    pex_corner_csv: Optional[Path] = None,
) -> dict[str, Any]:
    """Parse one surrogate sweep CSV; write per-point + Phase 4.5 summary artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = marker or Path(csv_path).stem

    points_df = sample_wide_viva_csv(
        csv_path,
        signal,
        sample_time_ns,
        variable_group=variable_group,
        marker=marker,
    )
    points_path = out_dir / "surrogate_mc_points.csv"
    points_df.to_csv(points_path, index=False)

    summary = summarize_sweep_points(
        points_df,
        variable_group=variable_group,
        marker=marker,
        signal=_normalize_signal(signal),
        sample_time_ns=sample_time_ns,
    )
    summary_path = out_dir / "surrogate_mc_summary.csv"
    export_phase45_summary_csv([summary], summary_path)

    metrics: dict[str, Any] = {
        "phase_label": PHASE_LABEL,
        "profile_display_name": PROFILE_DISPLAY_NAME,
        "profile_kind": PROFILE_KIND,
        "profile_is_statistical": True,
        "profile_is_foundry_certified": False,
        "sigma_source": SIGMA_SOURCE,
        "profile_warning": PROFILE_WARNING,
        "source_csv": str(Path(csv_path).resolve()),
        "signal": _normalize_signal(signal),
        "sample_time_ns": sample_time_ns,
        "variable_group": variable_group,
        "marker": marker,
        "n_points": summary["n_points"],
        "spread_output": summary["spread_output"],
        "points_csv": str(points_path.resolve()),
        "summary_csv": str(summary_path.resolve()),
        "summary": summary,
    }

    if write_pvt_corners and nopex_corner_csv and pex_corner_csv:
        corner_df = process_pvt_pex_wide_corners(
            nopex_corner_csv,
            pex_corner_csv,
            signal=signal,
            sample_time_ns=sample_time_ns,
        )
        corner_path = out_dir / "pvt_pex_corners.csv"
        corner_df.to_csv(corner_path, index=False)
        metrics["pvt_pex_corners_csv"] = str(corner_path.resolve())
        metrics["pvt_pex_corners"] = corner_df.to_dict(orient="records")

    metrics_path = out_dir / "surrogate_mc_metrics.json"
    save_json(metrics_path, metrics)
    metrics["metrics_json"] = str(metrics_path.resolve())
    return metrics


def run_default_sweeps(
    repo_root: Path | None = None,
    out_base: Path = Path("results/surrogate_mc"),
    sample_time_ns: float = 200.25,
) -> dict[str, Any]:
    """Process bundled ``stuff_from_cadence`` surrogate + corner exports."""
    root = repo_root or _REPO_ROOT
    cadence = root / "stuff_from_cadence"
    results: dict[str, Any] = {}

    cap_csv = cadence / "manual_mc_2_var_cap_1.csv"
    dvth_csv = cadence / "manual_mc_4_var_1.csv"
    if cap_csv.is_file():
        results["cap_sweep"] = run_surrogate_mc(
            cap_csv,
            out_base / "cap_sweep",
            sample_time_ns=sample_time_ns,
            variable_group="mom_cap_grid",
            marker="mom_cap_grid",
        )
    if dvth_csv.is_file():
        results["dvth0_sweep"] = run_surrogate_mc(
            dvth_csv,
            out_base / "dvth0_sweep",
            sample_time_ns=sample_time_ns,
            variable_group="dvth0_grid",
            marker="dvth0_grid",
        )

    nopex = cadence / "no_pex_oa_only_3_corners.csv"
    pex = cadence / "with_pex_oa_only_3_corners.csv"
    if nopex.is_file() and pex.is_file():
        corner_dir = out_base / "pvt_pex_corners"
        corner_dir.mkdir(parents=True, exist_ok=True)
        corner_df = process_pvt_pex_wide_corners(
            nopex, pex, sample_time_ns=sample_time_ns
        )
        corner_path = corner_dir / "pvt_pex_corners.csv"
        corner_df.to_csv(corner_path, index=False)
        save_json(
            corner_dir / "pvt_pex_corners_metrics.json",
            {
                "phase_label": PHASE_LABEL,
                "profile_warning": (
                    "Deterministic PVT/post-layout corner comparison; not Monte Carlo sigma"
                ),
                "corners": corner_df.to_dict(orient="records"),
            },
        )
        results["pvt_pex_corners"] = {"csv": str(corner_path)}

    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Phase 4.5 — Surrogate Monte Carlo (user-defined Gaussian parametric variation): "
            "parse wide VIVA sweep CSVs"
        )
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Wide VIVA surrogate sweep CSV (omit with --all-defaults)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("results/surrogate_mc/sweep"))
    ap.add_argument("--signal", default="/OA_Charge")
    ap.add_argument("--sample-time-ns", type=float, default=200.25)
    ap.add_argument("--variable-group", default="sweep")
    ap.add_argument("--marker", default=None)
    ap.add_argument(
        "--all-defaults",
        action="store_true",
        help="Process bundled stuff_from_cadence cap, dvth0, and PVT/PEX corner CSVs",
    )
    ap.add_argument(
        "--pvt-corners",
        action="store_true",
        help="Also write PVT/PEX corner table (requires --nopex-corners and --pex-corners)",
    )
    ap.add_argument("--nopex-corners", type=Path, default=None)
    ap.add_argument("--pex-corners", type=Path, default=None)
    args = ap.parse_args()

    if args.all_defaults:
        out = run_default_sweeps(sample_time_ns=args.sample_time_ns)
        print(json.dumps({k: v.get("summary_csv", v) for k, v in out.items()}, indent=2))
        return

    if args.csv is None:
        ap.error("Provide --csv or use --all-defaults")

    metrics = run_surrogate_mc(
        args.csv,
        args.out_dir,
        signal=args.signal,
        sample_time_ns=args.sample_time_ns,
        variable_group=args.variable_group,
        marker=args.marker,
        write_pvt_corners=args.pvt_corners,
        nopex_corner_csv=args.nopex_corners,
        pex_corner_csv=args.pex_corners,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
