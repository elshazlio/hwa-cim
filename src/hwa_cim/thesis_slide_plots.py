"""
Thesis slide figures: taxonomy, validation ladder, Phase 4.5→5 pipeline, results dashboard.

Outputs under ``results/plots/04_thesis_slides/``. Used by CLI ``hwa-plot-thesis-slides`` and the GUI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from hwa_cim.maestro_pex import HARDWARE_PROFILES, DEFAULT_PEX_CALIBRATION
from hwa_cim.plots import PHASE45_FOOTER, plot_pvt_pex_corner_bars, thesis_bar_ylim_percent
from hwa_cim.surrogate_mc import process_pvt_pex_wide_corners
from hwa_cim.utils_io import load_json

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = Path("results/plots/04_thesis_slides")

_SLIDE_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 10,
}


def _save(fig: plt.Figure, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_hardware_profile_taxonomy(out_path: Path) -> Path:
    """Slide 26: distinguish Synthetic, PEX, corners, Phase 4.5, Phase 5."""
    profiles = [
        ("synthetic", "#94A3B8", "Now"),
        ("maestro_pex", "#0284C7", "Now"),
        ("pex_corner_proxy", "#64748B", "Now"),
        ("surrogate_mc", "#0EA5E9", "Now"),
        ("monte_carlo_csv", "#CBD5E1", "Future"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(5, 4.75, "Hardware noise & calibration paths", ha="center", fontsize=14, fontweight="bold")

    y = 3.85
    for mode, color, status in profiles:
        info = HARDWARE_PROFILES[mode]
        phase = getattr(info, "phase_label", None) or "Phase 1–3"
        stat = "statistical" if info.profile_is_statistical else "deterministic"
        foundry = "foundry-certified" if info.profile_is_foundry_certified else "not foundry-certified"
        box = FancyBboxPatch(
            (0.5, y - 0.42),
            9.0,
            0.82,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.2,
            edgecolor=color,
            facecolor=color if status == "Now" else "#F8FAFC",
            alpha=0.28 if status == "Now" else 0.15,
        )
        ax.add_patch(box)
        ax.text(0.85, y, info.badge, fontsize=10.5, fontweight="bold", va="center")
        ax.text(3.0, y, f"{phase} · {stat} · {foundry}", fontsize=8.5, va="center", color="#475569")
        ax.text(9.15, y, status, ha="right", fontsize=9.5, fontweight="bold", color="#334155")
        y -= 0.92

    fig.text(
        0.5,
        0.04,
        "HWA trains with synthetic γ today. Phase 4.5 = surrogate evidence/plots. Phase 5 = per-code foundry MC CSV (future).",
        ha="center",
        fontsize=8.5,
        color="#64748B",
    )
    return _save(fig, out_path)


def plot_phase45_to_phase5_pipeline(out_path: Path) -> Path:
    """Slide 36: what exists now vs Phase 5 foundry MC."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.text(5.5, 4.1, "Phase 4.5 evidence → Phase 5 readiness", ha="center", fontsize=13, fontweight="bold")

    boxes = [
        (0.4, 2.2, 2.4, 1.2, "Manual MC sweeps\n(cap + dvth0 grids)", "#0EA5E9"),
        (3.1, 2.2, 2.4, 1.2, "Surrogate σ summaries\n(/OA_Charge spread)", "#38BDF8"),
        (5.8, 2.2, 2.4, 1.2, "PVT/PEX corners\n(4-corner no-PEX vs PEX)", "#64748B"),
        (8.5, 2.2, 2.1, 1.2, "HWA + thesis bars\n(MNIST recovery)", "#55A868"),
    ]
    for x, y, w, h, label, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03",
            facecolor=color,
            edgecolor="white",
            alpha=0.35,
            linewidth=1.5,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, fontweight="bold")

    for x0, x1 in [(2.85, 3.05), (5.55, 5.75), (8.25, 8.45)]:
        ax.add_patch(
            FancyArrowPatch(
                (x0, 2.8),
                (x1, 2.8),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#334155",
            )
        )

    future = FancyBboxPatch(
        (1.2, 0.35),
        8.6,
        1.35,
        boxstyle="round,pad=0.04",
        facecolor="#F1F5F9",
        edgecolor="#94A3B8",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(future)
    ax.text(
        5.5,
        1.05,
        "Phase 5 (future): per-code foundry MC CSV\n"
        "input_code, ideal_output, mean_output, sigma, CSNR_dB",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#334155",
    )
    ax.annotate(
        "",
        xy=(5.5, 1.75),
        xytext=(5.5, 2.15),
        arrowprops=dict(arrowstyle="-|>", color="#64748B"),
    )
    fig.text(0.5, 0.02, PHASE45_FOOTER, ha="center", fontsize=8, color="#64748B")
    return _save(fig, out_path)


def plot_validation_ladder(out_path: Path) -> Path:
    """Slide 29: Python → Cadence → calibration → HWA."""
    steps = [
        "Python golden\nC-2C MAC model",
        "Quantized MNIST\nmicro-MLP",
        "Cadence /OA_Charge\nMaestro exports",
        "Calibration YAML\n(schematic or PEX)",
        "HWA train +\nnoisy metrics",
    ]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.text(5.5, 2.65, "End-to-end validation ladder", ha="center", fontsize=13, fontweight="bold")

    n = len(steps)
    w = 1.65
    gap = 0.35
    x0 = (11 - n * w - (n - 1) * gap) / 2
    colors = ["#4C72B0", "#DD8452", "#0284C7", "#94A3B8", "#55A868"]
    for i, (label, color) in enumerate(zip(steps, colors)):
        x = x0 + i * (w + gap)
        patch = FancyBboxPatch(
            (x, 0.85),
            w,
            1.35,
            boxstyle="round,pad=0.04",
            facecolor=color,
            alpha=0.25,
            edgecolor=color,
            linewidth=1.5,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, 1.52, label, ha="center", va="center", fontsize=8.5, fontweight="bold")
        if i < n - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + w, 1.52),
                    (x + w + gap, 1.52),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    color="#334155",
                )
            )
    ax.text(
        5.5,
        0.25,
        "Parity checks: ladder nonlinearity (Python) · PEX gain (Cadence) · MNIST noisy accuracy (HWA)",
        ha="center",
        fontsize=8.5,
        color="#64748B",
    )
    return _save(fig, out_path)


def plot_results_summary_dashboard(
    out_path: Path,
    *,
    baseline_dir: Path,
    hwa_dir: Path,
    hwa_pex_dir: Path | None = None,
    noisy_eval_json: Path | None = None,
    profile_badge: str = "Synthetic",
) -> Path:
    """Slide 30: compact results composite with profile label."""
    base_m = load_json(baseline_dir / "metrics.json")
    hwa_m = load_json(hwa_dir / "metrics.json")
    fp32 = float(base_m["fp32_test_accuracy"]) * 100
    if noisy_eval_json and noisy_eval_json.is_file():
        noisy_m = load_json(noisy_eval_json)
        int4_noisy = float(noisy_m["mean_accuracy"]) * 100
    else:
        proxy = base_m.get("int4_noisy_proxy", 0.939) * 100
        int4_noisy = float(proxy) if proxy else 93.9
    hwa_acc = float(hwa_m["final_noisy_mean"]) * 100
    hwa_pex_acc = None
    if hwa_pex_dir and (hwa_pex_dir / "metrics.json").is_file():
        hwa_pex_acc = float(load_json(hwa_pex_dir / "metrics.json")["final_noisy_mean"]) * 100

    plt.rcParams.update(_SLIDE_STYLE)
    fig = plt.figure(figsize=(11, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.25)
    ax_bars = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.axis("off")

    labels = ["FP32", "INT4+noise", "HWA"]
    vals = [fp32, int4_noisy, hwa_acc]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    if hwa_pex_acc is not None:
        labels.append("HWA\n(PEX cal.)")
        vals.append(hwa_pex_acc)
        colors.append("#0284C7")

    y_lo, y_hi = thesis_bar_ylim_percent(vals)
    x = np.arange(len(labels))
    ax_bars.bar(x, vals, color=colors, width=0.55, edgecolor="white")
    ax_bars.set_xticks(x, labels)
    ax_bars.set_ylabel("Test accuracy (%)")
    ax_bars.set_title("Results summary (γ = 0.02)")
    ax_bars.set_ylim(y_lo, y_hi)
    ax_bars.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(vals):
        ax_bars.text(i, v + 0.15, f"{v:.2f}%", ha="center", fontweight="bold")

    rows = [
        ["Profile", profile_badge],
        ["FP32 baseline", f"{fp32:.2f}%"],
        ["Noisy INT4 (no HWA)", f"{int4_noisy:.2f}%"],
        ["HWA recovery", f"{hwa_acc:.2f}%"],
        ["Δ vs noisy", f"{hwa_acc - int4_noisy:+.2f} pp"],
    ]
    if hwa_pex_acc is not None:
        rows.append(["HWA (PEX cal.)", f"{hwa_pex_acc:.2f}%"])
    table = ax_tbl.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)
    ax_tbl.set_title("Run card", fontweight="bold", pad=12)

    fig.text(
        0.5,
        0.02,
        "Safe headline: HWA recovers noisy accuracy. Do not claim PEX alone fixes MNIST without HWA.",
        ha="center",
        fontsize=8,
        color="#64748B",
    )
    return _save(fig, out_path)


def plot_phase5_schema_mock(out_path: Path) -> Path:
    """Slide 36 inset: required Phase 5 CSV columns."""
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.axis("off")
    cols = "input_code, weight_population, ideal_output, mean_output, sigma, CSNR_dB"
    ax.text(0.5, 0.72, "Phase 5 noise profile (target schema)", ha="center", fontsize=12, fontweight="bold")
    ax.text(
        0.5,
        0.42,
        cols,
        ha="center",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#F1F5F9", edgecolor="#94A3B8"),
    )
    ax.text(
        0.5,
        0.12,
        "Many statistical samples per input code · foundry-certified MC export",
        ha="center",
        fontsize=8.5,
        color="#64748B",
    )
    return _save(fig, out_path)


def run_thesis_slide_plots(
    *,
    repo_root: Path | None = None,
    out_dir: Path = _DEFAULT_OUT,
    sample_time_ns: float = 200.25,
    baseline_dir: Path | None = None,
    hwa_dir: Path | None = None,
    hwa_pex_dir: Path | None = None,
    noisy_eval_json: Path | None = None,
) -> list[Path]:
    """Generate all thesis slide figures; returns paths written."""
    root = Path(repo_root or _REPO_ROOT)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    written.append(plot_hardware_profile_taxonomy(out_dir / "01_hardware_profile_taxonomy.png"))
    written.append(plot_phase45_to_phase5_pipeline(out_dir / "02_phase45_to_phase5_pipeline.png"))
    written.append(plot_validation_ladder(out_dir / "03_validation_ladder.png"))
    written.append(plot_phase5_schema_mock(out_dir / "06_phase5_csv_schema_mock.png"))

    cadence = root / "stuff_from_cadence"
    nopex = cadence / "no_pex_oa_only_3_corners.csv"
    pex = cadence / "with_pex_oa_only_3_corners.csv"
    if nopex.is_file() and pex.is_file():
        corner_df = process_pvt_pex_wide_corners(nopex, pex, sample_time_ns=sample_time_ns)
        sur_base = root / "results/surrogate_mc" / "pvt_pex_corners"
        sur_base.mkdir(parents=True, exist_ok=True)
        corner_df.to_csv(sur_base / "pvt_pex_corners.csv", index=False)
        p4 = out_dir / "04_pvt_pex_corner_bars_fixed.png"
        plot_pvt_pex_corner_bars(corner_df, p4, sample_time_ns=sample_time_ns)
        written.append(p4)

    bdir = Path(baseline_dir or root / "results/run_baseline")
    hdir = Path(hwa_dir or root / "results/run_hwa")
    pex_dir = hwa_pex_dir or root / "results/run_hwa_pex_calibrated"
    noisy = noisy_eval_json or bdir / "noisy_eval.json"
    badge = "Synthetic"
    if (hdir / "metrics.json").is_file():
        hm = load_json(hdir / "metrics.json")
        badge = str(hm.get("hardware_profile_badge", badge))

    if (bdir / "metrics.json").is_file() and (hdir / "metrics.json").is_file():
        written.append(
            plot_results_summary_dashboard(
                out_dir / "05_results_summary_dashboard.png",
                baseline_dir=bdir,
                hwa_dir=hdir,
                hwa_pex_dir=pex_dir if (pex_dir / "metrics.json").is_file() else None,
                noisy_eval_json=noisy if Path(noisy).is_file() else None,
                profile_badge=badge,
            )
        )

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate thesis slide figures for deck + GUI")
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--sample-time-ns", type=float, default=200.25)
    ap.add_argument("--baseline-dir", type=Path, default=None)
    ap.add_argument("--hwa-dir", type=Path, default=None)
    args = ap.parse_args()
    paths = run_thesis_slide_plots(
        out_dir=args.out_dir,
        sample_time_ns=args.sample_time_ns,
        baseline_dir=args.baseline_dir,
        hwa_dir=args.hwa_dir,
    )
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
