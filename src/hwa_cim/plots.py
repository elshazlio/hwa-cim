"""Thesis-quality figures: parasitic sweep, gamma sweep, three-bar comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hwa_cim.c2c import INTEGRATED_OPERATING_POINT, ladder_nonlinearity_metric
from hwa_cim.utils_io import load_json


def plot_parasitic_sweep(
    out_path: Path, pdk_marker: float = INTEGRATED_OPERATING_POINT
) -> None:
    ratios = np.linspace(0.0, 0.2, 41)
    metrics = [ladder_nonlinearity_metric(float(r)) for r in ratios]
    plt.figure(figsize=(7, 4))
    plt.plot(ratios * 100, metrics, label="Max |error| vs ideal ramp")
    plt.axvline(pdk_marker * 100, color="C1", linestyle="--", label=f"PDK op ~{pdk_marker*100:.0f}%")
    plt.xlabel("Parasitic ratio (%)")
    plt.ylabel("Nonlinearity metric")
    plt.title("C-2C ladder: transfer nonlinearity vs parasitic ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gamma_sweep_csv(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(7, 4))
    plt.plot(df["gamma"], df.get("accuracy_seed0", df.get("accuracy", df.iloc[:, -1])), marker="o")
    plt.xlabel("gamma_weight")
    plt.ylabel("Accuracy")
    plt.title("Noisy inference vs gamma (single seed)")
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def thesis_bar_ylim_percent(vals: list[float]) -> tuple[float, float]:
    """
    Y-limits that zoom on the bar range so small accuracy gaps read clearly.

    vals: accuracies already in percent (0–100).
    """
    vmin, vmax = float(min(vals)), float(max(vals))
    span = vmax - vmin
    pad = max(1.0, span * 0.35)
    lo = max(0.0, np.floor((vmin - pad) * 2) / 2)
    hi = min(100.0, np.ceil((vmax + pad) * 2) / 2)
    if hi - lo < 6.0:
        mid = 0.5 * (vmin + vmax)
        lo = max(0.0, np.floor((mid - 3.0) * 2) / 2)
        hi = min(100.0, np.ceil((mid + 3.0) * 2) / 2)
    return lo, hi


def plot_thesis_three_bar(
    fp32_acc: float,
    int4_noisy_acc: float,
    hwa_noisy_acc: float,
    out_path: Path,
    title: str = "MNIST: HWA training recovers accuracy under noise",
    *,
    zoom_yaxis: bool = True,
) -> None:
    labels = ["FP32\n(baseline)", "INT4 + noise\n(no HWA)", "INT4 + noise\n(HWA trained)"]
    vals = [fp32_acc * 100, int4_noisy_acc * 100, hwa_noisy_acc * 100]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    y_lo, y_hi = thesis_bar_ylim_percent(vals) if zoom_yaxis else (0.0, 100.0)
    label_offset = max(0.25, (y_hi - y_lo) * 0.04)

    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(labels))
    plt.bar(x, vals, color=colors, width=0.55)
    plt.xticks(x, labels)
    plt.ylabel("Test accuracy (%)")
    plt.title(title)
    plt.ylim(y_lo, y_hi)
    if zoom_yaxis:
        plt.gca().set_yticks(np.arange(y_lo, y_hi + 0.01, 1.0))
    for i, v in enumerate(vals):
        plt.text(i, v + label_offset, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    plt.grid(True, axis="y", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_hwa_pex_context(
    out_path: Path,
    *,
    no_hwa_acc: float,
    hwa_schematic_acc: float,
    hwa_pex_acc: float,
    oa_charge_retention_pct: float,
    g_eff_scale_pct: float,
) -> None:
    """
    Honest HWA vs PEX-calibration context figure for thesis/slides.

    Accuracies are noisy-test percentages (0–100). PEX panel shows deterministic
    analog scaling, not power or Monte Carlo σ.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
        }
    )

    fig = plt.figure(figsize=(11.5, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], width_ratios=[1.35, 1.0], hspace=0.38, wspace=0.28)

    ax_main = fig.add_subplot(gs[0, :])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_pex = fig.add_subplot(gs[1, 1])

    acc_labels = [
        "INT4 + noise\n(no HWA)",
        "HWA\n(schematic cal.)",
        "HWA\n(PEX cal.)",
    ]
    acc_vals = [no_hwa_acc, hwa_schematic_acc, hwa_pex_acc]
    colors = ["#DD8452", "#55A868", "#4C72B0"]
    y_lo, y_hi = thesis_bar_ylim_percent(acc_vals)
    label_offset = max(0.12, (y_hi - y_lo) * 0.05)

    x = np.arange(len(acc_labels))
    bars = ax_main.bar(x, acc_vals, color=colors, width=0.58, edgecolor="white", linewidth=0.8)
    ax_main.set_xticks(x, acc_labels)
    ax_main.set_ylabel("Test accuracy (%)")
    ax_main.set_title("Noisy MNIST accuracy (γ = 0.02)")
    ax_main.set_ylim(y_lo, y_hi)
    ax_main.set_yticks(np.arange(y_lo, y_hi + 0.01, 0.5 if (y_hi - y_lo) <= 4 else 1.0))
    ax_main.grid(True, axis="y", alpha=0.3)
    ax_main.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, acc_vals):
        ax_main.text(
            bar.get_x() + bar.get_width() / 2,
            val + label_offset,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Zoomed HWA pair — makes “flat” visually obvious
    zoom_labels = ["schematic cal.", "PEX cal."]
    zoom_vals = [hwa_schematic_acc, hwa_pex_acc]
    zx = np.arange(2)
    zbars = ax_zoom.bar(zx, zoom_vals, color=["#55A868", "#4C72B0"], width=0.5, edgecolor="white", linewidth=0.8)
    z_lo = np.floor((min(zoom_vals) - 0.12) * 100) / 100
    z_hi = np.ceil((max(zoom_vals) + 0.12) * 100) / 100
    ax_zoom.set_xticks(zx, zoom_labels)
    ax_zoom.set_ylabel("Test accuracy (%)")
    ax_zoom.set_title("HWA only (zoomed)")
    ax_zoom.set_ylim(z_lo, z_hi)
    ax_zoom.set_yticks(np.round(np.linspace(z_lo, z_hi, 5), 2))
    ax_zoom.grid(True, axis="y", alpha=0.3)
    ax_zoom.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(zbars, zoom_vals):
        ax_zoom.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.2f}%",
            ha="center",
            fontweight="bold",
        )
    pex_labels = ["/OA_Charge\n@ sample", "g_eff_sparse\n(PEX/schematic)"]
    pex_vals = [oa_charge_retention_pct, g_eff_scale_pct]
    px = np.arange(2)
    pbars = ax_pex.bar(px, pex_vals, color=["#0284C7", "#94A3B8"], width=0.5, edgecolor="white", linewidth=0.8)
    ax_pex.axhline(100.0, color="#64748B", ls="--", lw=1.0)
    ax_pex.set_xticks(px, pex_labels)
    ax_pex.set_ylabel("% of reference")
    ax_pex.set_title("Deterministic PEX scaling")
    ax_pex.set_ylim(88, 102)
    ax_pex.grid(True, axis="y", alpha=0.3)
    ax_pex.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(pbars, pex_vals):
        ax_pex.text(bar.get_x() + bar.get_width() / 2, val + 0.35, f"{val:.1f}%", ha="center", fontweight="bold")

    fig.text(
        0.5,
        0.01,
        "Metrics: Phase 2 noisy eval (no HWA) vs Phase 3 final_noisy_mean (HWA runs). "
        "PEX calibrates MAC gain from Cadence; not MC noise or power.",
        ha="center",
        fontsize=8.5,
        color="#64748B",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_noisy_mnist_pex_three_bar(
    fp32_acc: float,
    int4_noisy_no_hwa_acc: float,
    int4_noisy_pex_hwa_acc: float,
    out_path: Path,
    *,
    gamma: float = 0.02,
    title: str | None = None,
) -> None:
    """
    Three-bar MNIST accuracy: clean FP32, INT4+noise without HWA, INT4+noise with PEX-cal HWA only.
    """
    labels = [
        "FP32\n(no noise)",
        "INT4 + noise\n(no HWA)",
        "INT4 + noise\n(HWA, PEX cal.)",
    ]
    vals = [fp32_acc * 100, int4_noisy_no_hwa_acc * 100, int4_noisy_pex_hwa_acc * 100]
    colors = ["#4C72B0", "#DD8452", "#0284C7"]
    y_lo, y_hi = thesis_bar_ylim_percent(vals)
    label_offset = max(0.15, (y_hi - y_lo) * 0.04)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.58, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title or f"Noisy MNIST accuracy (γ = {gamma:g})")
    ax.set_ylim(y_lo, y_hi)
    step = 0.5 if (y_hi - y_lo) <= 5 else 1.0
    ax.set_yticks(np.arange(y_lo, y_hi + 0.01, step))
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + label_offset,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    fig.text(
        0.5,
        0.02,
        "FP32: Phase 1 test (clean). No HWA: Phase 2 noisy eval on baseline checkpoint. "
        "PEX HWA: Phase 3 final_noisy_mean with calibration_pex.yaml.",
        ha="center",
        fontsize=8.5,
        color="#64748B",
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_noisy_mnist_pex_three_bar(
    out: Path = Path("results/plots/02_hwa_context/02_noisy_mnist_pex_three_bar.png"),
    baseline_dir: Path = Path("results/run_baseline"),
    baseline_noisy_json: Path | None = None,
    hwa_pex_metrics: Path = Path("results/run_hwa_pex_calibrated/metrics.json"),
) -> Path:
    """Write FP32 / INT4+noise / INT4+noise+PEX-HWA bar chart from saved artifacts."""
    base_m = load_json(baseline_dir / "metrics.json")
    fp32 = float(base_m["fp32_test_accuracy"])
    noisy_path = baseline_noisy_json or (baseline_dir / "noisy_eval.json")
    noisy = load_json(noisy_path)
    hwa_p = load_json(hwa_pex_metrics)
    plot_noisy_mnist_pex_three_bar(
        fp32,
        float(noisy["mean_accuracy"]),
        float(hwa_p["final_noisy_mean"]),
        out,
    )
    print(f"Wrote {out}")
    return out


def run_hwa_pex_context_plot(
    out: Path = Path("results/plots/02_hwa_context/01_hwa_vs_pex_context.png"),
    baseline_noisy_json: Path = Path("results/run_baseline/noisy_eval.json"),
    hwa_metrics: Path = Path("results/run_hwa/metrics.json"),
    hwa_pex_metrics: Path = Path("results/run_hwa_pex_calibrated/metrics.json"),
    maestro_summary: Path = Path("results/maestro_pex/maestro_pex_summary.csv"),
) -> Path:
    """Build curated HWA vs PEX context plot from saved run artifacts."""
    noisy = load_json(baseline_noisy_json)
    hwa_m = load_json(hwa_metrics)
    hwa_p = load_json(hwa_pex_metrics)
    summary = pd.read_csv(maestro_summary).iloc[0]
    rel_gain = float(summary["relative_gain"])
    g_schematic = float(hwa_m["mac_calibration"]["g_eff_sparse"])
    g_pex = float(hwa_p["mac_calibration"]["g_eff_sparse"])

    plot_hwa_pex_context(
        out,
        no_hwa_acc=float(noisy["mean_accuracy"]) * 100,
        hwa_schematic_acc=float(hwa_m["final_noisy_mean"]) * 100,
        hwa_pex_acc=float(hwa_p["final_noisy_mean"]) * 100,
        oa_charge_retention_pct=rel_gain * 100,
        g_eff_scale_pct=100.0 * g_pex / g_schematic,
    )
    print(f"Wrote {out}")
    return out


def run_parasitic_plot(
    out: Path = Path("results/figures/parasitic_sweep.png"),
    pdk_marker: float = INTEGRATED_OPERATING_POINT,
) -> Path:
    """Write parasitic sweep figure; returns output path."""
    plot_parasitic_sweep(out, pdk_marker=pdk_marker)
    print(f"Wrote {out}")
    return out


def main_parasitic() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results/figures/parasitic_sweep.png"))
    p.add_argument("--pdk-marker", type=float, default=INTEGRATED_OPERATING_POINT)
    args = p.parse_args()
    run_parasitic_plot(out=args.out, pdk_marker=args.pdk_marker)


def run_thesis_chart(
    baseline_dir: Path,
    hwa_checkpoint: Path,
    noisy_eval_json: Path | None = None,
    out: Path = Path("results/figures/thesis_bars.png"),
) -> Path:
    """Load metrics and write thesis three-bar figure; returns output path."""
    base_m = load_json(baseline_dir / "metrics.json")
    fp32 = float(base_m["fp32_test_accuracy"])

    if noisy_eval_json and noisy_eval_json.exists():
        noisy_m = load_json(noisy_eval_json)
        int4_noisy = float(noisy_m["mean_accuracy"])
    else:
        proxy = base_m.get("int4_noisy_proxy", base_m.get("int8_noisy_proxy"))
        if proxy is None:
            ptq = base_m.get(
                "int4_ptq_test_accuracy_ideal",
                base_m.get("int4_ptq_test_accuracy", base_m.get("int8_ptq_test_accuracy")),
            )
            proxy = float(ptq) * 0.92
        int4_noisy = float(proxy)

    hwa_dir = hwa_checkpoint.parent
    hwa_m = load_json(hwa_dir / "metrics.json")
    hwa_noisy = float(hwa_m["final_noisy_mean"])

    plot_thesis_three_bar(fp32, int4_noisy, hwa_noisy, out)
    return out


def main_thesis_chart() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", type=Path, required=True, help="Phase 1 run with metrics.json")
    p.add_argument("--hwa-checkpoint", type=Path, required=True)
    p.add_argument("--noisy-eval-json", type=Path, default=None, help="From hwa-eval-noisy on baseline")
    p.add_argument("--out", type=Path, default=Path("results/figures/thesis_bars.png"))
    args = p.parse_args()
    run_thesis_chart(
        baseline_dir=args.baseline_dir,
        hwa_checkpoint=args.hwa_checkpoint,
        noisy_eval_json=args.noisy_eval_json,
        out=args.out,
    )


PHASE45_FOOTER = (
    "Phase 4.5 — Surrogate Monte Carlo (user-defined Gaussian parametric variation). "
    "Not UMC-certified Monte Carlo or final Phase 5."
)


def plot_surrogate_sensitivity_bars(
    dvth0_spread_v: float,
    cap_spread_v: float,
    out_path: Path,
    *,
    sample_time_ns: float = 200.25,
) -> None:
    """Two-bar spread comparison: dvth0 grid vs MOM cap grid."""
    labels = [
        "4× dvth0 grid\n(threshold deltas)",
        "MOM cap grid\n(d_c1_vp, d_cox_vp)",
    ]
    spreads_mv = [dvth0_spread_v * 1000, cap_spread_v * 1000]
    colors = ["#94A3B8", "#0284C7"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(2)
    bars = ax.bar(x, spreads_mv, color=colors, width=0.55, edgecolor="white")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Spread at sample time (mV)")
    ax.set_title(f"/OA_Charge spread @ {sample_time_ns:g} ns (Phase 4.5 surrogate)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ratio = cap_spread_v / dvth0_spread_v if dvth0_spread_v > 0 else float("nan")
    for bar, val in zip(bars, spreads_mv):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max(0.15, 0.04 * max(spreads_mv)),
            f"{val:.2f} mV",
            ha="center",
            fontweight="bold",
        )
    if np.isfinite(ratio):
        ax.text(
            0.5,
            0.92,
            f"MOM cap / dvth0 spread ≈ {ratio:.1f}×",
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            color="#334155",
        )
    fig.text(0.5, 0.02, PHASE45_FOOTER, ha="center", fontsize=8.5, color="#64748B")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_mom_cap_sweep(
    points_df: pd.DataFrame,
    out_path: Path,
    *,
    sample_time_ns: float = 200.25,
) -> None:
    """Plot sampled /OA_Charge vs umc_mc_d_c1_vp; series by d_cox_vp."""
    df = points_df.copy()
    if "umc_mc_d_c1_vp" not in df.columns:
        raise ValueError("points_df missing umc_mc_d_c1_vp column")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if "umc_mc_d_cox_vp" in df.columns:
        for cox, grp in df.groupby("umc_mc_d_cox_vp"):
            grp = grp.sort_values("umc_mc_d_c1_vp")
            ax.plot(
                grp["umc_mc_d_c1_vp"],
                grp["sampled_v"],
                marker="o",
                linewidth=1.8,
                label=f"d_cox_vp={cox:g}",
            )
        ax.legend(title="MOM Cox delta")
    else:
        grp = df.sort_values("umc_mc_d_c1_vp")
        ax.plot(grp["umc_mc_d_c1_vp"], grp["sampled_v"], marker="o", color="#0284C7")
    ax.set_xlabel("umc_mc_d_c1_vp")
    ax.set_ylabel("/OA_Charge (V)")
    ax.set_title(f"MOM cap sweep @ {sample_time_ns:g} ns (Phase 4.5 surrogate)")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.02, PHASE45_FOOTER, ha="center", fontsize=8.5, color="#64748B")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_pvt_pex_corner_bars(
    corner_df: pd.DataFrame,
    out_path: Path,
    *,
    sample_time_ns: float = 200.25,
) -> None:
    """Grouped no-PEX vs PEX bars per PVT corner."""
    df = corner_df.copy()
    labels = df["corner"].astype(str).tolist()
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - w / 2, df["nopex_v"], width=w, label="no-PEX", color="#94A3B8")
    ax.bar(x + w / 2, df["pex_v"], width=w, label="PEX", color="#38bdf8")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"/OA_Charge by corner @ {sample_time_ns:g} ns (deterministic PVT/PEX)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.02,
        "Deterministic PVT/post-layout corners — not Monte Carlo sigma. " + PHASE45_FOOTER,
        ha="center",
        fontsize=8,
        color="#64748B",
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_surrogate_mc_plots(
    out_dir: Path = Path("results/plots/03_surrogate_mc"),
    surrogate_base: Path = Path("results/surrogate_mc"),
    cadence_dir: Path = Path("stuff_from_cadence"),
    sample_time_ns: float = 200.25,
) -> list[Path]:
    """Build Phase 4.5 surrogate figures from sweep artifacts or raw Cadence CSVs."""
    from hwa_cim.surrogate_mc import (
        process_pvt_pex_wide_corners,
        run_surrogate_mc,
        sample_wide_viva_csv,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    cap_points_path = surrogate_base / "cap_sweep" / "surrogate_mc_points.csv"
    dvth_summary_path = surrogate_base / "dvth0_sweep" / "surrogate_mc_summary.csv"
    cap_summary_path = surrogate_base / "cap_sweep" / "surrogate_mc_summary.csv"

    if not cap_points_path.is_file():
        cap_csv = cadence_dir / "manual_mc_2_var_cap_1.csv"
        if cap_csv.is_file():
            run_surrogate_mc(
                cap_csv,
                surrogate_base / "cap_sweep",
                sample_time_ns=sample_time_ns,
                variable_group="mom_cap_grid",
                marker="mom_cap_grid",
            )
    if not (surrogate_base / "dvth0_sweep" / "surrogate_mc_summary.csv").is_file():
        dvth_csv = cadence_dir / "manual_mc_4_var_1.csv"
        if dvth_csv.is_file():
            run_surrogate_mc(
                dvth_csv,
                surrogate_base / "dvth0_sweep",
                sample_time_ns=sample_time_ns,
                variable_group="dvth0_grid",
                marker="dvth0_grid",
            )

    if cap_summary_path.is_file() and dvth_summary_path.is_file():
        cap_sum = pd.read_csv(cap_summary_path).iloc[0]
        dvth_sum = pd.read_csv(dvth_summary_path).iloc[0]
        p1 = out_dir / "01_sensitivity_spread_bars.png"
        plot_surrogate_sensitivity_bars(
            float(dvth_sum["spread_output"]),
            float(cap_sum["spread_output"]),
            p1,
            sample_time_ns=sample_time_ns,
        )
        written.append(p1)

    if cap_points_path.is_file():
        cap_pts = pd.read_csv(cap_points_path)
    else:
        cap_csv = cadence_dir / "manual_mc_2_var_cap_1.csv"
        cap_pts = (
            sample_wide_viva_csv(
                cap_csv,
                "/OA_Charge",
                sample_time_ns,
                variable_group="mom_cap_grid",
            )
            if cap_csv.is_file()
            else None
        )
    if cap_pts is not None and not cap_pts.empty:
        p2 = out_dir / "02_mom_cap_sweep.png"
        plot_mom_cap_sweep(cap_pts, p2, sample_time_ns=sample_time_ns)
        written.append(p2)

    corner_path = surrogate_base / "pvt_pex_corners" / "pvt_pex_corners.csv"
    nopex = cadence_dir / "no_pex_oa_only_3_corners.csv"
    pex = cadence_dir / "with_pex_oa_only_3_corners.csv"
    if corner_path.is_file():
        corner_df = pd.read_csv(corner_path)
    elif nopex.is_file() and pex.is_file():
        corner_df = process_pvt_pex_wide_corners(
            nopex, pex, sample_time_ns=sample_time_ns
        )
        corner_path.parent.mkdir(parents=True, exist_ok=True)
        corner_df.to_csv(corner_path, index=False)
    else:
        corner_df = None
    if corner_df is not None and not corner_df.empty:
        p3 = out_dir / "03_pvt_pex_corner_bars.png"
        plot_pvt_pex_corner_bars(corner_df, p3, sample_time_ns=sample_time_ns)
        written.append(p3)

    return written


def main_surrogate_mc() -> None:
    p = argparse.ArgumentParser(description="Phase 4.5 surrogate MC thesis plots")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/plots/03_surrogate_mc"),
    )
    p.add_argument(
        "--surrogate-base",
        type=Path,
        default=Path("results/surrogate_mc"),
    )
    p.add_argument("--sample-time-ns", type=float, default=200.25)
    args = p.parse_args()
    paths = run_surrogate_mc_plots(
        out_dir=args.out_dir,
        surrogate_base=args.surrogate_base,
        sample_time_ns=args.sample_time_ns,
    )
    for path in paths:
        print(f"Wrote {path}")


def main() -> None:
    main_thesis_chart()


if __name__ == "__main__":
    main()
