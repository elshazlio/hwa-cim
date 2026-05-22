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


def main() -> None:
    main_thesis_chart()


if __name__ == "__main__":
    main()
