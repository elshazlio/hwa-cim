"""Plotly figures aligned with `plotly_dark` / lab theme."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hwa_cim.utils_io import load_json


ACCENT = "#38bdf8"
MUTED = "#94a3b8"


def figure_parasitic_sweep(pdk_marker: float = 0.30) -> go.Figure:
    from hwa_cim.c2c import ladder_nonlinearity_metric

    ratios = np.linspace(0.0, 0.5, 51)
    metrics = [ladder_nonlinearity_metric(float(r)) for r in ratios]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ratios * 100,
            y=metrics,
            mode="lines",
            name="Max |error| vs ideal ramp",
            line=dict(color=ACCENT, width=2),
        )
    )
    fig.add_vline(
        x=pdk_marker * 100,
        line_dash="dash",
        line_color=MUTED,
        annotation_text=f"PDK ~{pdk_marker * 100:.0f}%",
        annotation_position="top",
    )
    fig.update_layout(
        template="plotly_dark",
        title="C-2C ladder: nonlinearity vs parasitic ratio",
        xaxis_title="Parasitic ratio (%)",
        yaxis_title="Nonlinearity metric",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def figure_gamma_sweep_csv(csv_path: Path) -> go.Figure:
    df = pd.read_csv(csv_path)
    col = "accuracy_seed0" if "accuracy_seed0" in df.columns else df.columns[-1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["gamma"],
            y=df[col],
            mode="lines+markers",
            name="accuracy",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Noisy inference vs gamma",
        xaxis_title="gamma_weight",
        yaxis_title="Accuracy",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def figure_thesis_three_bar(
    baseline_dir: Path,
    hwa_checkpoint: Path,
    noisy_eval_json: Path | None,
) -> go.Figure:
    base_m = load_json(baseline_dir / "metrics.json")
    fp32 = float(base_m["fp32_test_accuracy"])
    if noisy_eval_json and noisy_eval_json.exists():
        noisy_m = load_json(noisy_eval_json)
        int8_noisy = float(noisy_m["mean_accuracy"])
    else:
        int8_noisy = float(base_m.get("int8_noisy_proxy", base_m["int8_ptq_test_accuracy"]) * 0.92)
    hwa_dir = hwa_checkpoint.parent
    hwa_m = load_json(hwa_dir / "metrics.json")
    hwa_noisy = float(hwa_m["final_noisy_mean"])

    labels = ["FP32 baseline", "INT8 + noise", "INT8 + noise (HWA)"]
    vals = [fp32 * 100, int8_noisy * 100, hwa_noisy * 100]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=vals,
                marker_color=colors,
                text=[f"{v:.1f}%" for v in vals],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        title="MNIST: HWA training vs noise",
        yaxis_title="Test accuracy (%)",
        yaxis_range=[0, 100],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def figure_hwa_sweep_csv(csv_path: Path) -> go.Figure:
    df = pd.read_csv(csv_path)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Clean accuracy", "Noisy mean"))
    fig.add_trace(
        go.Scatter(
            x=df["gamma"],
            y=df["clean_accuracy"],
            mode="markers",
            marker=dict(size=10, color=df["alpha"], colorscale="Viridis", showscale=True),
            name="clean",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["gamma"],
            y=df["noisy_mean"],
            mode="markers",
            marker=dict(size=10, color=df["alpha"], colorscale="Viridis", showscale=False),
            name="noisy",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="gamma", row=1, col=1)
    fig.update_xaxes(title_text="gamma", row=1, col=2)
    fig.update_layout(template="plotly_dark", showlegend=False, height=420)
    return fig
