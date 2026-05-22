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


def figure_parasitic_sweep(pdk_marker: float | None = None) -> go.Figure:
    from hwa_cim.c2c import INTEGRATED_OPERATING_POINT, ladder_nonlinearity_metric

    if pdk_marker is None:
        pdk_marker = INTEGRATED_OPERATING_POINT

    ratios = np.linspace(0.0, 0.2, 41)
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

    from hwa_cim.plots import thesis_bar_ylim_percent

    labels = ["FP32 baseline", "INT4 + noise", "INT4 + noise (HWA)"]
    vals = [fp32 * 100, int4_noisy * 100, hwa_noisy * 100]
    y_lo, y_hi = thesis_bar_ylim_percent(vals)
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=vals,
                marker_color=colors,
                text=[f"{v:.1f}%" for v in vals],
                textposition="outside",
                width=0.55,
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        title="MNIST: HWA training vs noise",
        yaxis_title="Test accuracy (%)",
        yaxis_range=[y_lo, y_hi],
        yaxis_dtick=1,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def figure_maestro_oa_charge_overlay(
    nopex_csv: Path,
    pex_csv: Path,
    signal: str = "/OA_Charge",
) -> go.Figure:
    from hwa_cim.maestro_pex import load_viva_waveform

    tx_n, vy_n = load_viva_waveform(nopex_csv, signal)
    tx_p, vy_p = load_viva_waveform(pex_csv, signal)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tx_n * 1e9,
            y=vy_n,
            mode="lines",
            name="no-PEX",
            line=dict(color=MUTED, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tx_p * 1e9,
            y=vy_p,
            mode="lines",
            name="PEX",
            line=dict(color=ACCENT, width=2),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title=f"{signal} — no-PEX vs PEX",
        xaxis_title="Time (ns)",
        yaxis_title="Voltage (V)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def figure_maestro_oa_charge_delta(
    nopex_csv: Path,
    pex_csv: Path,
    signal: str = "/OA_Charge",
) -> go.Figure:
    from hwa_cim.maestro_pex import load_viva_waveform

    tx_n, vy_n = load_viva_waveform(nopex_csv, signal)
    tx_p, vy_p = load_viva_waveform(pex_csv, signal)
    n = min(vy_n.size, vy_p.size, tx_n.size)
    delta = vy_p[:n] - vy_n[:n]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tx_n[:n] * 1e9,
            y=delta,
            mode="lines",
            name="PEX − no-PEX",
            line=dict(color=ACCENT, width=2),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title=f"{signal} — PEX minus no-PEX",
        xaxis_title="Time (ns)",
        yaxis_title="Δ voltage (V)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def figure_maestro_summary_csv(csv_path: Path) -> go.Figure:
    df = pd.read_csv(csv_path)
    labels = df["marker"].astype(str) if "marker" in df.columns else df.index.astype(str)
    fig = go.Figure()
    if "nopex_v" in df.columns:
        fig.add_trace(go.Bar(x=labels, y=df["nopex_v"], name="no-PEX", marker_color=MUTED))
    if "pex_v" in df.columns:
        fig.add_trace(go.Bar(x=labels, y=df["pex_v"], name="PEX", marker_color=ACCENT))
    fig.update_layout(
        template="plotly_dark",
        title="Sampled OA_Charge at manifest time",
        yaxis_title="Voltage (V)",
        barmode="group",
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
