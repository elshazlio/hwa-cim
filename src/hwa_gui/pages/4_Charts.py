"""Interactive Plotly charts (dark theme).

Only the selected chart type is loaded/rendered (avoids eager PyTorch imports from
`st.tabs` executing every panel each rerun).
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from hwa_gui.components import apply_page_style, root
from hwa_gui.paths import project_root

os.chdir(project_root())

apply_page_style()

st.title("Charts")

kind = st.selectbox(
    "Chart type",
    [
        "Parasitic sweep",
        "Gamma sweep (CSV)",
        "HWA sweep (CSV)",
        "Thesis bars",
    ],
    index=0,
)

if kind == "Parasitic sweep":
    from hwa_gui.plotly_charts import figure_parasitic_sweep

    pdk = st.slider("PDK marker (ratio)", 0.0, 0.5, 0.30, 0.01)
    st.plotly_chart(figure_parasitic_sweep(pdk_marker=float(pdk)), use_container_width=True)

elif kind == "Gamma sweep (CSV)":
    from hwa_gui.plotly_charts import figure_gamma_sweep_csv

    path = st.text_input("Path to gamma_sweep.csv", value="results/phase2_sweep/gamma_sweep.csv")
    p = root() / path
    if p.exists():
        st.plotly_chart(figure_gamma_sweep_csv(p), use_container_width=True)
    else:
        st.info("Run a gamma sweep first, or adjust the path.")

elif kind == "HWA sweep (CSV)":
    from hwa_gui.plotly_charts import figure_hwa_sweep_csv

    path = st.text_input("Path to hwa_sweep.csv", value="results/sweep_hwa/hwa_sweep.csv")
    p = root() / path
    if p.exists():
        st.plotly_chart(figure_hwa_sweep_csv(p), use_container_width=True)
    else:
        st.info("Run HWA sweep first, or adjust the path.")

else:
    from hwa_gui.plotly_charts import figure_thesis_three_bar

    c1, c2 = st.columns(2)
    bdir = c1.text_input("Baseline dir", value="results/run_baseline")
    hw = c2.text_input("HWA checkpoint", value="results/run_hwa/best.pt")
    noisy = st.text_input("Noisy eval JSON (optional)", value="results/run_baseline/noisy_eval.json")
    nb = root() / noisy
    base_dir = root() / bdir
    hw_ck = root() / hw
    base_metrics = base_dir / "metrics.json"
    hwa_metrics = hw_ck.parent / "metrics.json"
    if not base_metrics.is_file():
        st.info(
            f"Baseline **metrics.json** not found at `{base_metrics}`. "
            "Run **Phase 1** from the Run page (or point “Baseline dir” at a folder that contains `metrics.json`)."
        )
    elif not hwa_metrics.is_file():
        st.info(
            f"HWA **metrics.json** not found at `{hwa_metrics}`. "
            "Run **HWA train** first, or set “HWA checkpoint” to a `best.pt` whose folder has `metrics.json`."
        )
    else:
        st.plotly_chart(
            figure_thesis_three_bar(
                baseline_dir=base_dir,
                hwa_checkpoint=hw_ck,
                noisy_eval_json=nb if nb.is_file() else None,
            ),
            use_container_width=True,
        )
