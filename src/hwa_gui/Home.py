"""HWA-CiM lab dashboard — entrypoint and multipage navigation."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from hwa_gui.paths import project_root

os.chdir(project_root())

_PKG = Path(__file__).resolve().parent
_PAGES = _PKG / "pages"


def _render_home() -> None:
    st.set_page_config(
        page_title="HWA-CiM Lab",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    from hwa_gui.components import (
        apply_page_style,
        health_banner,
        pipeline_quick_reference_md,
        render_pipeline_sidebar,
        root,
    )

    apply_page_style()
    render_pipeline_sidebar(current="Home")

    st.title("HWA-CiM Lab")
    st.caption("Hardware-aware training for C-2C SRAM CiM — thesis pipeline console")

    st.markdown(
        """
### What this is (30 seconds)

This dashboard wraps the same **command-line** experiments you would run for the thesis: train a small
MNIST model, evaluate it with **CiM-shaped noise**, optionally train **hardware-aware (HWA)** weights,
then export **figures** and **metrics** under the `results/` folder in your repo.

**You do not need the GUI to run science** — it is here to **launch jobs**, **watch logs**, and **browse outputs** in one place (good for demos and sanity checks).
"""
    )

    with st.expander("Where to start (recommended path)", expanded=True):
        st.markdown(
            """
1. Open **Run** in the sidebar navigation (top) or use the quick links below.
2. Use the tabs **left to right** the first time: **Baseline** → **Noisy eval** → **HWA train** → **Thesis plot**.
3. Open **Results** to see files; **Compare** to line up `metrics.json` from different runs; **Charts** for interactive Plotly.

The sidebar shows **✓/○** for default folders so you can see where to **resume** after a partial run.
"""
        )

    with st.expander("Phase reference (artifacts & partial runs)", expanded=False):
        st.markdown(pipeline_quick_reference_md())

    st.markdown(
        """
### Metrics note (Phase 1 baseline)

Recent runs record INT4 PTQ accuracy as **`int4_ptq_test_accuracy_ideal`** vs **`int4_ptq_test_accuracy_hardware`**
(optional hardware-shaped calibration). Older `metrics.json` may only have **`int4_ptq_test_accuracy`** — treat that as legacy. See **AgDR-0001** in `docs/agdr/`.
"""
    )

    health_banner()

    st.divider()
    st.subheader("Quick links")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.page_link("pages/1_Run.py", label="Run", icon="▶")
    c2.page_link("pages/2_Results.py", label="Results", icon="📁")
    c3.page_link("pages/3_Compare.py", label="Compare", icon="⚖️")
    c4.page_link("pages/4_Charts.py", label="Charts", icon="📊")
    c5.page_link("pages/5_Noise_Profile.py", label="Noise CSV", icon="🎚️")

    st.caption(f"Project root: `{root()}`")


def _navigation_pages():
    return [
        st.Page(_render_home, title="Home", icon="🏠", default=True),
        st.Page(_PAGES / "1_Run.py", title="Run", icon="▶️"),
        st.Page(_PAGES / "2_Results.py", title="Results", icon="📁"),
        st.Page(_PAGES / "3_Compare.py", title="Compare", icon="⚖️"),
        st.Page(_PAGES / "4_Charts.py", title="Charts", icon="📊"),
        st.Page(_PAGES / "5_Noise_Profile.py", title="Hardware profiles", icon="🎚️"),
    ]


st.navigation(_navigation_pages()).run()
