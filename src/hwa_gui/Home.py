"""HWA-CiM lab dashboard — home."""

from __future__ import annotations

import os

import streamlit as st

from hwa_gui.components import apply_page_style, health_banner, root
from hwa_gui.paths import project_root

os.chdir(project_root())

st.set_page_config(
    page_title="HWA-CiM Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_page_style()

st.title("HWA-CiM Lab")
st.caption("Hardware-aware training for C-2C SRAM CiM — pipeline console")

st.markdown(
    """
Welcome. Use the sidebar to **run** training/eval jobs, **browse** `results/`, **compare** metrics,
open **interactive charts**, and **validate** Phase-5 noise CSVs.

**Pipeline (thesis workflow)**  
1. **Baseline** — FP32 + INT8 PTQ + MAC parity  
2. **Noisy eval / gamma sweep** — Phase 2 on a baseline checkpoint  
3. **HWA train** (optional sweep) — noise-aware training  
4. **Distill** — teacher–student  
5. **Plots** — thesis bar chart, parasitic sweep  
6. **Noise profile** — Monte Carlo CSV when hardware data arrives  
"""
)

health_banner()

st.divider()
st.subheader("Quick links")
c1, c2, c3 = st.columns(3)
c1.page_link("pages/1_Run.py", label="Run pipeline", icon="▶")
c2.page_link("pages/2_Results.py", label="Results browser", icon="📁")
c3.page_link("pages/4_Charts.py", label="Charts", icon="📊")

st.caption(f"Project root: `{root()}`")
