"""Wizard Step 0 — Introduction."""

from __future__ import annotations

import os

import streamlit as st

st.set_page_config(
    page_title="Intro · SRAM HWA Demo",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from hwa_gui.components import root
from hwa_gui.paths import project_root
from hwa_gui.wizard.copy import intro_markdown, phase5_future_callout
from hwa_gui.wizard.layout import apply_wizard_style, render_progress_rail
from hwa_gui.wizard.state import STEP_BASELINE, bump_step_max, init_wizard_state

os.chdir(project_root())

init_wizard_state()
apply_wizard_style()
render_progress_rail(current_step=0)

st.title("SRAM CiM — Hardware-Aware Training Demo")
st.markdown(intro_markdown())

st.info(phase5_future_callout())

st.markdown(
    """
```
  [ MNIST ] ──► [ C-2C SRAM MAC ] ──► analog noise / PEX / variation
                      │
                      ▼
              HWA retrains weights in software
                      │
                      ▼
              recover noisy accuracy (thesis bars)
```
"""
)

if st.button("Start demo", type="primary", key="intro_start"):
    bump_step_max(STEP_BASELINE)
    st.switch_page("pages/wizard/1_Baseline.py")

st.caption(
    "Power users: use **Advanced lab** in the sidebar for full Run tabs, sweeps, and CSV paths."
)
st.caption(f"Project root: `{root()}`")
