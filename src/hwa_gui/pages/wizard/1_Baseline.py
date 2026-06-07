"""Wizard Step 1 — Clean software baseline."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="1 · Baseline · SRAM HWA Demo",
    page_icon="1️⃣",
    layout="wide",
    initial_sidebar_state="expanded",
)

from hwa_gui.components import root
from hwa_gui.paths import project_root
from hwa_gui.wizard.actions import (
    baseline_artifacts_present,
    load_metrics_json,
    make_baseline_thunk,
)
from hwa_gui.wizard.charts import show_baseline_metrics_cards
from hwa_gui.wizard.layout import render_expert_paths, setup_wizard_page
from hwa_gui.wizard.runner import (
    is_run_armed,
    maybe_start_job,
    render_wizard_job_status,
    set_run_armed,
)
from hwa_gui.wizard.state import STEP_BASELINE, STEP_HARDWARE, mark_baseline_ready

os.chdir(project_root())

if not setup_wizard_page(
    step=STEP_BASELINE,
    title="Step 1 — Clean software baseline",
    subtitle="Train or load FP32 + INT4 PTQ metrics before hardware noise.",
    required_step=STEP_BASELINE,
):
    st.stop()

paths = render_expert_paths(key_prefix="s1")
repo = root()
baseline_dir = repo / paths["baseline_dir"]
metrics_path = paths["baseline_metrics"]
quick = st.session_state.get("wizard_demo_mode") == "quick"

if quick and baseline_artifacts_present(repo, paths["baseline_ckpt"]):
    metrics = load_metrics_json(repo, metrics_path)
    if metrics:
        st.success("Loaded existing baseline results.")
        show_baseline_metrics_cards(metrics)
        mark_baseline_ready()
elif quick:
    st.warning("No baseline checkpoint yet. Switch to **Live** mode or run training below.")

if not quick or not baseline_artifacts_present(repo, paths["baseline_ckpt"]):
    if st.button("Train baseline (Phase 1)", type="primary", key="s1_train"):
        set_run_armed("baseline", baseline_dir)

    if is_run_armed("baseline", baseline_dir):
        thunk = make_baseline_thunk(
            repo=repo,
            data_dir=repo / paths["data_dir"],
            out_dir=baseline_dir,
        )
        maybe_start_job("baseline", baseline_dir, thunk)

render_wizard_job_status()

j_done = st.session_state.get("wizard_baseline_ready") or baseline_artifacts_present(
    repo, paths["baseline_ckpt"]
)
if j_done:
    metrics = load_metrics_json(repo, metrics_path)
    if metrics and not quick:
        show_baseline_metrics_cards(metrics)
    if baseline_artifacts_present(repo, paths["baseline_ckpt"]):
        mark_baseline_ready()

if baseline_artifacts_present(repo, paths["baseline_ckpt"]):
    if st.button("Continue → Pick hardware reality", type="primary", key="s1_next"):
        st.switch_page("pages/wizard/2_Hardware_Reality.py")
