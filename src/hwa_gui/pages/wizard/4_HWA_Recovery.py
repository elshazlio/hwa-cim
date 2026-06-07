"""Wizard Step 4 — Hardware-aware recovery."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="4 · HWA · SRAM HWA Demo",
    page_icon="4️⃣",
    layout="wide",
    initial_sidebar_state="expanded",
)

from hwa_gui.components import root
from hwa_gui.paths import project_root
from hwa_gui.wizard.actions import (
    DEFAULT_SURROGATE_SUMMARY,
    hwa_artifacts_present,
    load_metrics_json,
    load_noisy_eval,
    make_hwa_train_thunk,
)
from hwa_gui.wizard.charts import show_hwa_recovery_metrics
from hwa_gui.wizard.layout import render_expert_paths, render_wizard_hardware_context, setup_wizard_page
from hwa_gui.wizard.runner import (
    clear_run_armed,
    is_run_armed,
    maybe_start_job,
    render_wizard_job_status,
    set_run_armed,
)
from hwa_gui.wizard.state import STEP_HWA, STEP_PROOF, mark_hwa_ready

os.chdir(project_root())

if not setup_wizard_page(
    step=STEP_HWA,
    title="Step 4 — Hardware-aware recovery",
    subtitle="Train HWA weights with the profile selected in Step 2.",
    required_step=STEP_HWA,
):
    st.stop()

paths = render_expert_paths(key_prefix="s4")
repo = root()
profile = render_wizard_hardware_context(step=STEP_HWA)

if profile == "monte_carlo_csv":
    st.error("Phase 5 MC is not available in the guided demo.")
    st.stop()

hwa_dir = repo / Path(paths["hwa_ckpt"]).parent
quick = st.session_state.get("wizard_demo_mode") == "quick"
noisy = load_noisy_eval(repo, paths["noisy_json"])

if quick and hwa_artifacts_present(repo, paths["hwa_ckpt"]):
    hwa_m = load_metrics_json(repo, paths["hwa_metrics"])
    if hwa_m:
        st.success("Loaded HWA results.")
        show_hwa_recovery_metrics(hwa_m, noisy)
        mark_hwa_ready()
elif quick:
    st.warning("No HWA checkpoint yet. Switch to Live or train below.")

if not quick or not hwa_artifacts_present(repo, paths["hwa_ckpt"]):
    if st.button("Train HWA (Phase 3)", type="primary", key="s4_train"):
        set_run_armed("HWA train", hwa_dir)

    if is_run_armed("HWA train", hwa_dir):
        sur_summary = (
            repo / DEFAULT_SURROGATE_SUMMARY
            if profile == "cadence_surrogate_stress"
            else None
        )
        thunk, err = make_hwa_train_thunk(
            repo=repo,
            data_dir=repo / paths["data_dir"],
            out_dir=hwa_dir,
            profile_mode=profile,
            device="cpu",
            surrogate_summary=sur_summary,
        )
        if err:
            clear_run_armed("HWA train", hwa_dir)
            st.error(err)
        elif thunk:
            maybe_start_job("HWA train", hwa_dir, thunk)

render_wizard_job_status()

if hwa_artifacts_present(repo, paths["hwa_ckpt"]):
    hwa_m = load_metrics_json(repo, paths["hwa_metrics"])
    if hwa_m:
        if not quick:
            show_hwa_recovery_metrics(hwa_m, noisy)
        mark_hwa_ready()

if hwa_artifacts_present(repo, paths["hwa_ckpt"]):
    if st.button("Continue → Thesis proof", type="primary", key="s4_next"):
        st.switch_page("pages/wizard/5_Thesis_Proof.py")
