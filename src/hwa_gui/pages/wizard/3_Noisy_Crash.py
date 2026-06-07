"""Wizard Step 3 — Noisy evaluation."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="3 · Noise · SRAM HWA Demo",
    page_icon="3️⃣",
    layout="wide",
    initial_sidebar_state="expanded",
)

from hwa_gui.components import root
from hwa_gui.paths import project_root
from hwa_gui.wizard.actions import (
    DEFAULT_SURROGATE_SUMMARY,
    baseline_artifacts_present,
    load_metrics_json,
    load_noisy_eval,
    make_noisy_eval_thunk,
)
from hwa_gui.wizard.charts import show_noisy_crash_metrics
from hwa_gui.wizard.layout import render_expert_paths, render_wizard_hardware_context, setup_wizard_page
from hwa_gui.wizard.runner import (
    clear_run_armed,
    is_run_armed,
    maybe_start_job,
    render_wizard_job_status,
    set_run_armed,
)
from hwa_gui.wizard.state import STEP_HWA, STEP_NOISY, mark_noise_ready

os.chdir(project_root())

if not setup_wizard_page(
    step=STEP_NOISY,
    title="Step 3 — Inject noise",
    subtitle="Evaluate the baseline checkpoint under CiM-shaped noise.",
    required_step=STEP_NOISY,
):
    st.stop()

paths = render_expert_paths(key_prefix="s3")
repo = root()
profile = render_wizard_hardware_context(step=STEP_NOISY)

if not baseline_artifacts_present(repo, paths["baseline_ckpt"]):
    st.error("Complete Step 1 first — baseline checkpoint missing.")
    st.stop()

baseline_m = load_metrics_json(repo, paths["baseline_metrics"])
ckpt = repo / paths["baseline_ckpt"]
noisy_rel = paths["noisy_json"]
if profile == "cadence_surrogate_stress":
    noisy_rel = "results/run_baseline/noisy_eval_cadence_stress.json"
noisy_path = repo / noisy_rel
quick = st.session_state.get("wizard_demo_mode") == "quick"


def _noisy_ready() -> bool:
    return noisy_path.is_file()


if quick and _noisy_ready():
    noisy = load_noisy_eval(repo, noisy_rel)
    if baseline_m and noisy:
        st.success("Loaded noisy evaluation results.")
        show_noisy_crash_metrics(baseline_m, noisy)
        mark_noise_ready()
elif quick:
    if profile == "cadence_surrogate_stress":
        st.warning(
            f"No `{noisy_path.name}` yet for **Cadence-informed stress**. "
            "Switch demo mode to **Live (run training)** and click **Run noisy evaluation**, "
            "or run Phase 2 from Advanced lab → Run."
        )
    else:
        st.warning(f"No `{noisy_path.name}` yet. Switch to Live or run below.")

_RUN_LABEL = "noisy eval"
_out_dir = ckpt.parent

if not quick or not _noisy_ready():
    if st.button("Run noisy evaluation", type="primary", key="s3_run"):
        set_run_armed(_RUN_LABEL, _out_dir)

    if is_run_armed(_RUN_LABEL, _out_dir):
        sur_summary = (
            repo / DEFAULT_SURROGATE_SUMMARY
            if profile == "cadence_surrogate_stress"
            else None
        )
        thunk, err = make_noisy_eval_thunk(
            repo=repo,
            checkpoint=ckpt,
            data_dir=repo / paths["data_dir"],
            gamma=0.02,
            seeds=10,
            device="cpu",
            out_json=noisy_path,
            profile_mode=profile,
            surrogate_summary=sur_summary,
        )
        if err:
            clear_run_armed(_RUN_LABEL, _out_dir)
            st.error(err)
        elif thunk:
            maybe_start_job(_RUN_LABEL, _out_dir, thunk)

render_wizard_job_status()

if _noisy_ready():
    noisy = load_noisy_eval(repo, noisy_rel)
    if baseline_m and noisy:
        if not quick:
            show_noisy_crash_metrics(baseline_m, noisy)
        mark_noise_ready()

if _noisy_ready():
    if st.button("Continue → HWA recovery", type="primary", key="s3_next"):
        st.switch_page("pages/wizard/4_HWA_Recovery.py")
