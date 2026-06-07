"""Wizard Step 5 — Thesis proof chart and claims."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="5 · Proof · SRAM HWA Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from hwa_gui.components import hardware_profile_badge_from_metrics, root
from hwa_gui.paths import project_root
from hwa_gui.wizard.actions import (
    hwa_artifacts_present,
    load_metrics_json,
    load_noisy_eval,
)
from hwa_gui.wizard.charts import show_cadence_stress_recovery, show_thesis_three_bar
from hwa_gui.wizard.copy import SAFE_CLAIMS, UNSAFE_CLAIMS
from hwa_gui.wizard.layout import render_expert_paths, setup_wizard_page
from hwa_gui.wizard.layout import render_wizard_hardware_context
from hwa_gui.wizard.state import STEP_PROOF

os.chdir(project_root())

if not setup_wizard_page(
    step=STEP_PROOF,
    title="Step 5 — Thesis proof",
    subtitle="The three-bar story: FP32 · INT4+noise · HWA recovery.",
    required_step=STEP_PROOF,
):
    st.stop()

paths = render_expert_paths(key_prefix="s5")
repo = root()
profile = render_wizard_hardware_context(step=STEP_PROOF, allow_change=True)
baseline_dir = repo / paths["baseline_dir"]
hwa_ckpt = repo / paths["hwa_ckpt"]
noisy_rel = paths["noisy_json"]
if profile == "cadence_surrogate_stress":
    noisy_rel = "results/run_baseline/noisy_eval_cadence_stress.json"
noisy_json = repo / noisy_rel

if not hwa_artifacts_present(repo, paths["hwa_ckpt"]):
    st.error("Complete Step 4 first — HWA metrics missing.")
    st.stop()
hwa_m = load_metrics_json(repo, paths["hwa_metrics"]) or {}
st.markdown(
    f"**Selected profile (Step 2):** `{hardware_profile_badge_from_metrics(hwa_m)}` "
    f"(`{profile}`)"
)

nb = noisy_json if noisy_json.is_file() else None
if profile == "cadence_surrogate_stress":
    st.subheader("Cadence-informed stress recovery")
    show_cadence_stress_recovery(repo, baseline_dir=baseline_dir, hwa_ckpt=hwa_ckpt)
else:
    try:
        show_thesis_three_bar(
            repo,
            baseline_dir=baseline_dir,
            hwa_ckpt=hwa_ckpt,
            noisy_json=nb,
        )
    except Exception as e:
        st.error(f"Could not build chart: {e}")

thesis_png = repo / "results/figures/thesis_bars.png"
if thesis_png.is_file():
    st.subheader("Static figure (repo)")
    st.image(str(thesis_png), use_container_width=True)

st.subheader("What you can say")
for line in SAFE_CLAIMS:
    st.markdown(f"- {line}")

st.subheader("What not to say")
for line in UNSAFE_CLAIMS:
    st.markdown(f"- {line}")

st.subheader("Thesis slide pack")
from hwa_gui.thesis_slides import render_thesis_slide_gallery

render_thesis_slide_gallery(show_generate_button=True)

st.success(
    "Demo complete. Use **Advanced lab** for gamma sweeps, distillation, "
    "and full hardware profile tooling."
)

if st.button("← Restart from Intro", key="s5_restart"):
    st.session_state["wizard_step_max"] = 0
    st.switch_page("pages/wizard/0_Intro.py")
