"""Wizard Step 2 — Pick physical hardware reality."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="2 · Hardware · SRAM HWA Demo",
    page_icon="2️⃣",
    layout="wide",
    initial_sidebar_state="expanded",
)

from hwa_gui.components import render_hardware_profile_banner, root
from hwa_gui.paths import project_root
from hwa_gui.wizard.charts import (
    show_maestro_overlay_if_csvs_exist,
    show_pex_context_plot,
    show_surrogate_mc_plots,
)
from hwa_gui.wizard.copy import (
    PROFILE_CARDS,
    SAFE_CLAIMS,
    UNSAFE_CLAIMS,
    phase5_future_callout,
)
from hwa_gui.wizard.layout import setup_wizard_page
from hwa_gui.wizard.state import (
    STEP_HARDWARE,
    STEP_NOISY,
    bump_step_max,
    set_profile_mode,
)

os.chdir(project_root())

if not setup_wizard_page(
    step=STEP_HARDWARE,
    title="Step 2 — Pick physical reality",
    subtitle="Choose how layout and variation enter the story (honest labels per AgDR-0004 / 0005).",
    required_step=STEP_HARDWARE,
):
    st.stop()

enabled = [c for c in PROFILE_CARDS if c["enabled"]]
labels = [c["title"] for c in enabled]
modes = [str(c["mode"]) for c in enabled]
current = st.session_state.get("wizard_profile_mode", "synthetic")
try:
    default_ix = modes.index(current)
except ValueError:
    default_ix = 0

choice = st.radio(
    "Hardware profile",
    options=labels,
    index=default_ix,
    key="s2_profile_radio",
)
selected = enabled[labels.index(choice)]
mode = str(selected["mode"])
set_profile_mode(mode)

st.markdown(f"**{selected['title']}** — {selected['subtitle']}")
render_hardware_profile_banner(mode)

repo = root()

if mode == "cadence_surrogate_stress":
    from hwa_gui.wizard.actions import DEFAULT_SURROGATE_SUMMARY

    summary = repo / DEFAULT_SURROGATE_SUMMARY
    st.info(
        "Cadence-informed stress uses Phase 4.5 `/OA_Charge` statistics as **relative output noise** "
        "in Steps 3–4. Not foundry Monte Carlo or Phase 5."
    )
    show_surrogate_mc_plots(repo)
    if summary.is_file():
        try:
            from hwa_cim.cadence_stress import load_cadence_stress_profile

            prof = load_cadence_stress_profile(summary)
            c1, c2, c3 = st.columns(3)
            c1.metric("σ_output (V)", f"{prof.sigma_output:.6f}")
            c2.metric("|mean_output| (V)", f"{abs(prof.mean_output):.6f}")
            c3.metric("σ_rel (stress)", f"{prof.surrogate_sigma_rel * 100:.3f}%")
            st.caption(f"Source: `{summary.relative_to(repo)}` · marker `{prof.marker}`")
        except Exception as e:
            st.warning(str(e))
    else:
        st.warning(
            f"Missing `{summary.relative_to(repo)}`. Use **Refresh Phase 4.5 plots** below."
        )
    cap_csv = repo / "stuff_from_cadence/manual_mc_2_var_cap_1.csv"
    dvth_csv = repo / "stuff_from_cadence/manual_mc_4_var_1.csv"
    if st.button("Refresh Phase 4.5 plots", key="s2_cadence_sur_plots"):
        if not cap_csv.is_file():
            st.error(f"Missing {cap_csv.relative_to(repo)}")
        else:
            try:
                from hwa_cim.surrogate_mc import run_surrogate_mc
                from hwa_cim.plots import run_surrogate_mc_plots

                sur_out = repo / "results/surrogate_mc"
                run_surrogate_mc(
                    cap_csv,
                    sur_out / "cap_sweep",
                    sample_time_ns=200.25,
                    variable_group="mom_cap_grid",
                    marker="mom_cap_grid",
                )
                if dvth_csv.is_file():
                    run_surrogate_mc(
                        dvth_csv,
                        sur_out / "dvth0_sweep",
                        sample_time_ns=200.25,
                        variable_group="dvth0_grid",
                        marker="dvth0_grid",
                    )
                paths = run_surrogate_mc_plots(
                    surrogate_base=sur_out,
                    sample_time_ns=200.25,
                )
                st.success(f"Wrote {len(paths)} plot(s).")
                st.rerun()
            except Exception as e:
                st.error(str(e))

elif mode == "surrogate_mc":
    st.warning(
        "Phase 4.5 surrogate MC is **not** UMC-certified Monte Carlo. "
        "HWA training still uses **synthetic** noise until code-indexed profiles exist; "
        "this step shows **silicon evidence** and provenance."
    )
    show_surrogate_mc_plots(repo)
    with st.expander("Thesis slide figures (noise taxonomy & pipeline)", expanded=False):
        from hwa_gui.thesis_slides import render_thesis_slide_gallery

        render_thesis_slide_gallery(show_generate_button=True)
    cap_csv = repo / "stuff_from_cadence/manual_mc_2_var_cap_1.csv"
    dvth_csv = repo / "stuff_from_cadence/manual_mc_4_var_1.csv"
    if st.button("Refresh Phase 4.5 plots", key="s2_sur_plots"):
        if not cap_csv.is_file():
            st.error(f"Missing {cap_csv.relative_to(repo)}")
        else:
            try:
                from hwa_cim.surrogate_mc import run_surrogate_mc
                from hwa_cim.plots import run_surrogate_mc_plots

                sur_out = repo / "results/surrogate_mc"
                run_surrogate_mc(
                    cap_csv,
                    sur_out / "cap_sweep",
                    sample_time_ns=200.25,
                    variable_group="mom_cap_grid",
                    marker="mom_cap_grid",
                )
                if dvth_csv.is_file():
                    run_surrogate_mc(
                        dvth_csv,
                        sur_out / "dvth0_sweep",
                        sample_time_ns=200.25,
                        variable_group="dvth0_grid",
                        marker="dvth0_grid",
                    )
                paths = run_surrogate_mc_plots(
                    surrogate_base=sur_out,
                    sample_time_ns=200.25,
                )
                st.success(f"Wrote {len(paths)} plot(s).")
                st.rerun()
            except Exception as e:
                st.error(str(e))

elif mode == "maestro_pex":
    show_pex_context_plot(repo)
    show_maestro_overlay_if_csvs_exist(repo)

elif mode == "pex_corner_proxy":
    st.info("Corner proxy is for stress testing — generate corners in Advanced lab if needed.")

elif mode == "synthetic":
    st.info("Default thesis path: γ·|W| Gaussian noise + optional schematic calibration YAML.")

st.markdown("---")
with st.expander("Safe to say in a presentation", expanded=False):
    for line in SAFE_CLAIMS:
        st.markdown(f"- {line}")
with st.expander("Do not claim", expanded=False):
    for line in UNSAFE_CLAIMS:
        st.markdown(f"- {line}")

st.caption(phase5_future_callout())

if st.button("Continue → Inject noise", type="primary", key="s2_next"):
    set_profile_mode(mode)
    bump_step_max(STEP_NOISY)
    st.switch_page("pages/wizard/3_Noisy_Crash.py")
