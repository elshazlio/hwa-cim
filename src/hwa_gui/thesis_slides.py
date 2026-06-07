"""Thesis slide plot gallery for Streamlit (PNG + generate button)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from hwa_gui.components import root

SLIDE_PLOT_DIR = Path("results/plots/04_thesis_slides")

SLIDE_FIGURES: list[tuple[str, str, str]] = [
    (
        "01_hardware_profile_taxonomy.png",
        "Slide 26 — Hardware profile taxonomy",
        "Synthetic vs PEX vs corners vs Phase 4.5 vs future Phase 5.",
    ),
    (
        "02_phase45_to_phase5_pipeline.png",
        "Slide 36 — Phase 4.5 → Phase 5 pipeline",
        "What we have now vs foundry per-code MC CSV.",
    ),
    (
        "03_validation_ladder.png",
        "Slide 29 — Validation ladder",
        "Python MAC → MNIST → Cadence → calibration → HWA.",
    ),
    (
        "04_pvt_pex_corner_bars_fixed.png",
        "Slide 31 — PVT/PEX corners (fixed pairing)",
        "Grouped no-PEX vs PEX per nominal/FF/SS/FNSP.",
    ),
    (
        "05_results_summary_dashboard.png",
        "Slide 30 — Results dashboard",
        "FP32, noisy INT4, HWA (+ optional PEX-cal HWA).",
    ),
    (
        "06_phase5_csv_schema_mock.png",
        "Slide 36 — Phase 5 CSV schema",
        "Target columns for future foundry MC.",
    ),
]


def generate_thesis_slide_plots() -> list[Path]:
    from hwa_cim.thesis_slide_plots import run_thesis_slide_plots

    return run_thesis_slide_plots(repo_root=root())


def render_thesis_slide_gallery(*, show_generate_button: bool = True) -> None:
    """Show thesis slide PNGs; optional button to regenerate."""
    repo = root()
    out = repo / SLIDE_PLOT_DIR

    if show_generate_button:
        if st.button("Generate thesis slide figures", type="primary", key="gen_thesis_slides"):
            with st.spinner("Building slide figures…"):
                try:
                    paths = generate_thesis_slide_plots()
                    st.success(f"Wrote {len(paths)} figure(s) under `{SLIDE_PLOT_DIR}/`.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    any_found = False
    for filename, title, caption in SLIDE_FIGURES:
        p = out / filename
        if p.is_file():
            any_found = True
            st.subheader(title)
            st.caption(caption)
            st.image(str(p), use_container_width=True)

    if not any_found:
        st.info(
            f"No figures in `{SLIDE_PLOT_DIR}/` yet. Click **Generate thesis slide figures** "
            "or run `hwa-plot-thesis-slides` from the repo root."
        )
