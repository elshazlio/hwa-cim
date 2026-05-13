"""Phase 5 — validate and preview Monte Carlo noise CSV."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Noise profile · HWA-CiM Lab",
    page_icon="🎚️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from hwa_cim.noise import NoiseProfileCSV

from hwa_gui.components import apply_page_style, render_pipeline_sidebar, root
from hwa_gui.paths import project_root

os.chdir(project_root())

apply_page_style()
render_pipeline_sidebar(current="Noise profile")

st.title("Noise profile (CSV)")
st.info(
    "**Phase 5** — validate a hardware or Monte Carlo noise CSV before **Run → 3a HWA train** "
    "with noise mode **csv**."
)

st.markdown(
    """
Expected columns (case-insensitive): `input_code`, `ideal_output` (or `ideal`),
`mean_output` (or `mean`), `sigma` (or `std`). Optional: `CSNR_dB` / `csnr`.
"""
)

path_str = st.text_input("Path to CSV", value="")
up = st.file_uploader("Or upload a CSV", type=["csv"])

df_preview: pd.DataFrame | None = None
path_used: Path | None = None

if up is not None:
    df_preview = pd.read_csv(up)
    st.success("Uploaded file loaded for preview.")
    tmp = root() / "results" / ".dashboard_upload_noise.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(up.getbuffer())
    path_used = tmp
elif path_str.strip():
    p = root() / path_str.strip()
    if p.exists():
        df_preview = pd.read_csv(p)
        path_used = p
    else:
        st.warning("File not found at that path (relative to project root).")

if df_preview is not None:
    st.subheader("Preview")
    st.dataframe(df_preview.head(50), use_container_width=True)

if path_used is not None and df_preview is not None:
    try:
        prof = NoiseProfileCSV.load(path_used)
        st.subheader("NoiseProfileCSV summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", len(prof.input_code))
        m2.metric("sigma_mean", f"{prof.sigma_mean:.6g}")
        m3.metric("sigma_max", f"{prof.sigma_max:.6g}")
        st.caption(f"Resolved path: `{prof.path}`")
    except Exception as e:
        st.error(f"Validation failed: {e}")

st.caption("Use **Run → HWA train** with noise mode **csv** and this profile path for Phase 5 training.")
