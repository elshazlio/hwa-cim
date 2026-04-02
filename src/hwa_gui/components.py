"""Shared Streamlit helpers."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from hwa_gui.paths import project_root


def root() -> Path:
    return project_root()


def apply_page_style() -> None:
    st.markdown(
        """
<style>
    .block-container { padding-top: 1.25rem; max-width: 72rem; }
    div[data-testid="stSidebarNav"] { font-weight: 600; }
    h1 { letter-spacing: -0.02em; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
</style>
        """,
        unsafe_allow_html=True,
    )


def health_banner() -> None:
    """Lightweight status only — avoids importing PyTorch on every Home load."""
    r = root()
    data_ok = (r / "data").exists()
    c1, c2, c3 = st.columns(3)
    c1.metric("Project root", str(r))
    c2.metric("MNIST data folder", "found" if data_ok else "missing (will download)")
    c3.metric("GPU for training", "choose `cuda` in Run if installed")


def confirm_overwrite(out_dir: Path) -> bool:
    """Return True if safe to write, or user confirmed overwrite."""
    if not out_dir.exists():
        return True
    if any(out_dir.iterdir()):
        return st.checkbox(
            f"Output directory `{out_dir}` is not empty. Overwrite / add files anyway?",
            value=False,
            key=f"confirm_{hash(str(out_dir))}",
        )
    return True
