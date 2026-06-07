"""SRAM HWA Lab — guided demo + advanced lab navigation (sram-hwa-hybrid-ai)."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from hwa_gui.paths import project_root

os.chdir(project_root())

_PKG = Path(__file__).resolve().parent
_WIZ = _PKG / "pages" / "wizard"
_ADV = _PKG / "pages"


def _navigation_pages():
    return {
        "Guided demo": [
            st.Page(
                _WIZ / "0_Intro.py",
                title="Intro",
                icon="🏠",
                default=True,
            ),
            st.Page(_WIZ / "1_Baseline.py", title="1 · Baseline", icon="1️⃣"),
            st.Page(_WIZ / "2_Hardware_Reality.py", title="2 · Hardware", icon="2️⃣"),
            st.Page(_WIZ / "3_Noisy_Crash.py", title="3 · Noise crash", icon="3️⃣"),
            st.Page(_WIZ / "4_HWA_Recovery.py", title="4 · HWA recovery", icon="4️⃣"),
            st.Page(_WIZ / "5_Thesis_Proof.py", title="5 · Thesis proof", icon="📊"),
        ],
        "Advanced lab": [
            st.Page(_ADV / "1_Run.py", title="Run", icon="▶️"),
            st.Page(_ADV / "2_Results.py", title="Results", icon="📁"),
            st.Page(_ADV / "3_Compare.py", title="Compare", icon="⚖️"),
            st.Page(_ADV / "4_Charts.py", title="Charts", icon="📊"),
            st.Page(_ADV / "5_Noise_Profile.py", title="Hardware profiles", icon="🎚️"),
        ],
    }


st.navigation(_navigation_pages()).run()
