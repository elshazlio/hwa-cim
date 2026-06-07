"""Compare metrics.json runs side by side."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Compare · SRAM HWA Lab",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from hwa_gui.components import (
    apply_page_style,
    hardware_profile_badge_from_metrics,
    render_pipeline_sidebar,
    root,
)
from hwa_gui.paths import project_root

os.chdir(project_root())

apply_page_style()
render_pipeline_sidebar(current="Compare")

st.title("Compare runs")
st.info("Select multiple `metrics.json` files to compare runs side by side (e.g. Phase 1 baseline vs Phase 3 HWA).")

r = root()
res = r / "results"
files = sorted(res.rglob("metrics.json")) if res.exists() else []
files = [p for p in files if ".venv" not in p.parts]

if not files:
    st.warning("No metrics.json files found under results.")
    st.stop()

choices = st.multiselect(
    "Select metrics.json files",
    options=files,
    default=files[: min(4, len(files))],
    format_func=lambda p: str(p.relative_to(r)),
)

rows: list[dict] = []
for p in choices:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        flat = {
            "run": str(p.relative_to(r)),
            "profile_badge": hardware_profile_badge_from_metrics(d),
            **{k: d[k] for k in d},
        }
        rows.append(flat)
    except Exception as e:
        rows.append({"run": str(p), "error": str(e)})

if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

st.caption(
    "Tip: include Phase 1 baseline + Phase 3 HWA to compare `fp32_test_accuracy` vs `final_noisy_mean`. "
    "Phase 1 logs `int4_ptq_test_accuracy_ideal` and `int4_ptq_test_accuracy_hardware` "
    "(legacy runs may use `int4_ptq_test_accuracy` only — see **AgDR-0001**)."
)
