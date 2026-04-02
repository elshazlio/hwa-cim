"""Compare metrics.json runs side by side."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from hwa_gui.components import apply_page_style, root
from hwa_gui.paths import project_root

os.chdir(project_root())

apply_page_style()

st.title("Compare runs")

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
        flat = {"run": str(p.relative_to(r)), **{k: d[k] for k in d}}
        rows.append(flat)
    except Exception as e:
        rows.append({"run": str(p), "error": str(e)})

if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

st.caption("Tip: include Phase 1 baseline + Phase 3 HWA to compare `fp32_test_accuracy` vs `final_noisy_mean`.")
