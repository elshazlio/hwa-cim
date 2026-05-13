"""Browse `results/` artifacts: metrics, checkpoints, CSVs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Results · HWA-CiM Lab",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded",
)

from hwa_gui.components import apply_page_style, render_pipeline_sidebar, root
from hwa_gui.paths import project_root

os.chdir(project_root())

apply_page_style()
render_pipeline_sidebar(current="Results")

st.title("Results browser")
st.info(
    "Browse everything under `results/`: `metrics.json`, checkpoints (`.pt`), CSVs, and figures. "
    "Use this after a **partial run** to confirm files exist before the next **Run** tab."
)

r = root()
res = r / "results"
if not res.exists():
    st.info("No `results/` folder yet. Run a job from **Run** first.")
    st.stop()

mfiles = sorted(res.rglob("metrics.json"))

st.subheader("metrics.json")
pick = st.selectbox("Select run", options=mfiles, format_func=lambda p: str(p.relative_to(r)))
if pick and pick.exists():
    data = json.loads(pick.read_text(encoding="utf-8"))
    st.json(data)

st.subheader("Checkpoints & artifacts")
patterns = ["**/*.pt", "**/*.csv", "**/*.json", "**/*.png"]
seen: set[Path] = set()
rows: list[Path] = []
for pat in patterns:
    for p in res.glob(pat):
        if p.is_file() and p not in seen and p.name != ".dashboard_last.log":
            seen.add(p)
            rows.append(p)
rows.sort(key=lambda x: str(x))
for p in rows[:400]:
    rel = p.relative_to(r)
    st.markdown(f"- `{rel}` ({p.stat().st_size // 1024} KB)")
if len(rows) > 400:
    st.caption(f"Showing first 400 of {len(rows)} files. Narrow with OS search if needed.")

st.caption(f"Project root: `{r}`")
