"""Live log panel using Streamlit fragments (polls log file)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from hwa_gui.job_runner import fragment_interval, get_job, tail_log

JOB_ID = "hwa_gui_main"


def render_live_log(log_path: Path | None, job_id: str = JOB_ID) -> None:
    """Show tail of log file; auto-refreshes while job is running."""
    if not log_path:
        return

    @st.fragment(run_every=fragment_interval())
    def _frag() -> None:
        j = get_job(job_id)
        running = j is not None and j.running
        text = tail_log(log_path)
        title = "Run log (live)" if running else "Run log"
        st.caption(title)
        st.code(text, language=None)

    _frag()
