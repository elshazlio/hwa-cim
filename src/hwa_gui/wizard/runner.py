"""Start background jobs from wizard pages (same contract as Run page)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import streamlit as st

from hwa_gui.components import confirm_overwrite, root
from hwa_gui.job_runner import get_job, start_job
from hwa_gui.log_panel import JOB_ID, render_live_log


def run_arm_key(fn_label: str, out_dir: Path) -> str:
    return f"run_arm_{fn_label}_{out_dir.resolve()}"


def set_run_armed(fn_label: str, out_dir: Path) -> None:
    st.session_state[run_arm_key(fn_label, out_dir)] = True


def clear_run_armed(fn_label: str, out_dir: Path) -> None:
    st.session_state.pop(run_arm_key(fn_label, out_dir), None)


def is_run_armed(fn_label: str, out_dir: Path) -> bool:
    return bool(st.session_state.get(run_arm_key(fn_label, out_dir)))


def maybe_start_job(fn_label: str, out_dir: Path, thunk: Callable[[], None]) -> bool:
    """Start a background job when armed and overwrite is confirmed."""
    if not is_run_armed(fn_label, out_dir):
        return False
    j = get_job(JOB_ID)
    if j and j.running:
        st.warning("A job is already running. Wait for it to finish.")
        return False
    if not confirm_overwrite(out_dir):
        st.info("Enable the overwrite confirmation to proceed.")
        return False
    clear_run_armed(fn_label, out_dir)
    log_path = root() / "results" / ".dashboard_last.log"
    st.session_state["log_path"] = log_path
    start_job(JOB_ID, log_path, thunk)
    st.success(f"Started: {fn_label}")
    st.rerun()
    return True


def wizard_maybe_start(fn_label: str, out_dir: Path, thunk: Callable[[], None]) -> bool:
    """Return True if job was started."""
    return maybe_start_job(fn_label, out_dir, thunk)


def render_wizard_job_status() -> None:
    j = get_job(JOB_ID)
    if not j:
        return
    if j.running:
        st.warning("Job running — log updates below.")
    elif j.done and j.error:
        st.error("Last job failed — see log.")
    elif j.done:
        st.success("Last job finished.")
    log_path = st.session_state.get("log_path")
    if log_path:
        render_live_log(Path(log_path))
