"""Background training jobs with tee'd logs for live Streamlit refresh."""

from __future__ import annotations

import sys
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any


@dataclass
class JobState:
    running: bool = False
    done: bool = False
    error: str | None = None
    log_path: Path | None = None


_jobs: dict[str, JobState] = {}


class _Tee:
    def __init__(self, *streams: object) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)  # type: ignore[attr-defined]
            s.flush()  # type: ignore[attr-defined]
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()  # type: ignore[attr-defined]


def _run_in_thread(
    job_id: str,
    log_path: Path,
    fn: Callable[[], Any],
) -> None:
    st = _jobs[job_id]
    st.running = True
    st.done = False
    st.error = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            tee = _Tee(sys.__stdout__, logf)
            tee_err = _Tee(sys.__stderr__, logf)
            old_out, old_err = sys.stdout, sys.stderr
            try:
                sys.stdout = tee  # type: ignore[assignment]
                sys.stderr = tee_err  # type: ignore[assignment]
                fn()
            finally:
                sys.stdout = old_out
                sys.stderr = old_err
    except Exception:
        st.error = traceback.format_exc()
        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write("\n--- exception ---\n")
            logf.write(st.error)
    finally:
        st.running = False
        st.done = True


def start_job(job_id: str, log_path: Path, fn: Callable[[], Any]) -> JobState:
    """Start `fn` in a daemon thread; stdout/stderr appended to log_path."""
    if job_id not in _jobs:
        _jobs[job_id] = JobState()
    st = _jobs[job_id]
    if st.running:
        raise RuntimeError("A job is already running.")
    st.log_path = log_path
    t = threading.Thread(target=_run_in_thread, args=(job_id, log_path, fn), daemon=True)
    t.start()
    return st


def get_job(job_id: str) -> JobState | None:
    return _jobs.get(job_id)


def tail_log(log_path: Path, max_chars: int = 24_000) -> str:
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def fragment_interval() -> timedelta:
    return timedelta(seconds=0.7)
