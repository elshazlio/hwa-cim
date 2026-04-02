"""Resolve project root so CLI and Streamlit share `data/` and `results/`."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Directory containing `pyproject.toml` (repo root)."""
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").is_file():
            return p
    return here.parent.parent
