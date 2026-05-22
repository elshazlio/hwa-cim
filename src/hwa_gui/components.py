"""Shared Streamlit helpers."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from hwa_cim.maestro_pex import HARDWARE_PROFILES, HardwareProfileInfo
from hwa_gui.paths import project_root


def root() -> Path:
    return project_root()


# Default artifact locations (match Run page placeholders) for "where to resume" hints.
_DEFAULT_BASELINE_DIR = Path("results/run_baseline")
_DEFAULT_HWA_CKPT = Path("results/run_hwa/best.pt")
_DEFAULT_NOISY_JSON = Path("results/run_baseline/noisy_eval.json")


def default_pipeline_artifacts() -> dict[str, Path | None]:
    """Presence checks on conventional paths; user may use other folders."""
    r = root()
    bdir = r / _DEFAULT_BASELINE_DIR
    noisy = r / _DEFAULT_NOISY_JSON
    hwa = r / _DEFAULT_HWA_CKPT
    return {
        "baseline_metrics": bdir / "metrics.json" if (bdir / "metrics.json").is_file() else None,
        "baseline_ckpt": bdir / "best.pt" if (bdir / "best.pt").is_file() else None,
        "noisy_eval_json": noisy if noisy.is_file() else None,
        "hwa_ckpt": hwa if hwa.is_file() else None,
    }


def suggested_next_phase_label(art: dict[str, Path | None]) -> str:
    if not art["baseline_ckpt"]:
        return "Run **Phase 1 · Baseline** first — it writes `best.pt` and `metrics.json`."
    if not art["noisy_eval_json"]:
        return "Optional: **Phase 2 · Noisy eval** for the middle bar of the thesis chart (or point the plot at your own JSON)."
    if not art["hwa_ckpt"]:
        return "Next: **Phase 3 · HWA train** (then thesis plot / Compare)."
    return "Core path complete — use **Thesis plot** tab, **Charts**, or **Compare**."


def pipeline_quick_reference_md() -> str:
    """Short markdown: phases, artifacts, partial runs."""
    return """
| Step | In the app | You get (typical paths) |
|------|------------|-------------------------|
| **1** | **Run** → Baseline | `results/run_baseline/best.pt`, `metrics.json` (dual INT4: `…_ideal` / `…_hardware`) |
| **2** | **Run** → Noisy eval or Γ sweep | `noisy_eval.json` and/or CSV under `results/phase2_sweep/` |
| **3** | **Run** → HWA train (or HWA sweep) | `results/run_hwa/best.pt`, `metrics.json` |
| **4** | **Run** → Distill (optional) | `results/run_distill/…` |
| **5** | **Hardware profiles** page | Synthetic / Maestro PEX / corners / MC CSV; then **Run** → HWA |
| **Figures** | **Run** → Thesis / Parasitic **or** **Charts** | PNGs or interactive Plotly |

**Partial runs:** each step reads checkpoints or metrics from disk. Stop after any step, open **Results** or **Compare**, then pick the next tab when you return—nothing is tied to a browser session except the one **running job** on the Run page.
"""


def render_pipeline_sidebar(*, current: str | None = None) -> None:
    """Sidebar: lab blurb, resume checklist, suggested next (built-in nav holds page names)."""
    art = default_pipeline_artifacts()
    with st.sidebar:
        st.markdown("### HWA-CiM Lab")
        st.caption(
            "Train and evaluate a tiny MNIST MLP shaped for a **C-2C SRAM** compute-in-memory "
            "story—noisy eval, hardware-aware training, and thesis figures."
        )
        st.caption("Switch pages with the **navigation** menu at the top of the sidebar.")
        if current:
            st.caption(f"You are on: **{current}**")

        st.divider()
        st.markdown("#### Progress (default folders)")
        c1 = "✓" if art["baseline_ckpt"] else "○"
        c2 = "✓" if art["noisy_eval_json"] else "○"
        c3 = "✓" if art["hwa_ckpt"] else "○"
        st.markdown(
            f"{c1} **1** Baseline &nbsp;·&nbsp; {c2} **2** Noisy / sweep &nbsp;·&nbsp; {c3} **3** HWA"
        )
        st.info(suggested_next_phase_label(art))
        st.caption(
            "✓/○ only reflect `results/run_baseline` and `results/run_hwa` defaults. "
            "Other output dirs still work—use Results to find them."
        )


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


def hardware_profile_info(mode: str) -> HardwareProfileInfo:
    return HARDWARE_PROFILES.get(mode, HARDWARE_PROFILES["synthetic"])


def render_hardware_profile_banner(mode: str) -> None:
    """Visible mode banner so PEX calibration is not confused with Monte Carlo."""
    info = hardware_profile_info(mode)
    st.info(f"**{info.badge}** — {info.banner}")


def hardware_profile_badge_from_metrics(metrics: dict) -> str:
    """Badge label for Results / Compare from saved ``metrics.json``."""
    if metrics.get("hardware_profile_badge"):
        return str(metrics["hardware_profile_badge"])
    mode = metrics.get("hardware_profile_mode", "")
    if mode:
        return hardware_profile_info(str(mode)).badge
    if metrics.get("noise_mode") == "csv":
        return HARDWARE_PROFILES["monte_carlo_csv"].badge
    return HARDWARE_PROFILES["synthetic"].badge


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
