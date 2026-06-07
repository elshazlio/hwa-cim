"""Wizard layout: progress rail, locks, demo toggle."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from hwa_gui.components import apply_page_style, hardware_profile_info, render_hardware_profile_banner, root
from hwa_gui.wizard.state import (
    DEFAULT_PATHS,
    STEP_BASELINE,
    STEP_HARDWARE,
    STEP_HWA,
    STEP_INTRO,
    STEP_NOISY,
    STEP_PROOF,
    can_open_step,
    enabled_profile_cards,
    get_wizard_profile_mode,
    init_wizard_state,
    profile_mode_from_title,
    set_profile_mode,
    sync_profile_from_step2_radio,
)

_STEP_LABELS = [
    (STEP_INTRO, "Intro"),
    (STEP_BASELINE, "1 · Baseline"),
    (STEP_HARDWARE, "2 · Hardware"),
    (STEP_NOISY, "3 · Noise"),
    (STEP_HWA, "4 · HWA"),
    (STEP_PROOF, "5 · Proof"),
]


def apply_wizard_style() -> None:
    apply_page_style()
    st.markdown(
        """
<style>
    .wizard-rail { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
    .wizard-step-done { font-weight: 600; color: #38bdf8; }
    .wizard-step-pending { opacity: 0.45; }
    .wizard-step-current { font-weight: 700; text-decoration: underline; }
</style>
        """,
        unsafe_allow_html=True,
    )


def setup_wizard_page(
    *,
    step: int,
    title: str,
    subtitle: str,
    required_step: int | None = None,
) -> bool:
    """Common page header. Returns False if step is locked (caller should stop)."""
    init_wizard_state()
    apply_wizard_style()
    render_progress_rail(current_step=step)
    st.title(title)
    st.caption(subtitle)
    render_demo_live_toggle()
    if required_step is not None and not can_open_step(required_step):
        st.warning(
            f"Complete the previous step first. "
            f"You can open up to step {st.session_state.get('wizard_step_max', 0)}."
        )
        if st.button("← Back to previous step", key=f"back_{step}"):
            prev = max(STEP_INTRO, required_step - 1)
            st.switch_page(
                {
                    STEP_INTRO: "pages/wizard/0_Intro.py",
                    STEP_BASELINE: "pages/wizard/1_Baseline.py",
                    STEP_HARDWARE: "pages/wizard/2_Hardware_Reality.py",
                    STEP_NOISY: "pages/wizard/3_Noisy_Crash.py",
                    STEP_HWA: "pages/wizard/4_HWA_Recovery.py",
                }.get(prev, "pages/wizard/0_Intro.py")
            )
        st.stop()
        return False
    return True


def render_progress_rail(*, current_step: int) -> None:
    max_step = int(st.session_state.get("wizard_step_max", STEP_INTRO))
    parts: list[str] = []
    for num, label in _STEP_LABELS:
        if num == current_step:
            cls = "wizard-step-current"
        elif num <= max_step:
            cls = "wizard-step-done"
        else:
            cls = "wizard-step-pending"
        mark = "✓ " if num < current_step and num <= max_step else ""
        parts.append(f'<span class="{cls}">{mark}{label}</span>')
    st.markdown(
        f'<div class="wizard-rail">{" → ".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def render_demo_live_toggle() -> str:
    mode = st.radio(
        "Demo mode",
        options=["quick", "live"],
        format_func=lambda x: "Quick (load existing results)" if x == "quick" else "Live (run training)",
        horizontal=True,
        key="wizard_demo_mode_radio",
        index=0 if st.session_state.get("wizard_demo_mode", "quick") == "quick" else 1,
    )
    st.session_state["wizard_demo_mode"] = mode
    return mode


def render_wizard_hardware_context(
    *,
    step: int,
    allow_change: bool = True,
) -> str:
    """
    Show the active hardware profile (synced from Step 2) and one honest banner.

    Returns the resolved profile mode string.
    """
    sync_profile_from_step2_radio()
    cards = enabled_profile_cards()
    labels = [str(c["title"]) for c in cards]
    modes = [str(c["mode"]) for c in cards]
    current = get_wizard_profile_mode()
    try:
        default_ix = modes.index(current)
    except ValueError:
        default_ix = 0

    if allow_change and step >= STEP_NOISY:
        choice = st.radio(
            "Hardware profile (from Step 2)",
            options=labels,
            index=default_ix,
            key=f"wizard_profile_radio_s{step}",
            horizontal=True,
        )
        mode = profile_mode_from_title(choice) or modes[default_ix]
        set_profile_mode(mode)
        if "s2_profile_radio" in st.session_state:
            st.session_state["s2_profile_radio"] = choice
        current = mode
    else:
        info = hardware_profile_info(current)
        st.caption(f"**Selected profile:** {info.badge}")

    render_hardware_profile_banner(current)

    if current == "cadence_surrogate_stress":
        if step == STEP_NOISY:
            st.caption(
                "Step 3 evaluates the Step 1 checkpoint with **Cadence-informed surrogate stress** "
                "(Phase 4.5 /OA_Charge spread → relative output noise). "
                "Output: `noisy_eval_cadence_stress.json`."
            )
        elif step == STEP_HWA:
            st.caption(
                "Step 4 trains HWA under the **same Cadence-informed stress** as Step 3."
            )
    elif step == STEP_NOISY:
        st.caption(
            f"Profile **{hardware_profile_info(current).badge}** mainly shapes Step 4 HWA training. "
            "This step uses synthetic γ noise on the baseline checkpoint."
        )
    elif current == "surrogate_mc" and step == STEP_HWA:
        st.warning(
            "Phase 4.5 surrogate MC: training uses **synthetic** noise; metrics record provenance only."
        )

    return current


def render_expert_paths(*, key_prefix: str) -> dict[str, str]:
    """Collapsed path overrides for power users."""
    with st.expander("Expert paths (optional)", expanded=False):
        st.caption(f"Project root: `{root()}`")
        c1, c2 = st.columns(2)
        baseline_dir = c1.text_input(
            "Baseline dir",
            value=DEFAULT_PATHS["baseline_dir"],
            key=f"{key_prefix}_bdir",
        )
        hwa_dir = c2.text_input(
            "HWA output dir",
            value=str(Path(DEFAULT_PATHS["hwa_ckpt"]).parent),
            key=f"{key_prefix}_hdir",
        )
    return {
        "baseline_dir": baseline_dir,
        "baseline_ckpt": str(Path(baseline_dir) / "best.pt"),
        "baseline_metrics": str(Path(baseline_dir) / "metrics.json"),
        "noisy_json": str(Path(baseline_dir) / "noisy_eval.json"),
        "hwa_ckpt": str(Path(hwa_dir) / "best.pt"),
        "hwa_metrics": str(Path(hwa_dir) / "metrics.json"),
        "data_dir": DEFAULT_PATHS["data_dir"],
    }

