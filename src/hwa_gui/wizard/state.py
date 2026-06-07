"""Wizard session state and default artifact paths."""

from __future__ import annotations

import streamlit as st

STEP_INTRO = 0
STEP_BASELINE = 1
STEP_HARDWARE = 2
STEP_NOISY = 3
STEP_HWA = 4
STEP_PROOF = 5

DEFAULT_PATHS = {
    "baseline_dir": "results/run_baseline",
    "baseline_ckpt": "results/run_baseline/best.pt",
    "baseline_metrics": "results/run_baseline/metrics.json",
    "noisy_json": "results/run_baseline/noisy_eval.json",
    "hwa_ckpt": "results/run_hwa/best.pt",
    "hwa_metrics": "results/run_hwa/metrics.json",
    "data_dir": "data",
}

WIZARD_PAGE_PATHS = {
    STEP_INTRO: "pages/wizard/0_Intro.py",
    STEP_BASELINE: "pages/wizard/1_Baseline.py",
    STEP_HARDWARE: "pages/wizard/2_Hardware_Reality.py",
    STEP_NOISY: "pages/wizard/3_Noisy_Crash.py",
    STEP_HWA: "pages/wizard/4_HWA_Recovery.py",
    STEP_PROOF: "pages/wizard/5_Thesis_Proof.py",
}


def init_wizard_state() -> None:
    defaults = {
        "wizard_step_max": STEP_INTRO,
        "wizard_profile_mode": "synthetic",
        "wizard_demo_mode": "quick",
        "wizard_baseline_ready": False,
        "wizard_noise_ready": False,
        "wizard_hwa_ready": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def bump_step_max(step: int) -> None:
    st.session_state["wizard_step_max"] = max(
        int(st.session_state.get("wizard_step_max", STEP_INTRO)),
        int(step),
    )


def can_open_step(step: int) -> bool:
    return int(step) <= int(st.session_state.get("wizard_step_max", STEP_INTRO))


def enabled_profile_cards() -> list[dict[str, str | bool]]:
    from hwa_gui.wizard.copy import PROFILE_CARDS

    return [c for c in PROFILE_CARDS if c["enabled"]]


def profile_mode_from_title(title: str) -> str | None:
    for card in enabled_profile_cards():
        if card["title"] == title:
            return str(card["mode"])
    return None


def sync_profile_from_step2_radio() -> str | None:
    """Map Step 2 radio widget state to ``wizard_profile_mode`` when present."""
    label = st.session_state.get("s2_profile_radio")
    if not label:
        return None
    mode = profile_mode_from_title(str(label))
    if mode:
        set_profile_mode(mode)
    return mode


def get_wizard_profile_mode() -> str:
    sync_profile_from_step2_radio()
    mode = str(st.session_state.get("wizard_profile_mode", "synthetic"))
    known = {str(c["mode"]) for c in enabled_profile_cards()}
    return mode if mode in known else "synthetic"


def set_profile_mode(mode: str) -> None:
    st.session_state["wizard_profile_mode"] = mode


def mark_baseline_ready() -> None:
    st.session_state["wizard_baseline_ready"] = True
    bump_step_max(STEP_HARDWARE)


def mark_noise_ready() -> None:
    st.session_state["wizard_noise_ready"] = True
    bump_step_max(STEP_HWA)


def mark_hwa_ready() -> None:
    st.session_state["wizard_hwa_ready"] = True
    bump_step_max(STEP_PROOF)


def is_quick_demo() -> bool:
    return st.session_state.get("wizard_demo_mode", "quick") == "quick"


def go_to_step(step: int) -> None:
    bump_step_max(step)
    st.switch_page(WIZARD_PAGE_PATHS[step])
