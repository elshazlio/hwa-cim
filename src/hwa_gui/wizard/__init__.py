"""Guided thesis demo wizard — session state and navigation helpers."""

from hwa_gui.wizard.state import (
    STEP_BASELINE,
    STEP_HARDWARE,
    STEP_HWA,
    STEP_INTRO,
    STEP_NOISY,
    STEP_PROOF,
    bump_step_max,
    can_open_step,
    init_wizard_state,
    set_profile_mode,
)

__all__ = [
    "STEP_INTRO",
    "STEP_BASELINE",
    "STEP_HARDWARE",
    "STEP_NOISY",
    "STEP_HWA",
    "STEP_PROOF",
    "init_wizard_state",
    "bump_step_max",
    "can_open_step",
    "set_profile_mode",
]
