"""User-facing wizard copy — profiles, claims, intro."""

from __future__ import annotations

PROFILE_CARDS: list[dict[str, str | bool]] = [
    {
        "mode": "synthetic",
        "title": "Synthetic AFM noise",
        "subtitle": "Fast software noise model for HWA experiments (default thesis path).",
        "enabled": True,
    },
    {
        "mode": "maestro_pex",
        "title": "Maestro PEX (deterministic)",
        "subtitle": "Post-layout gain on /OA_Charge — calibration, not statistical MC.",
        "enabled": True,
    },
    {
        "mode": "surrogate_mc",
        "title": "Phase 4.5 — Surrogate MC",
        "subtitle": "User-defined Gaussian parametric variation; σ from FF/SS ÷ 3.",
        "enabled": True,
    },
    {
        "mode": "cadence_surrogate_stress",
        "title": "Phase 4.5 — Cadence-informed stress",
        "subtitle": "Normalized /OA_Charge spread → output noise for eval + HWA recovery.",
        "enabled": True,
    },
    {
        "mode": "pex_corner_proxy",
        "title": "PEX corner proxy",
        "subtitle": "Deterministic corner spread — stress test, not mismatch statistics.",
        "enabled": True,
    },
    {
        "mode": "monte_carlo_csv",
        "title": "Phase 5 — Foundry Monte Carlo",
        "subtitle": "Future work: per-code statistical CSV from foundry MC exports.",
        "enabled": False,
    },
]

SAFE_CLAIMS: list[str] = [
    "Hardware-aware training improves noisy inference vs a fixed baseline checkpoint.",
    "Maestro PEX provides deterministic post-layout gain evidence on /OA_Charge.",
    "Phase 4.5 surrogate MC shows MOM capacitor variation dominates tested threshold deltas on /OA_Charge.",
    "HWA recovers accuracy under a Cadence-informed surrogate stress model (same stress in Steps 3–4).",
    "The software pipeline is ready to consume a true Phase 5 statistical CSV when available.",
]

UNSAFE_CLAIMS: list[str] = [
    "We ran foundry-certified UMC Monte Carlo (native mc_sp was not used).",
    "Phase 4.5 surrogate summaries complete Phase 5 or replace per-code σ tables.",
    "This proves recovery from foundry-certified MC mismatch (Cadence stress is a proxy).",
    "PEX calibration alone recovers MNIST accuracy lost to noise (HWA training is the recovery story).",
    "The three-corner PVT spread is Monte Carlo mismatch sigma.",
]


def intro_markdown() -> str:
    return """
### SRAM compute-in-memory — why this demo exists

Analog **compute-in-memory (CiM)** can cut energy by doing multiply-accumulate in the SRAM array.
Physical effects — variation, parasitics, and layout — distort those analog MACs.

This walkthrough shows a **small MNIST model** on a **C-2C SRAM CiM** story:

1. **Clean software baseline** (FP32 / INT4 PTQ)
2. **Pick a hardware reality** (synthetic, PEX, Phase 4.5 surrogate MC, …)
3. **Noisy evaluation** — accuracy under CiM-shaped noise
4. **Hardware-aware (HWA) training** — retrain with noise in the loop
5. **Thesis proof** — the three-bar figure your committee expects

Use **Quick demo** if `results/` already has checkpoints from a prior run.
"""


def phase5_future_callout() -> str:
    return (
        "**Phase 5 (foundry Monte Carlo CSV)** is reserved for per-code statistical profiles "
        "(`input_code`, `mean_output`, `sigma`). That path is **future work** in this repo — "
        "use **Advanced lab → Hardware profiles** when a CSV exists."
    )
