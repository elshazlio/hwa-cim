---
id: AgDR-0007
timestamp: 2026-05-23T12:00:00Z
agent: cursor
trigger: user-prompt
status: executed
---

# Cadence-informed surrogate stress training mode

> In the context of **Phase 4.5 `/OA_Charge` surrogate summaries existing but Phase 5 per-code MC CSVs not yet available**, facing **the need to demonstrate HWA recovery under silicon-informed stress without mislabeling it as foundry Monte Carlo**, I decided **a separate `cadence_stress` noise mode and `cadence_surrogate_stress` hardware profile that maps normalized Phase 4.5 output spread to relative output noise**, to achieve **honest baseline + HWA evaluation under the same Cadence-informed proxy**, accepting **this is not UMC-certified mismatch statistics and does not replace `--noise-mode csv` Phase 5**.

## Context

- AgDR-0005 keeps `surrogate_mc` as provenance/plot-only; HWA trained with synthetic γ.
- Cap sweep summary (`results/surrogate_mc/cap_sweep/surrogate_mc_summary.csv`) yields `surrogate_sigma_rel ≈ σ_output / |mean_output| ≈ 1.29%`.
- Thesis claim target: *HWA recovers accuracy under a Cadence-informed surrogate hardware stress model* — not foundry MC or final Phase 5.

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **A. Reuse `surrogate_mc` profile with synthetic γ** | No new mode | Misleading; baseline and HWA use different stress models |
| **B. New `cadence_stress` mode + normalized output noise from Phase 4.5 summary** | Same stress for eval and train; clear provenance | Still a proxy, not per-code MC |
| **C. Fabricate per-code rows from summary for `--noise-mode csv`** | Reuses CSV path | Violates Phase 5 boundary (AgDR-0003/0005) |
| **D. Wait for foundry MC** | Clean statistics | Blocks thesis recovery narrative |

## Decision

Chosen: **B**.

- `noise_mode="cadence_stress"` applies `additive_relative_output_noise(y, sigma_rel)` scaled by `max(abs(y))` per forward pass, after MAC/hardware shaping and before ADC.
- `load_cadence_stress_profile()` reads Phase 4.5 `surrogate_mc_summary.csv` only; rejects Phase 5 `NoiseProfileCSV` schema.
- `HARDWARE_PROFILES["cadence_surrogate_stress"]` labels **Phase 4.5 — Cadence-informed surrogate stress**.
- `surrogate_mc` unchanged (evidence only).

## Consequences

- CLI: `hwa-train-hwa --noise-mode cadence_stress --surrogate-summary …`; `hwa-eval-noisy` gains matching flags; `hwa-plot-cadence-stress` for recovery figure.
- Metrics record `surrogate_sigma_rel`, `surrogate_summary`, `phase_label=Phase 4.5`, `profile_is_foundry_certified=false`, and explicit warning.
- GUI Run page, Hardware profiles tab, and wizard Steps 2–5 expose the mode without calling it Phase 5.
- `NoiseProfileCSV.load()` unchanged; Phase 4.5 summaries still rejected there.

## What not to claim

- Foundry-certified Monte Carlo or Phase 5 completion.
- Per-code mismatch statistics from a single summary row.
- PEX calibration alone recovers MNIST accuracy.

**Safe claim:** HWA improves noisy accuracy when baseline and HWA both see the same Cadence-informed relative output stress derived from Phase 4.5 `/OA_Charge` statistics.
