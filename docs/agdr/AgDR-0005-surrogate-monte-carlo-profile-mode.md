---
id: AgDR-0005
timestamp: 2026-05-23T00:00:00Z
agent: cursor
trigger: user-prompt
status: executed
---

# Phase 4.5 surrogate Monte Carlo profile mode (user-defined Gaussian parametric variation)

> In the context of **native UMC Monte Carlo being blocked while wide VIVA surrogate sweep exports exist**, facing **the risk of mislabeling grid/parametric sweeps as foundry-certified Phase 5 MC**, I decided **a separate `surrogate_mc` hardware profile and `hwa-surrogate-mc` parser path labeled Phase 4.5 — Surrogate Monte Carlo (user-defined Gaussian parametric variation)**, to achieve **thesis-grade sensitivity artifacts and honest provenance without overloading `--noise-mode csv`**, accepting **no code-indexed training profile until future multi-stimulus surrogate exports exist**.

## Context

- Cadence manual MC bypass produced wide VIVA CSVs (`manual_mc_2_var_cap_1.csv`, `manual_mc_4_var_1.csv`) and 4-corner PEX/PVT exports under `stuff_from_cadence/`.
- Sigma values are estimated from FF/SS corner deltas (`sigma ≈ |corner delta| / 3`), not foundry-certified mismatch statistics.
- AgDR-0003 reserves `--noise-mode csv` + `NoiseProfileCSV` for per-code statistical profiles (`input_code`, `ideal_output`, `mean_output`, `sigma`).
- AgDR-0004 separates deterministic PEX calibration from MC; corner proxy CSVs are explicitly non-MC.
- Readiness report classifies current work as **Phase 4.5**, not final Phase 5.

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **A. Feed surrogate summaries through `monte_carlo_csv` + `NoiseProfileCSV`** | Reuses training path | Mislabels provenance; Phase 5S summary lacks `input_code` |
| **B. Separate parser + `surrogate_mc` profile; no CSV training hook yet** | Honest boundary; thesis plots + metrics | HWA cannot train on surrogate σ until code-indexed profile exists |
| **C. Reshape surrogate σ into fake per-code rows** | Enables `--noise-mode csv` quickly | Misleading thesis claims; hard to audit |
| **D. Wait for foundry MC before any software path** | Clean statistics only | Wastes existing surrogate evidence |

## Decision

Chosen: **B**, with user-facing naming:

- **Phase label:** `Phase 4.5`
- **Display name:** `Surrogate Monte Carlo with user-defined Gaussian parametric variation`
- **Internal mode:** `surrogate_mc`
- **Profile kind:** `user_defined_gaussian_parametric_surrogate_mc`

Implementation:

- `hwa-surrogate-mc` parses wide VIVA CSVs, samples `/OA_Charge` at `sample_time_ns` (default 200.25 ns), writes per-point + summary artifacts.
- `HARDWARE_PROFILES["surrogate_mc"]` records `profile_is_statistical=True`, `profile_is_foundry_certified=False`, `sigma_source=corner_delta_div_3`, and explicit warning.
- GUI **Hardware profiles** tab and Run page list **Phase 4.5 — Surrogate Monte Carlo (user-defined Gaussian parametric variation)**; training uses synthetic noise + optional schematic calibration unless a future code-indexed surrogate profile is added.
- Phase 5 `NoiseProfileCSV.load()` remains unchanged; Phase 4.5 summaries are rejected if passed without `input_code`.

## Consequences

- New CLI: `hwa-surrogate-mc`; optional `hwa-plot-surrogate-mc` for curated figures under `results/plots/03_surrogate_mc/`.
- Extended `HardwareProfileInfo` / `hardware_profile_metrics_extra()` with `phase_label`, `profile_display_name`, `profile_kind`, `profile_is_foundry_certified`, `sigma_source`.
- `--hardware-profile-mode` gains `surrogate_mc` in `hwa-train-hwa`.
- Follow-up: multi-stimulus surrogate sweeps per `input_code` could justify a labeled HWA experiment; foundry MC still uses `monte_carlo_csv`.

## What not to claim

- “We ran UMC Monte Carlo.”
- “This completes Phase 5.”
- “3-corner PVT spread is MC sigma.”
- “Surrogate summaries are foundry-certified mismatch σ.”

**Safe claims:** surrogate flow implemented; MOM cap variation dominates tested `/OA_Charge` sensitivity; PEX/PVT corner plots are deterministic post-layout evidence; software ready for true Phase 5 CSV when available.
