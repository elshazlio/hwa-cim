---
id: AgDR-0004
timestamp: 2026-05-22T00:00:00Z
agent: cursor
trigger: user-prompt
status: executed
---

# Maestro PEX calibration path separate from Monte Carlo CSV

> In the context of **Maestro/VIVA PEX exports that only cover inference charge sharing (no read mode / sense amp)**, facing **the need for defensible HWA artifacts before true Phase 5 MC CSVs exist**, I decided **a parallel `hwa-maestro-pex` path that calibrates from `/OA_Charge` no-PEX vs PEX pairs and keeps `--noise-mode csv` for statistical MC only**, to achieve **deterministic post-layout gain evidence in HWA training without mislabeling corner or PEX deltas as Monte Carlo σ**, accepting **a second calibration YAML (`calibration_pex.yaml`), GUI mode complexity, and explicit warnings in `metrics.json`**.

## Context

- `stuff_from_cadence/nopex1.csv` and `pex1.csv` are VIVA waveform exports with many probes; this testbench never engages read mode, so `Read_Out_*` must not drive calibration.
- AgDR-0003 added schematic YAML + rich MC CSV; Phase 5 remains blocked on **statistical** MC tables (`input_code`, `ideal_output`, `mean_output`, `sigma`).
- Thesis wording must distinguish **deterministic PEX distortion** on `/OA_Charge` from **Phase 5 statistical noise profiles**.

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **A. Fold PEX CSVs into Phase 5 `NoiseProfileCSV`** | One loader | Misrepresents single-point PEX delta as per-code σ; breaks MC schema semantics |
| **B. Separate Maestro PEX module + manifest + CLI** | Clear hardware boundary; manifest documents signal/sample time | Extra command and GUI modes to maintain |
| **C. Manual gain tweak only (no tool)** | Zero code | Not reproducible; poor thesis traceability |
| **D. Wait for MC before any HWA calibration** | Clean statistics only | Blocks thesis progress with existing PEX data |

## Decision

Chosen: **B**.

- Parser samples `/OA_Charge` at `sample_time_ns` from manifest; ignores `Read_Out_*`.
- `hwa-maestro-pex` writes `maestro_pex_summary.csv` + `maestro_pex_metrics.json`; optional `calibration_pex.yaml` scales MAC gains from `relative_gain = pex_v / nopex_v` when metadata allows.
- Recommended HWA experiment: `--calibration-yaml config/calibration_pex.yaml --noise-mode synthetic` (AFM-style noise), **not** `--noise-mode csv`.
- Future `corner_runs` in manifest may export a **corner-derived σ proxy** CSV, explicitly labeled non-MC.
- `metrics.json` records `hardware_profile_mode`, `profile_signal`, `profile_is_statistical`, `profile_warning`.

## Consequences

- New CLI: `hwa-maestro-pex`; GUI **Hardware profiles** page with Synthetic / Maestro PEX / PEX corners / Monte Carlo CSV tabs.
- Run page maps **Hardware profile mode** to internal `noise_mode` + calibration paths.
- Follow-up: multiple PEX corners → corner proxy profile; true MC export → existing Phase 5 path unchanged.

## Empirical outcome (MNIST pipeline, documented so claims stay honest)

- **HWA vs no HWA:** large gain (~94% → ~97% noisy test accuracy at γ=0.02 on baseline checkpoint path).
- **PEX-calibrated HWA vs schematic-calibrated HWA:** **no meaningful uplift** (~97.0% vs ~97.1%); PEX scales `g_eff` (~0.91× from `/OA_Charge` at 200.25 ns) but does not substitute for MC σ training.
- **`thesis_bars.png`:** unchanged narrative (FP32 / INT4+noise / HWA); **no** isolated “accuracy after PEX only” bar unless a future eval mode is added.
- **Power (W):** not modeled; do not cite Maestro PEX as reducing power loss.

## What not to claim in thesis text

- “PEX CSV replaced the Monte Carlo noise profile.”
- “HWA recovered accuracy lost to PEX” **without** a dedicated PEX-only eval bar and data.
- “PEX calibration improved MNIST accuracy” based on the example `run_hwa_pex_calibrated` vs `run_hwa` comparison.

**Safe claims:** post-layout **deterministic** gain evidence on `/OA_Charge`; readiness for Phase 5 MC CSV; HWA improves robustness vs naive noisy inference.
