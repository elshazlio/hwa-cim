---
id: AgDR-0001
timestamp: 2026-05-13T00:00:00Z
agent: cursor
trigger: user-prompt
status: executed
---

# Hardware-aware MAC calibration in the PyTorch pipeline

> In the context of **modeling a schematic-calibrated C-2C SRAM MAC in `hwa-cim`**, facing **ideal INT4 matmul not matching gain/offset behavior from verification**, I decided **to add an optional `hardware_aware` path (population-based G_eff and dense offset) shared by `c2c_mac` and `NoisyQuantLinear`, defaulting off for parity and Phase 2**, to achieve **honest INT4 baselines and HWA training that can see systematic nonlinearity**, accepting **approximate per-row population vs per-tile hardware, and duplicated metric keys in Phase 1 JSON**.

## Context

- The analog macro is expected to show **pattern-dependent effective gain** (sparse vs dense weights) and a **residual offset** in the dense regime, per schematic-level verification notes (`background_info/HWA_CIM_Required_Changes.md`).
- The codebase had **`c2c_mac` = ideal dequantized matmul** and **HWA layers using `F.linear` + Gaussian noise only**, so the training loop did not see the same systematic transfer function as the INT4 evaluation story.
- Thesis work needs **reproducible comparisons**: ideal parity tests must remain valid; Phase 1 metrics must not silently change meaning.

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **A. Always-on hardware correction** | Single code path, always “physical” | Breaks `parity_linear_vs_c2c` and existing tests; blurs “ideal MAC spec” vs “calibrated model”. |
| **B. Optional flag, default off for MAC and noisy layer; Phase 3 turns on** | Preserves Phase 2; explicit A/B via CLI; parity stays ideal | Two behavioral modes to document; risk of forgetting flag in new scripts. |
| **C. YAML-only calibration file** | Single source of truth outside code | Extra dependency and loader work; deferred until needed. |
| **D. Float heuristic on activations for G_eff** (avg magnitude) | Easy with STE weights | Second, inconsistent definition of “population” vs INT4 `c2c_mac`. |

## Decision

Chosen: **B**, with **population computed from `symmetric_quantize_int4` of clipped weights** in `NoisyQuantLinear` (same INT4 grid as the `c2c_mac` path), **not** the float heuristic in D.

- **`c2c_mac`**: `hardware_aware=False` default; optional keyword-only calibration overrides.
- **`NoisyQuantLinear` / `NoisyMicroMLP`**: `hardware_aware=False` default so `run_noisy_eval` / Phase 2 unchanged.
- **`train_hwa` / `train_distill`**: pass `hardware_aware=True` by default, with **`--no-hardware-aware`** for legacy comparison.
- **Phase 1 metrics**: replace single INT4 key with **`int4_ptq_test_accuracy_ideal`** and **`int4_ptq_test_accuracy_hardware`**.
- **`HWAConfig`**: extended fields mirroring defaults imported from `src/hwa_cim/c2c.py` (single numeric source there).

Rejected for now: **C** (optional follow-up). Rejected: **D** for training path to stay aligned with INT4 population model.

## Consequences

- New agents should read `src/hwa_cim/c2c.py` constants and this AgDR before changing defaults or “simplifying” to one INT4 number.
- Plots/GUI fall back to legacy `int4_ptq_test_accuracy` keys when present; prefer `_ideal` for proxies.
- **Tradeoff accepted:** row-wise population average is a first-order stand-in for per-tile hardware behavior until explicit tiling or Monte Carlo profiles land (Phase 5).
