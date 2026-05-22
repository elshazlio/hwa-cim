---
id: AgDR-0003
timestamp: 2026-05-21T00:00:00Z
agent: cursor
trigger: user-prompt
status: executed
---

# YAML MAC calibration and richer Phase-5 CSV noise

> In the context of **thesis-grade traceability of schematic knobs and Monte Carlo exports**, facing **constants scattered in `c2c.py` and a single `sigma_mean` for CSV training**, I decided **to add `config/calibration.yaml` + `MacCalibrationConfig` loader (PyYAML) and per-code output/weight σ lookup in `NoiseProfileCSV`**, to achieve **Git-diffable calibration and code-dependent noise without breaking legacy CSV fixtures**, accepting **PyYAML as a dependency and verified-vector regression tests remaining skipped until mV scaling is agreed**.

## Context

- AgDR-0001 landed optional `hardware_aware` gain/offset; numeric defaults still lived only in `src/hwa_cim/c2c.py`.
- Phase 5 needs richer MC tables (σ vs code, optional population / measured G_eff columns) per `background_info/HWA_CIM_Required_Changes.md`.
- Eight fixed Virtuoso vectors need regression tests only after **software ↔ mV** mapping is validated — placeholder tests must not ship guessed conversions.

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **A. JSON config only** | No new dependency | Thesis narrative asked for YAML; less familiar to analog teammates using YAML in roadmaps |
| **B. YAML + dataclass loader, constants remain fallback** | Diffable config; safe if file missing | Two sources unless loader is the documented path |
| **C. Always load YAML (fail if missing)** | Single source | Breaks minimal installs / tests without file |
| **D. Ship 8-vector test with rough mV scale** | Immediate regression | Misleading pass/fail per roadmap author warning |

## Decision

Chosen: **B** for calibration, **skip (not fail) 8-vector test** until scaling is agreed, **per-code σ interpolation** for CSV mode on layer outputs (and optional per-weight σ from `input_code` column).

- Default YAML: `config/calibration.yaml` at repo root; override via `--calibration-yaml`.
- `INTEGRATED_OPERATING_POINT = 0.17` in `c2c.py` for parasitic plot marker (heuristic, documented as non-silicon).
- Extended CSV columns parsed when present; old 5-column fixtures unchanged.

## Consequences

- CLI: `hwa-train-hwa` / `hwa-train-distill` accept `--calibration-yaml`; values flow into `NoisyMicroMLP` / `c2c_mac` kwargs.
- README gains a **Limitations** section; `docs/agdr/README.md` index updated.
- Follow-up: enable `test_verified_vectors` when Virtuoso ↔ dequant mapping is frozen; tile-level G_eff if schematic demands it.
