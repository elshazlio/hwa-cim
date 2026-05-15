# Software follow-ups (mission alignment)

This list ties the **`hwa-cim`** codebase to the **hardware + AFM** story in `background_info/Bird's Eye View of Our Thesis.md` and `background_info/HWA_Training_Pipeline_Plan.md`. Items are **not** all urgent; they are flags for when hardware data or design decisions land.

## When the custom SAR comparator replaces the ideal library cell

- Consider a **small optional model** (threshold noise, kickback, or meta-stability probability) in the forward path or in **ADC STE** — only if Spectre exports justify it; default can stay off.
- Re-run **Phase 2 / 3** baselines and add an **AgDR** if training semantics change.

## Phase 5 — real noise (PEX + Monte Carlo)

- **`noise.py`:** Replace **`sigma_mean`** summarization with **per-code** (and optionally per **weight-population** class) σ injection when the CSV schema grows — already sketched in `background_info/HWA_CIM_Required_Changes.md`.
- **Regression:** Add tests that load a **tiny fixture CSV** and assert deterministic noise shaping (golden tensors or bounds).
- **Docs:** One-page “how to export from Spectre → column names” in `docs/` once the layout team freezes the schema.

## Schematic ↔ Python numerical closure

- **Verified vectors:** Implement (or finish) parity tests against **fixed Virtuoso stimulus vectors** once **physical units** and **dequant mapping** are agreed — the colleague doc draft had placeholder mV conversion; do not ship guesses.
- **`calibration.yaml`:** Optional single source for **G_eff**, offsets, and sim metadata (clock 20 MHz / 50 ns, VDD) when you want Git-diffable calibration history without code edits.

## Parasitic / ladder plots vs silicon intent

- Default **`pdk_marker`** / ladder narrative should track **G-05 MOMCAP** extraction when available (`HWA_Training_Pipeline_Plan.md` Appendix); avoid presenting **heuristic** markers as measured silicon.

## GUI / thesis artifacts

- Ensure **Streamlit Run** presets and **README** quick start mention **dual INT4 metrics** and **`--no-hardware-aware`** where relevant (already in CLI table).
- Commit **frozen sweep JSON/CSV** for thesis figures when runs stabilize (reproducibility).

## Inter-array power (not this repo)

- No code change expected here; track as **measurement or Cadence power** task. Link thesis figures from hardware supplementary material when ready.

---

**Mission in one line:** Keep the **PyTorch side** honest about **what is ideal math**, **what is schematic calibration**, and **what is post-layout MC**, so thesis claims stay traceable to **Virtuoso exports** and AgDRs in `docs/agdr/`.
