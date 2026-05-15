Based on your project status and the **Analog Foundation Models** methodology outlined in your thesis reports, here is the **Bird's Eye View** of where you stand and what remains.

You are transitioning from **schematic-complete system integration** in Virtuoso toward **layout, parasitic extraction, Monte Carlo hardening**, and **AI co-design at scale** (training on real or PEX-shaped statistics). Three parallel tracks still matter: **physical implementation**, **extracted-netlist validation**, and **hardware-aware software**.

---

## 1. Circuit track — schematic status (Virtuoso)

**Done at schematic level (UMC 65 nm, full custom unless noted):** integrated **SRAM compute-in-memory array (4×4)**, **wordline / row decode**, **DAC**, and **SAR ADC**, with the **SRAM core and MAC array** implemented as **full custom** on the process kit. References remain **Wang (charge-domain C-2C)** and **Analog Foundation Models** for the training story.

**Deliberate shortcut (schedule):** the **comparator inside the SAR** is currently an **ideal model from a built-in Cadence Virtuoso library**, which closed the loop in simulation; the team plans to **replace it with a fully custom comparator** so the entire SAR matches the rest of the macro.

**Still ahead on this track:** silicon-ready **layout** of the same hierarchy, DRC/LVS closure, and sign-off views (not required for the Python repo to run, but required for tapeout narrative).

---

## 2. Physical track — layout and parasitics (the “pain” phase)

This is where **UMC 65 nm** reality dominates over ideal schematic wires.

- **Unit cell and ladder caps:** Layout the **6T SRAM** bitcell together with the **C-2C ladder** capacitors. In this project the ladder is oriented around **MOM finger capacitors** (`MOMCAPS_SY_MMKF` in the FDK), **not** MIM-first planning — see `background_info/HWA_Training_Pipeline_Plan.md` Phase 2 for the MOM vs MIM note and software defaults (`C_UNIT_MOM_DEFAULT` / `C_SERIES_MOM_DEFAULT` in code).
- **PEX:** Extract netlists with parasitics; re-sim staircase and MAC transfer curves.
- **Tuning:** Expect **non-ideal ratios** after extraction; iteratively resize **series** ladder branches (often >2C effective) to recover linearity — this is a core thesis engineering contribution.

---

## 3. AI track — hardware–software co-design (Analog Foundation Models)

Once post-layout behavior is characterized, the emphasis shifts from “fix the analog in layout alone” to **train the model to live with the real statistics**.

- **Step A — Noise profiling (hardware → software):** Monte Carlo on **PEX** netlists; μ and σ (and CSNR) per code / pattern → **noise profile** (tables or CSV).
- **Step B — Noise-aware training:** PyTorch pipeline (`hwa-cim`) injects that profile in the forward pass; optional **schematic-calibrated gain/offset** (`hardware_aware`) already bridges pre-MC work — see `docs/agdr/AgDR-0001`.
- **Step C — Distillation:** Teacher–student to keep the small MNIST student aligned with methodology without endless GPU time.

---

## 4. Final thesis contribution — inter-array scaling

The literature gap on **power in multi-array scaling** still stands.

- **Experiment:** Once you have a **validated macro** (today’s scope is **4×4** schematic-verified; larger tiles are future work), simulate **two** macros and compare **data movement energy** vs **compute energy** on the shared column / interconnect story.
- **Measurement / conclusion:** Pie-chart style breakdown: **compute vs inter-array communication** to support the scalability claim in **65 nm**.

---

## Summary checklist (“the end”)

1. [done] **Schematic:** 4×4 **SRAM–MAC** with **C-2C** ladder logic, **decoder**, **DAC**, and **SAR ADC** integrated in Virtuoso (UMC 65 nm full custom; **SAR comparator** = ideal library cell for now → **custom comparator** planned).
2. [ ] **Layout:** Clean, DRC/LVS-clean **layout** of the macro in UMC 65 nm.
3. [ ] **Tuning:** Post-PEX simulation with acceptable linearity (serial cap / geometry tuning as needed).
4. [ ] **Noise profile:** Error vs output code from **Monte Carlo on extracted layout**; export CSV for Phase 5 in `hwa-cim`.
5. [ ] **AI result:** Chart — accuracy **without** HWA-style training vs **with** training (synthetic then real noise).
6. [ ] **Power breakdown:** Compute vs inter-array data movement (simulation or measurement).

**Related repo docs:** `README.md`, `background_info/HWA_Training_Pipeline_Plan.md`, `docs/software_mission_followups.md` (software gaps aligned to this checklist).
