# Hardware-Aware Training Pipeline — Parallel Work Plan

**Project:** Charge-Domain SRAM CiM for Edge Inference (UMC 65nm)
**Methodology:** Analog Foundation Models [Rasch et al., 2025]
**Target Model:** Micro-MLP for MNIST (thesis-scoped)
**Owner:** Omar (or delegated team member)
**Hardware dependency (for *this* Python repo):** NONE until Phase 5 **Monte Carlo CSV** is ready — you can train on synthetic noise and schematic-style gain/offset today.

**Cadence schematic status (team, 2026):** Virtuoso includes **schematic-level** integration of the **4×4 SRAM–MAC** macro, **decoder**, **DAC**, and **SAR ADC** (UMC 65 nm full custom for the SRAM/MAC core; **SAR comparator** currently an **ideal Cadence library** model for schedule, to be replaced with full custom). **Layout and PEX** remain the bridge from those schematics to the **Phase 5** files this repo ingests.

---

## Why This Can Start Now

The entire software pipeline — from baseline model to noise-injected training to data distillation — requires exactly one input from the hardware track: a noise profile (μ, σ) extracted from Monte Carlo simulations on the post-layout extracted netlist. That input arrives at the very end. Everything before it is pure Python/PyTorch and can be built, tested, and validated with synthetic placeholder noise.

The thesis deliverable checklist explicitly requires:

- **"Accuracy without Noise Training" vs. "Accuracy with Noise Training"** — a comparative chart
- **A noise profile graph** (Error vs. Output Code from Monte Carlo)
- **End-to-end validation** using the Analog Foundation Models methodology

Building the software infrastructure now means that when the hardware noise profile arrives, you plug it in and generate your thesis results. If you wait, the software becomes the critical path *after* the hardware is done — adding weeks.

---

## Phase 1: Ideal Baseline (No Hardware Knowledge Needed)

**Duration:** ~3–5 days
**Goal:** A working, well-characterized micro-MLP that classifies MNIST with known accuracy. This is your "golden reference" for all future comparisons.

### Tasks

1. **Build a micro-MLP in PyTorch**
  - Architecture: 784 → 128 → 64 → 10 (or similar small topology)
  - Why this size: it must be realistically mappable onto a 4×4 SRAM CiM array. Each 4×4 tile computes a partial MAC of 4 inputs against 4-bit weights per column; MNIST's 784-input layer requires tiling across multiple tile invocations.
  - Use standard training: SGD or Adam, cross-entropy loss, 10–20 epochs.
2. **Quantize weights to 4-bit integers**
  - Your C-2C ladder has 4 rows; each row stores 1 binary weight bit. Effective weight precision per column = 4 bits. Input activations are driven by a 4-bit DAC, one per row. The model must operate at this precision end-to-end.
  - Implement post-training quantization (PTQ) first: clamp weights to [−8, 7] (signed INT4), scale activations to 4-bit unsigned [0, 15].
  - Record accuracy drop from FP32 → INT4. This is your "quantization-only" baseline.
  - **Repository:** `hwa-train-baseline` writes **`int4_ptq_test_accuracy_ideal`** (ideal `c2c_mac`, matches `nn.Linear` parity) and **`int4_ptq_test_accuracy_hardware`** (same INT4 grid with schematic-style gain/offset). Report both where the thesis distinguishes ideal quantization from macro-shaped behavior.
3. **Build a software MAC simulator**
  - Write a Python function `c2c_mac(weights_4bit, activations_4bit)` that computes the ideal (noiseless) output of your C-2C ladder.
  - This is the mathematical model from Wang's equation: `V_OUT = V_REF * Σ(b_i * 2^(i-k))` where k=4 (4-bit precision).
  - For a 4×4 array: `OA = (1/4) * Σ_j Σ_i IA_j * W_{j,n,i} * 2^(i-4)` — passive charge-sharing averaging across 4 rows.
  - Validate: feed the same inputs through PyTorch's `nn.Linear` and through your MAC simulator. The outputs must match exactly (within floating-point tolerance) for the ideal case.

### Verifiable Deliverables


| #   | Deliverable               | Pass Criteria                                              |
| --- | ------------------------- | ---------------------------------------------------------- |
| 1.1 | Trained FP32 micro-MLP    | ≥97% MNIST test accuracy                                   |
| 1.2 | INT4 quantized model      | Accuracy recorded; expected drop from FP32 baseline        |
| 1.3 | Software MAC simulator    | Output matches `nn.Linear` within 1e-6 for all test inputs; validated at 4-bit input/weight precision |
| 1.4 | Baseline comparison table | FP32 vs INT4 accuracy logged (`int4_ptq_test_accuracy_ideal` + `int4_ptq_test_accuracy_hardware` in `metrics.json`) |


---

## Phase 2: C-2C Non-Ideality Model (PDK-Calibrated Placeholder Noise)

**Duration:** ~5–7 days
**Goal:** Build the noise injection layer that will eventually accept real hardware parameters, but for now uses synthetic noise calibrated to realistic ranges from the Analog Foundation Models paper AND from actual UMC 65nm process data.

### Tasks

1. **Model parasitic capacitance error (MOM cap, PDK-grounded)**
  - The C-2C ladder uses MOMCAPS_SY_MMKF symmetric MOM finger capacitors from the UMC 65nm FDK. Characterized values: C ≈ 15.1 fF (unit cap), 2C ≈ 30.2 fF (series cap). These are NOT MIM caps — all previous references to MIM bottom-plate parasitics are inapplicable.
  - MOM cap parasitics are dominated by fringe capacitance to adjacent metal fingers and substrate coupling, not a bottom-plate stack. Parasitic ratio for MOM is process-geometry dependent; use a sweep range of 0–20% (tighter than MIM's 25–35%).
  - Implement a `C2CLadderWithParasitics` class that takes a `parasitic_ratio` parameter and C_unit=15.1e-15, C_series=30.2e-15 as defaults. Compute the distorted transfer curve.
  - Sweep `parasitic_ratio` from 0% to 20% and plot the transfer curve nonlinearity (gaps/overlaps as described in Wang Fig. 11).
  - Mark the nominal operating point on the plot. Exact parasitic σ pending G-05 MOMCAP SPICE model — use placeholder until that document is available.
2. **Model capacitor mismatch (random noise)**
  - Per the Analog Foundation Models methodology, add Gaussian noise to the weights during the forward pass.
  - Implement the noise injection from equation (5) of the AFM paper: `W_noisy = W + (γ_weight · max(|W|) + β_weight · |W|) · τ` where `τ ~ N(0, I)`
  - Start with additive Gaussian noise (the AFM paper found additive performs comparably to affine for their use case): `W_noisy = W + γ_weight · max(|W|) · τ`
  - Use `γ_weight = 0.02` as the starting point (the optimal value from AFM paper's sweep in their Fig. 5).
3. **Model ADC quantization at the output**
  - Your SAR-ADC is 4-bit. The accumulated analog voltage gets quantized to 4 bits (16 levels) at readout.
  - Add 4-bit output quantization to the forward pass after the MAC. Full-scale range maps to [0, 15].
4. **Evaluate noisy inference (no retraining yet)**
  - Take your Phase 1 quantized model.
  - Run inference 10× with different random seeds (critical per the AFM paper methodology).
  - Record mean and std of accuracy under noise.
  - This is your "off-the-shelf model under hardware noise" baseline. The AFM paper saw ~8–10% accuracy drop at this stage.

### Verifiable Deliverables


| #   | Deliverable                        | Pass Criteria                                                                                                            |
| --- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 2.1 | Parasitic transfer curve sweep     | Plot showing nonlinearity vs. parasitic ratio. Matches Wang Fig. 11 behavior qualitatively                               |
| 2.2 | Noise injection layer              | Drops into PyTorch model as a custom `nn.Module`. Forward pass adds noise, backward pass uses straight-through estimator |
| 2.3 | Noisy inference results (10 seeds) | Table: mean accuracy ± std under γ = 0.02 noise. Expected ~5–15% drop from INT4 baseline                                 |
| 2.4 | Noise sensitivity sweep            | Plot: accuracy vs. γ_weight from 0.00 to 0.10. Reproduces the general shape of AFM paper Fig. 5                          |


---

## Phase 3: Hardware-Aware Training Loop

**Duration:** ~5–7 days
**Goal:** Train the micro-MLP with noise injection active during training, producing a noise-resilient model. This is the core thesis contribution on the software side.

### Tasks

1. **Implement weight clipping**
  - The AFM paper found weight clipping is more impactful than noise injection alone for LLMs. For small models it's still beneficial.
  - During training, clamp weight range to α standard deviations: `W_clipped = clamp(W, -α·std(W), α·std(W))`
  - The AFM paper used α = 3.0. Try α ∈ {2.0, 3.0, 4.0}.
  - Rationale: clipping maps small weights to larger conductance values with higher SNR.
2. **Noise-aware training**
  - Train from scratch (not fine-tuning the Phase 1 model).
  - Inject noise every forward pass during training with the same γ_weight used in Phase 2.
  - Use straight-through estimator (STE) for backpropagation through the quantization and noise layers.
  - **Repository:** `hwa-train-hwa` enables **schematic-style gain/offset** on the noisy linear forward path by default; use **`--no-hardware-aware`** to reproduce runs from before that behavior existed.
  - Train for 20–50 epochs (the model is small, this will be fast).
3. **Hyperparameter sweep**
  - Sweep γ_weight ∈ {0.01, 0.02, 0.04} × α ∈ {2.0, 3.0, 4.0} = 9 experiments.
  - Each experiment: train, then evaluate with noise 10× with different seeds.
  - Find the Pareto-optimal point: highest clean accuracy that maintains robustness under noise.
4. **Generate the thesis comparison chart**
  - This is explicitly on your checklist: "Accuracy without Noise Training (Low) vs. Accuracy with Noise Training (High)"
  - Three bars: (a) FP32 baseline, (b) INT4 + noise (no HWA training), (c) INT4 + noise (with HWA training)
  - The gap between (b) and (c) is your proof that HWA training works for your architecture.

### Verifiable Deliverables


| #   | Deliverable                  | Pass Criteria                                                                 |
| --- | ---------------------------- | ----------------------------------------------------------------------------- |
| 3.1 | Weight clipping ablation     | Table showing clean vs. noisy accuracy for each α value                       |
| 3.2 | Noise-aware trained model    | Noisy accuracy within 2–3% of clean INT4 baseline (vs. ~10% drop without HWA) |
| 3.3 | Hyperparameter sweep results | 9-cell table (γ × α) with mean ± std accuracy                                 |
| 3.4 | Thesis comparison bar chart  | Three-bar chart matching the checklist deliverable. Publishable quality       |


---

## Phase 4: Data Distillation Pipeline

**Duration:** ~3–5 days
**Goal:** Implement the teacher-student knowledge distillation framework described in your thesis methodology section 6.2.

### Tasks

1. **Train a "teacher" model**
  - Larger MLP: 784 → 512 → 256 → 128 → 10
  - Train in FP32 to maximum accuracy (~99%+)
  - This represents the "large pre-trained model" in the AFM methodology.
2. **Generate synthetic training data**
  - Use the teacher to label a synthetic dataset (or generate soft labels for existing MNIST data).
  - Soft labels = teacher's output probability distribution (not hard argmax).
  - This is the "data distillation" step from section 6.2.
3. **Train the student (your micro-MLP) with distillation + noise**
  - Loss = α·KL(teacher_soft_labels, student_output) + (1-α)·CE(hard_labels, student_output)
  - Noise injection active during training.
  - Compare distilled-student accuracy vs. direct-trained accuracy from Phase 3.

### Verifiable Deliverables


| #   | Deliverable                                | Pass Criteria                                           |
| --- | ------------------------------------------ | ------------------------------------------------------- |
| 4.1 | Trained teacher model                      | ≥99% MNIST test accuracy                                |
| 4.2 | Distilled student model (no noise)         | Accuracy ≥ Phase 1 direct training                      |
| 4.3 | Distilled student model (with noise + HWA) | Accuracy ≥ Phase 3 direct HWA training                  |
| 4.4 | Distillation comparison table              | Direct vs. distilled, clean vs. noisy, all combinations |


---

## Phase 5: Plug In Real Hardware Noise (BLOCKED — Needs Cadence Data)

**Duration:** ~3–5 days (once data arrives)
**Goal:** Replace placeholder noise with the actual noise profile extracted from Monte Carlo simulations on your post-layout extracted 65nm macro.

### Layout scope — decision log (fill in when frozen)

**Pending:** Lay out either **(A)** the SRAM unit cell with its MAC slice, or **(B)** the full **4×4 SRAM MAC array** — pick one as the first extraction target; note the choice here once decided.

**Handoff to the training/software track:** The PyTorch pipeline does **not** need GDSII or SKILL by itself. What unblocks Phase 5 is a **Monte Carlo summary file** (typically **CSV**) matching the schema below (`input_code`, `ideal_output`, `mean_output`, `sigma`, optional `CSNR_dB`), exported after analog MC on the **PEX/post-layout extracted** netlist of whichever block you laid out.

**PEX netlists:** Essential inside **Virtuoso/Spectre** as the extracted netlist used for MC — archive revision-controlled exports with your thesis supplementary material if possible. This repo **does not ingest raw PEX** today unless we add custom parsers; simulation-derived **CSV** remains the integration surface.

### Inputs Required From Hardware Track

1. **Monte Carlo simulation results** from extracted layout:
  - Run N = 100–1000 Monte Carlo iterations
  - For each iteration: sweep all 16 input codes (4-bit)
  - Record output voltage for each code per iteration
2. **Extract noise profile:**
  - For each output code: compute μ(V_out) and σ(V_out)
  - Compute CSNR = 10·log10(mean(DMAC²) / mean(|DMAC − AMAC|²)) per the Yoshioka formula
  - Export as a CSV lookup table: `[input_code, ideal_output, mean_output, sigma, CSNR_dB]`

### Tasks

1. Replace `γ_weight · max(|W|) · τ` with the real σ-per-weight from Monte Carlo
2. Re-run Phase 3 HWA training with real noise profile
3. Generate the final thesis noise profile graph: Error vs. Output Code
4. Generate the final accuracy comparison with real hardware noise
5. Compute CSNR and compare to the Yoshioka paper's results (they showed CNNs robust down to ~15 dB)

### Verifiable Deliverables


| #   | Deliverable               | Pass Criteria                                                                                          |
| --- | ------------------------- | ------------------------------------------------------------------------------------------------------ |
| 5.1 | Noise profile graph       | Error vs. Output Code from real Monte Carlo — thesis checklist item                                    |
| 5.2 | Final accuracy comparison | Real-noise version of the Phase 3 bar chart                                                            |
| 5.3 | CSNR measurement          | Reported in dB, compared to literature                                                                 |
| 5.4 | End-to-end validation     | Test vectors from Python → Cadence stimuli → analog output → Python comparison (section 6.3 of thesis) |


---

## Dependency Map

```
Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4
  (Baseline)   (Noise Model)  (HWA Training)  (Distillation)
                                                    │
                                                    ▼
                                              Phase 5
                                         (Real Hardware Noise)
                                                    ▲
                                                    │
                                    ┌────────────────┘
                                    │
                           [SRAM Cell Fix]
                                    │
                           [Array Assembly]
                                    │
                           [Layout + PEX]
                                    │
                           [Monte Carlo Sims]
```

Phases 1–4 are fully parallel with all hardware work. Phase 5 is the single merge point.

---

## Tools and Libraries

- **PyTorch** (≥2.0): Model training, custom layers, autograd
- **torchvision**: MNIST dataset
- **matplotlib / seaborn**: Thesis-quality plots
- **numpy**: Numerical computation for MAC simulator
- **pandas**: Results logging and comparison tables
- **Jupyter notebooks** (optional): For interactive development and documentation

No Cadence, no Virtuoso, no PDK access required for Phases 1–4.

---

## Notes on Scope

This plan uses MNIST with a micro-MLP because that's what the thesis commits to in section 6.3. This is *not* building an analog foundation model for an LLM — the Rasch paper's methodology is being adapted to an undergraduate-thesis-appropriate scale. The scientific contribution is demonstrating that HWA training improves inference accuracy on *your specific 65nm C-2C hardware*, not achieving state-of-the-art on a benchmark.

The comparison chart (Phase 3, deliverable 3.4) is the money plot for the thesis. Everything else supports getting there.

---

## Appendix A: UMC 65nm PDK Document Map

The following PDK documents have been reviewed and mapped to specific thesis tracks. This appendix serves as a reference for all team members to know which document answers which question.

### Document Inventory


| Document                   | DSM No.                                                   | What It Contains                                                                                                                       | Primary Track          |
| -------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| G-01 DSM (Ver 1.6_P1)      | G-01-LOGIC/MIXED_MODE65N-SP/LOW_K-DSM                     | Process overview, device types, Vt flavors, key design rules, metal stack options                                                      | All tracks (reference) |
| G-04 INTERCAP (Ver 1.2_P2) | G-04-LOGIC/MIXED_MODE65N-LOW_K-INTERCAP                   | Full interconnect capacitance model: dielectric stack, metal thicknesses, parasitic capacitance lookup tables for all 38 metal options | Circuit + HWA training |
| G-9FD FDK (Ver B04_PB)     | G-9FD-LOGIC/MIXED_MODE65N-SP/LOW_K/UMK65FDKSPC00000OA-FDK | Cadence Virtuoso FDK contents, PCell device list, DRC/LVS deck versions, EDA tool versions                                             | Layout track           |


### G-01 DSM — Key Extracted Parameters

**Device choices for the SRAM cell (core devices at VDD = 1.1V):**


| Parameter             | SP_LVT      | SP_RVT    | SP_HVT     | SP_SHVT     |
| --------------------- | ----------- | --------- | ---------- | ----------- |
| Vt_sat N/P (V)        | 0.175/0.135 | 0.22/0.18 | 0.29/0.255 | 0.350/0.325 |
| Idsat N/P (µA/µm)     | 1070/520    | 1005/480  | 860/410    | 730/340     |
| Gate delay (ps/stage) | 4.5         | 5.2       | 6.3        | 8.7         |
| Lmin (µm)             | 0.06        | 0.06      | 0.06       | 0.06        |


**Design rule minimums (core 1.1V devices):**


| Layer       | Width (µm) | Space (µm) | Pitch (µm) |
| ----------- | ---------- | ---------- | ---------- |
| Diffusion   | 0.08       | 0.11       | 0.19       |
| Poly        | 0.06       | 0.13       | 0.19       |
| Contact     | 0.09       | 0.11       | 0.20       |
| Metal1      | 0.09       | 0.09       | 0.18       |
| M2–M6 (1X)  | 0.10       | 0.10       | 0.20       |
| M7–M8 (2X)  | 0.20       | 0.20       | 0.40       |
| M9–M10 (4X) | 0.40       | 0.40       | 0.80       |


**Implications for the SRAM cell:**

- Minimum NMOS/PMOS width = 0.08 µm (80 nm). This is the pull-up PMOS sizing floor.
- Access transistors at ~1.5× → W ≈ 0.12 µm (120 nm)
- Pull-down NMOS at ~2× → W ≈ 0.16–0.20 µm (160–200 nm)
- All at Lmin = 60 nm for the 6T storage core.

**Vt flavor selection:**

- SP_RVT is the safe default for 6T SRAM: balanced Vt gives good read SNM without excessive leakage.
- SP_HVT could be considered for the 6T storage core to reduce standby leakage (SRAM weights are stationary during inference), at the cost of slower write speed — but write speed is not critical since weight loading happens infrequently vs. compute cycles.

### G-04 INTERCAP — Key Extracted Parameters

**Metal stack for the G-4M variant (1P10M-SP/MIM/LOW_K, the variant you need for MIM capacitors):**


| Conductor                | Thickness (µm) typ | Thickness ±15% | Width min (µm) |
| ------------------------ | ------------------ | -------------- | -------------- |
| Metal1                   | 0.180              | 0.153–0.207    | 0.090          |
| Metal2–Metal6            | 0.220 each         | 0.187–0.253    | 0.100          |
| Metal7                   | 0.360              | 0.306–0.414    | 0.200          |
| MMCBP (MIM bottom plate) | 0.110              | —              | —              |
| MMCTP (MIM top plate)    | 0.060              | —              | —              |
| Metal8                   | 0.360              | 0.306–0.414    | 0.200          |
| Metal9                   | 0.800              | 0.680–0.920    | 0.400          |
| Metal10                  | 0.800              | 0.680–0.920    | 0.400          |


**Dielectric stack (G-4M variant, critical for parasitic estimation):**


| Layer                               | Thickness (µm) | k (dielectric constant) | Notes                                |
| ----------------------------------- | -------------- | ----------------------- | ------------------------------------ |
| IMD6A (etch stop above M6)          | 0.05           | 5                       | High-k etch stop                     |
| IMD6B                               | 0.11           | 2.9                     | Low-K ILD                            |
| IMD6C                               | 0.22           | 2.9                     | Low-K ILD                            |
| IMD7A (etch stop above M6/below M7) | 0.06           | 7                       | High-k etch stop                     |
| IMD7B                               | 0.26           | 3.7                     | FSG                                  |
| IMD7C (M7 trench fill)              | 0.36           | 3.7                     | FSG                                  |
| IMD8A (etch stop above M7)          | 0.05           | 7                       | High-k etch stop                     |
| IMD81                               | 0.035          | 7                       | Thin high-k                          |
| IMD8B                               | 0.51           | 3.7                     | FSG — MIM cap sits between M7 and M8 |
| IMD8C (M8 trench fill)              | 0.36           | 3.7                     | FSG                                  |


**How to compute MIM bottom-plate parasitic from this data:**

The MIM capacitor sits between Metal7 and Metal8 in the G-4M variant. The bottom plate (MMCBP, 0.110 µm thick) has parasitic capacitance to Metal7 below it, through the dielectric between them.

The relevant dielectric layers between the MIM bottom plate and M7 top surface are:

- IMD8A: 0.05 µm, k = 7
- IMD81: 0.035 µm, k = 7

Total separation ≈ 0.085 µm through high-k dielectric (k ≈ 7).

Meanwhile, the MIM capacitor itself (between MMCBP and MMCTP) has an insulator thickness of approximately 0.060 µm (the MMCTP thickness represents the top plate, but the MIM dielectric is the insulator between them — the actual MIM insulator thickness and k value must come from the SPICE capacitor model document G-05, which is not among the uploaded files).

**Bottom-plate parasitic ratio estimate:** If the MIM insulator has k ≈ 7–10 (typical SiN for MIM) at ~20–30 nm thickness, and the bottom-plate parasitic path has k ≈ 7 at ~85 nm, then:

- Cp/Cmim ≈ (k_parasitic / t_parasitic) / (k_mim / t_mim)
- Cp/Cmim ≈ (7 / 0.085) / (7 / 0.025) ≈ 82.4 / 280 ≈ 0.29 (~29%)

This aligns with the thesis's ~20–50% range estimate and gives a more precise operating point of roughly **25–35%** for the Phase 2 parasitic sweep.

**What you still need (not in these uploads):**

- G-05 SPICE model documents — contain the actual MIMCAP device model (capacitance density in fF/µm²), mismatch model (σ of cap value vs. area), and the exact MIM insulator thickness and k.
- G-02 EDR (Electrical Design Rules) — min/max MIM cap sizes, spacing rules.
- G-03 TLR (Topological Layout Rules) — physical layout rules for DRC-clean MIM cap placement.

### G-9FD FDK — Key Extracted Parameters

**Available capacitor PCells in the FDK:**


| PCell Name                      | Type            | Notes                                                  |
| ------------------------------- | --------------- | ------------------------------------------------------ |
| MIMCAPS_20F_NWELL_RFKF          | MIM over N-well | For analog/RF, ~20 fF/µm² density (inferred from name) |
| MIMCAPS_20F_PSUB_RFKF           | MIM over P-sub  | Same density, different substrate                      |
| MIMCAPS_20F_M1_RFKF             | MIM over Metal1 | May have different parasitic characteristics           |
| MIMCAPS_20F_MM                  | MIM mixed-mode  | Likely the one for your design                         |
| MOMCAPS_SY_MMKF                 | MOM symmetric   | Metal-oxide-metal finger cap, no extra mask            |
| MOMCAPS_AS_MMKF                 | MOM asymmetric  | MOM variant                                            |
| MOMCAPS_ARRAY_VP3/VP4/VP5_RFVCL | MOM array       | Multi-via-pair MOM caps                                |


**Key decision: MIM vs. MOM for the C-2C ladder**

Wang's original design (22nm FinFET) used MOM (metal-oxide-metal) capacitors formed with BEOL structures placed above the SRAM array. In your 65nm process, you have both MIM and MOM options.

- **MIM (MIMCAPS_20F_MM):** Higher density (~~20 fF/µm²), dedicated mask layer, good matching, but incurs the bottom-plate parasitic discussed above (~~25–35%). Requires the MIM metal option in the process (G-4M variant).
- **MOM (MOMCAPS_SY_MMKF):** No extra mask, uses standard metal fingers, lower parasitic to substrate, but lower density and matching depends on lithographic precision. Can be placed on any metal layer.

The design uses MOM caps, not MIM. The target PCell is `MOMCAPS_SY_MMKF` (symmetric MOM finger capacitor). Characterized values: C ≈ 15.1 fF, 2C ≈ 30.2 fF. All MIM-specific parasitic calculations in Phase 2 and the dielectric stack analysis in Appendix A are superseded by MOM parasitics. The G-05 MOMCAP SPICE model (not G-05 MIMCAP) is the outstanding missing document for Phase 2 calibration.

The "20F" likely means ~20 fF/µm² capacitance density, which means:

- For a unit capacitor C = 50 fF, you need an area of 50/20 = 2.5 µm²
- For 2C = 100 fF (the serial cap), you need 5.0 µm²
- These are small enough to tile above an SRAM array without dominating area.

**EDA tool versions confirmed by the FDK:**


| Tool             | Version                     |
| ---------------- | --------------------------- |
| Cadence Virtuoso | IC6.1.5.500.1               |
| Simulation       | Spectre (via ADE) or HSPICE |
| DRC              | Calibre (Mentor)            |
| LVS              | Calibre (Mentor)            |


---

## Appendix B: What PDK Documents Are Still Missing

The uploaded documents cover the process overview (G-01), interconnect parasitics (G-04), and FDK contents (G-9FD). For a complete picture, the team should locate and reference:


| Document                            | Why You Need It                                                                    | Who Needs It                                                     |
| ----------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| G-02 EDR (Electrical Design Rules)  | Min/max MIM cap dimensions, spacing rules, ESD rules, latch-up rules               | Farah, Armia (layout)                                            |
| G-03 TLR (Topological Layout Rules) | DRC-clean layout rules for transistors, caps, wells, contacts                      | Farah, Armia (layout)                                            |
| G-05 SPICE Models (MOMCAP)          | Actual MOM cap value (fF), mismatch σ vs. finger count, temperature coefficients — MOMCAPS_SY_MMKF specifically | Omar (Phase 2 calibration), Abanoub/John/Mariam (ADC/DAC sizing) |
| G-05 SPICE Models (MOSFET)          | Transistor SPICE models for simulation — you already have these loaded in Virtuoso | Already in use                                                   |


The G-05 MOMCAP SPICE model is the single most important missing document for calibrating the Phase 2 noise model. It will give you the exact mismatch statistics (σ_ΔC/C vs. area) that replace the generic Gaussian noise placeholder. If you can find `G-05SP-MIXED_MODE/RFCMOS65N-MIM/LOW_K-SPICE/SPECTRE/CAPACITOR` in your PDK distribution, that's the file.