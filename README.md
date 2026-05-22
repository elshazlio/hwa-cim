# HWA-CiM: Hardware-Aware Training (MNIST)

Software pipeline for **charge-domain SRAM compute-in-memory** research aligned with **Analog Foundation Models** (AFM)–style training [Rasch et al., 2025]: a small **micro-MLP** on **MNIST**, **INT4** weights, **unsigned 4-bit activations**, **4-bit ADC** modeling per linear layer, and **synthetic Gaussian weight noise** (γ · max|W|) until **Phase 5**, when a **Monte Carlo CSV** from Cadence can drive HWA training.

The stack also supports an **optional schematic-style MAC model**: population-dependent **effective gain** and a **dense-regime offset** on top of ideal INT4 multiply–accumulate (`c2c_mac(..., hardware_aware=True)`), wired into **HWA training** by default (`hwa-train-hwa` / `hwa-train-distill`) while **Phase 2 noisy eval** stays on the legacy path unless you opt in. See **`docs/agdr/AgDR-0001-hardware-aware-mac-calibration.md`** for the decision record.

**Detailed methodology and thesis checklist:** [`background_info/HWA_Training_Pipeline_Plan.md`](background_info/HWA_Training_Pipeline_Plan.md)  
**Streamlit lab (GUI) — install, tour, partial runs:** [`GUI_RUN.md`](GUI_RUN.md)  
**Agent Decision Records (architecture / calibration choices):** [`docs/agdr/README.md`](docs/agdr/README.md)  
**Software vs hardware roadmap (follow-ups):** [`docs/software_mission_followups.md`](docs/software_mission_followups.md)

---

## Status (software)

| Phase | Scope | In repo |
| ----- | ----- | ------- |
| **1** | FP32 train, INT4 PTQ eval (**ideal + hardware-shaped**), `c2c_mac` parity vs `nn.Linear` (ideal path only) | Yes (`hwa-train-baseline`) |
| **2** | NoisyQuantLinear + γ noise (train mode), ADC STE, parasitic ladder plots | Yes (`hwa-eval-noisy`, `hwa-sweep-phase2`, `hwa-plot-parasitic`) |
| **3** | HWA training (noise + clipping + STE); **hardware-aware forward on by default** (`--no-hardware-aware` to match pre-change behavior) | Yes (`hwa-train-hwa`, `hwa-sweep-hwa`) |
| **4** | Teacher–student distillation + noisy student (same `--no-hardware-aware` flag) | Yes (`hwa-train-distill`) |
| **4.5** | **Surrogate Monte Carlo** (user-defined Gaussian parametric variation) — wide VIVA sweep parser + thesis plots | Yes (`hwa-surrogate-mc`, `hwa-plot-surrogate-mc`; **AgDR-0005**) — **not** final Phase 5 |
| **5** | Real noise profile from **PEX + Monte Carlo** → CSV → `--noise-mode csv` | Hook implemented; **waiting on hardware MC export** |
| **PEX cal** | Pre-MC **deterministic** gain from Maestro `/OA_Charge` (not a σ noise profile) | Yes (`hwa-maestro-pex`; **AgDR-0004**) — **does not replace Phase 5** |

The same phase order is exposed as tabs on the Streamlit **Run** page; see [`GUI_RUN.md`](GUI_RUN.md).

Sample checkpoints, `metrics.json`, sweeps, and figures are under **`results/`** (committed as examples; re-run CLI commands to regenerate on your machine).

---

## Requirements

- **Python 3.10–3.13** (`requires-python` in `pyproject.toml` is `>=3.10,<3.14`).
- **PyTorch** (CPU or CUDA); MNIST is downloaded on first training run into **`data/`** (gitignored).

Install **editable** with optional **`dev`** (pytest) and **`gui`** (Streamlit + Plotly):

```text
pip install -e ".[dev,gui]"
```

Minimal install without dashboard: `pip install -e ".[dev]"`.  
Dashboard without pytest extras: `pip install -e ".[gui]"`.

---

## Setup — macOS / Linux

**Use the real folder where you cloned this repo** (it must contain `pyproject.toml`).  
If `cd` fails, **do not** run `pip install` yet — from your home directory, `pip install -e .` can error with something like `file:///Users/yourname does not appear to be a Python project`.

**Paste safety:** if you copy a line that ends with a “comment” meant for humans but the leading `#` is missing, **zsh can treat `<` as input redirection** and you may see `no such file or directory` for `--`. Put real shell comments on their own lines starting with `#`, or omit those hints entirely.

Example when the repo lives under `Documents` with spaces in the name (change the path if yours differs):

```bash
cd "$HOME/Documents/My Projects/Thesis HW Codesign"
ls pyproject.toml
```

If `ls` says there is no such file, fix the `cd` path before continuing. Do not run `pip install` from your home directory.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,gui]"
```

On macOS you can also type `cd ` in Terminal, then **drag the project folder** into the window and press Enter (quotes are added automatically if needed).

---

## Setup — Windows (PowerShell)

```powershell
cd "C:\Users\YOURNAME\Documents\Thesis HW Codesign"
Get-Item pyproject.toml
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,gui]"
```

If `Get-Item` fails, fix the `cd` path first.

If activation is blocked: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

**CMD:** `\.venv\Scripts\activate.bat`

---

## Quick start (recommended order)

From repo root with venv activated:

```bash
hwa-train-baseline --data-dir data --out-dir results/run_baseline
hwa-eval-noisy --checkpoint results/run_baseline/best.pt --data-dir data --out results/run_baseline/noisy_eval.json --gamma 0.02 --seeds 10
hwa-train-hwa --data-dir data --out-dir results/run_hwa --gamma 0.02 --alpha 3.0
hwa-plot-thesis --baseline-dir results/run_baseline --hwa-checkpoint results/run_hwa/best.pt --noisy-eval-json results/run_baseline/noisy_eval.json --out results/figures/thesis_bars.png
```

Optional: **teacher + student** distillation — `hwa-train-distill --out-dir results/run_distill`.

Use **`--device cuda`** on any command that trains or evaluates if a GPU is available.

**Match legacy HWA training (no schematic gain/offset on the noisy layers):**

```bash
hwa-train-hwa --no-hardware-aware ...
hwa-train-distill --no-hardware-aware ...
```

---

## Dashboard (optional)

The **HWA-CiM Lab** is a Streamlit front-end (`src/hwa_gui/`) around the same phases as the CLI: **Run** (tabbed jobs + live log), **Results**, **Compare**, **Charts** (Plotly), and **Hardware profiles** (synthetic, Maestro PEX, corners, **Phase 4.5 surrogate MC**, Phase 5 MC CSV). Sidebar page names (**Home**, **Run**, …) are declared in `Home.py` via **`st.navigation`** (see **AgDR-0002**); the built-in nav sits at the top of the sidebar, with progress ✓/○ hints below. Optional **`[gui]`** requires **Streamlit ≥ 1.52**.

```bash
hwa-dashboard
```

Opens the app (typically `http://localhost:8501`). Equivalent from repo root:

```bash
python -m streamlit run src/hwa_gui/Home.py
```

Full walkthrough, one-job-at-a-time behavior, and tab-to-phase mapping: **[`GUI_RUN.md`](GUI_RUN.md)**.

---

## CLI reference

Console scripts are defined in `pyproject.toml`; after install they are on `PATH` inside the venv.

| Phase | Command | Notes |
| ----- | ------- | ----- |
| **1** Baseline | `hwa-train-baseline --out-dir results/run_baseline` | Writes `best.pt`, `metrics.json`: `fp32_test_accuracy`, **`int4_ptq_test_accuracy_ideal`**, **`int4_ptq_test_accuracy_hardware`**, `parity_linear_c2c_max_abs_error`, `epochs`, `seed`. |
| **2** Noisy eval | `hwa-eval-noisy --checkpoint …/best.pt --out …/noisy_eval.json` | Middle-bar input for thesis plot. |
| **2** Γ sweep | `hwa-sweep-phase2 --checkpoint …/best.pt --out-dir results/phase2_sweep` | CSV/JSON under `out-dir`. |
| **3** HWA | `hwa-train-hwa --out-dir results/run_hwa --gamma 0.02 --alpha 3.0` | Best noisy-validation checkpoint. Adds **`hardware_aware`** to `metrics.json`. Use **`--no-hardware-aware`** for the old training behavior. |
| **3** Grid | `hwa-sweep-hwa --out-dir results/sweep_hwa` | γ × α grid (long run). Honors **`--no-hardware-aware`**. |
| **4** Distill | `hwa-train-distill --out-dir results/run_distill` | Teacher + noisy student. **`--no-hardware-aware`** supported. |
| **Fig** Thesis bars | `hwa-plot-thesis --baseline-dir … --hwa-checkpoint …/best.pt [--noisy-eval-json …]` | Middle proxy uses **`int4_ptq_test_accuracy_ideal`**, then legacy `int4_ptq_test_accuracy` / `int8_ptq_test_accuracy` if present. |
| **Fig** Parasitic | `hwa-plot-parasitic --out results/figures/parasitic_sweep.png` | MOM-oriented sweep **0–20%** (`--pdk-marker` optional). |
| **PEX cal** | `hwa-maestro-pex --manifest config/maestro_pex.yaml --write-calibration config/calibration_pex.yaml` | PNGs under `results/maestro_pex/figures/`; **not** Phase 5 MC (AgDR-0004). |
| **4.5 Surrogate MC** | `hwa-surrogate-mc --all-defaults` | Parses wide VIVA sweeps → `results/surrogate_mc/`; **not** UMC-certified MC (AgDR-0005). |
| **4.5 Plots** | `hwa-plot-surrogate-mc` | PNGs under `results/plots/03_surrogate_mc/`. |

### Phase 4.5 — Surrogate Monte Carlo (user-defined Gaussian parametric variation)

**What this is:** A **stopgap between PEX calibration and true Phase 5**. Native UMC Monte Carlo was blocked; the team ran **TT models** with **user-defined Gaussian/grid perturbations** of PDK delta parameters (`umc_mc_dvth0_*`, `umc_mc_d_c1_vp`, …). Sigma is estimated from FF/SS corner deltas (÷3), not foundry-certified mismatch statistics.

**What this is not:**

- **Not** Phase 5 / `--noise-mode csv` (summaries lack per-`input_code` statistical profiles).
- **Not** UMC-certified Monte Carlo.
- **Not** proof that surrogate σ should drive HWA training today (Run → Phase 4.5 mode records provenance with **synthetic** γ only).

**Typical workflow:**

```bash
hwa-surrogate-mc --all-defaults
hwa-plot-surrogate-mc
```

Curated figures: `results/plots/03_surrogate_mc/` (dvth0 vs MOM cap spread, MOM cap sweep, PVT/PEX corner bars). Source CSVs: `stuff_from_cadence/manual_mc_*.csv`, `no_pex_oa_only_3_corners.csv`, `with_pex_oa_only_3_corners.csv`. Cadence flow: `monte_carlo_debugging/ADE_MAESTRO_MANUAL_MC.md`.

**Thesis wording (short):** *Phase 4.5 surrogate Monte Carlo with user-defined Gaussian parametric variation characterizes `/OA_Charge` sensitivity to MOM capacitance and threshold deltas; foundry statistical Phase 5 remains a separate CSV path.*

See **AgDR-0005** and `background_info/Surrogate_MC_Phase5_Readiness_Report.md`.

**Phase 5 (hardware CSV):**

```bash
hwa-train-hwa --noise-mode csv --noise-profile path/to/mc_profile.csv --out-dir results/run_hwa_mc ...
```

The loader expects columns such as `input_code`, `ideal_output`, `mean_output`, `sigma` (see `src/hwa_cim/noise.py`). With **`--noise-mode csv`**, training uses **per-code σ** on weights (nearest `input_code`) and on layer outputs when a profile is loaded. Optional columns: `weight_population`, `g_eff_measured`, `offset_measured`.

**MAC calibration YAML** (gain/offset defaults, plot operating point):

```bash
hwa-train-hwa --calibration-yaml config/calibration.yaml ...
```

Default file: [`config/calibration.yaml`](config/calibration.yaml). Override knobs without editing `src/hwa_cim/c2c.py` (see **AgDR-0003**).

### Maestro PEX calibration (pre-MC bridge — not a noise profile)

**What this is:** A **stopgap until Monte Carlo CSV exists**. Cadence Maestro/VIVA **no-PEX vs PEX** waveforms sample **`/OA_Charge`** at a manifest time and scale schematic MAC gains (`calibration_pex.yaml`). Training still uses **`--noise-mode synthetic`** (γ noise), not per-code σ from PEX.

**What this is not:**

- **Not** Phase 5 / `--noise-mode csv` (no `input_code`, `mean_output`, `sigma` table from MC).
- **Not** proof that “PEX hurt accuracy and HWA fixed it” on MNIST — `hwa-plot-thesis` stays **FP32 → INT4+noise (no HWA) → HWA**; it has no “PEX-only degradation” bar.
- **Not** power (W) or energy — only analog voltage / effective gain.

**Typical workflow:**

```bash
hwa-maestro-pex \
  --manifest config/maestro_pex.yaml \
  --out-dir results/maestro_pex \
  --write-calibration config/calibration_pex.yaml

hwa-train-hwa \
  --calibration-yaml config/calibration_pex.yaml \
  --noise-mode synthetic \
  --hardware-profile-mode maestro_pex \
  --gamma 0.02 --alpha 3.0 \
  --out-dir results/run_hwa_pex_calibrated
```

**What we observed (example run, γ=0.02):** noisy test accuracy **~93.9%** without HWA (Phase 2 on baseline checkpoint) → **~97.1%** with schematic HWA → **~97.0%** with PEX-calibrated HWA (essentially flat vs schematic HWA). PEX/no-PEX at 200.25 ns gave **relative_gain ≈ 0.91** on `/OA_Charge`. Use Maestro figures for **layout analog evidence**; cite ML gains from HWA vs no-HWA, not from PEX calibration alone.

**Thesis wording (short):** *Pre-MC Maestro PEX calibrates deterministic post-layout gain from `/OA_Charge` (inference testbench; read mode inactive). Statistical noise-aware training awaits Monte Carlo export to Phase 5 CSV.*

See **AgDR-0004** and `docs/software_mission_followups.md`.

---

## Limitations (read before citing numbers)

| Topic | What the repo does | What it is *not* |
| ----- | ------------------ | ---------------- |
| **Ideal vs hardware MAC** | `c2c_mac(..., hardware_aware=False)` matches dequant linear; parity test uses ideal only. | Not a SPICE netlist. |
| **HWA training forward** | `NoisyQuantLinear` uses float `F.linear` + INT4 popcount gain/offset (AgDR-0001), not integer `c2c_mac` in the backward path. | Not cycle-accurate tile MAC. |
| **Gain/offset model** | Row-wise population average over INT4 magnitudes; constants from schematic notes. | Not per-tile layout-accurate until refined. |
| **Parasitic plot marker** | Default **~17%** (`INTEGRATED_OPERATING_POINT`) is a **heuristic** from sparse G_eff, not extracted PDK silicon. | Do not cite as measured INL corner. |
| **Phase 5 CSV** | Per-code σ interpolation; extended columns optional. | Full MC + PEX closure still hardware-owned. |
| **Maestro PEX path** | `/OA_Charge` sample → scaled `g_eff`; PNGs in `results/maestro_pex/figures/`. | **Not** a noise profile; **not** MNIST accuracy uplift vs schematic HWA in our runs; no power model. |
| **Phase 4.5 surrogate** | Wide VIVA parser + spread/sweep plots; `profile_kind` in summary CSV. | **Not** foundry MC; **not** `NoiseProfileCSV` without `input_code`. |
| **8 verified vectors** | Test stub **skipped** until mV ↔ software scaling is agreed (`tests/test_verified_vectors.py`). | Do not use draft roadmap conversion as regression. |
| **INT4 metrics** | Phase 1 reports **`int4_ptq_test_accuracy_ideal`** and **`…_hardware`** separately. | A single “INT4 accuracy” without the suffix is ambiguous. |

---

## Outputs

| Path | Contents |
| ---- | -------- |
| `results/run_baseline/` | FP32 checkpoint, `metrics.json` (dual INT4 PTQ keys), optional `noisy_eval.json` |
| `results/run_hwa/` | HWA checkpoint, `metrics.json` (includes `hardware_aware`) |
| `results/run_distill/` | Teacher/student checkpoints, `metrics.json` (includes `hardware_aware`) |
| `results/phase2_sweep/` | `gamma_sweep.csv`, `.json` |
| `results/figures/` | `thesis_bars.png` (FP32 / INT4+noise / HWA — **not** PEX-isolated bars), `parasitic_sweep.png`, optional `pex_hwa_honest_comparison.png` |
| `results/maestro_pex/` | `maestro_pex_summary.csv`, `maestro_pex_metrics.json`, `figures/*.png` |
| `results/surrogate_mc/` | Phase 4.5 per-sweep points + summary CSV/JSON (`cap_sweep`, `dvth0_sweep`, `pvt_pex_corners`) |
| `results/plots/03_surrogate_mc/` | Surrogate sensitivity, MOM cap sweep, PVT/PEX corner bars (see `results/plots/README.md`) |
| `results/run_hwa_pex_calibrated/` | Example HWA run with `calibration_pex.yaml` + `hardware_profile_mode: maestro_pex` |

If Matplotlib warns about a non-writable config dir (e.g. CI), set:

```bash
export MPLCONFIGDIR="$PWD/results/.mplconfig"
mkdir -p "$MPLCONFIGDIR"
```

(`results/.mplconfig` is gitignored.)

---

## Hardware track (short)

**Cadence (Virtuoso) schematic:** The team target is **UMC 65 nm** full-custom **SRAM CiM**: **4×4** array, **decoder**, **DAC**, and **SAR ADC** integrated at schematic level with **C-2C** charge-domain MAC; the **SAR comparator** is currently an **ideal library** block for schedule, to be swapped for **full custom** later. **Software** uses **MOMCAPS_SY_MMKF**-style defaults for the **parasitic ladder toy model** (`src/hwa_cim/c2c.py`); see `background_info/HWA_Training_Pipeline_Plan.md` for MOM vs MIM notes.

**This repository** consumes **simulation-derived CSV** (μ, σ, codes) for **Phase 5** statistical noise — not raw VIVA waveforms as a σ profile. **Maestro PEX** (`hwa-maestro-pex`) is a **deterministic gain bridge** from `/OA_Charge`. **Phase 4.5** (`hwa-surrogate-mc`) parses **wide VIVA surrogate sweeps** for thesis sensitivity and PVT/PEX corner evidence — labeled *Surrogate Monte Carlo with user-defined Gaussian parametric variation*, distinct from foundry MC. Pre-layout **gain/offset** knobs (`hardware_aware`) mirror schematic verification numbers; **Monte Carlo on PEX** with per-code exports remains the thesis-grade Phase 5 path. Roadmap: **`background_info/Bird's Eye View of Our Thesis.md`**. Follow-ups: **`docs/software_mission_followups.md`**.

---

## Tests

```bash
pytest
```

(`python -m pytest` works on all platforms; `pythonpath` is set in `pyproject.toml`.)

---

## License

MIT — see `pyproject.toml`.
