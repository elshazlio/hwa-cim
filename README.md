# HWA-CiM: Hardware-Aware Training (MNIST)

Software pipeline for **charge-domain SRAM compute-in-memory** research aligned with **Analog Foundation Models** (AFM)–style training [Rasch et al., 2025]: a small **micro-MLP** on **MNIST**, **INT4** weights, **unsigned 4-bit activations**, **4-bit ADC** modeling per linear layer, and **synthetic Gaussian weight noise** (γ · max|W|) until **Phase 5**, when a **Monte Carlo CSV** from Cadence can drive HWA training.

**Detailed methodology and thesis checklist:** [`HWA_Training_Pipeline_Plan.md`](HWA_Training_Pipeline_Plan.md)  
**Streamlit dashboard notes:** [`GUI_RUN.md`](GUI_RUN.md)

---

## Status (software)

| Phase | Scope | In repo |
| ----- | ----- | ------- |
| **1** | FP32 train, INT4 PTQ eval, `c2c_mac` parity vs `nn.Linear` | Yes (`hwa-train-baseline`) |
| **2** | NoisyQuantLinear + γ noise (train mode), ADC STE, parasitic ladder plots | Yes (`hwa-eval-noisy`, `hwa-sweep-phase2`, `hwa-plot-parasitic`) |
| **3** | HWA training (noise + clipping + STE) | Yes (`hwa-train-hwa`, `hwa-sweep-hwa`) |
| **4** | Teacher–student distillation + noisy student | Yes (`hwa-train-distill`) |
| **5** | Real noise profile from **PEX + Monte Carlo** → CSV → `--noise-mode csv` | Hook implemented; **waiting on hardware MC export** |

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

---

## Setup — macOS / Linux

From the repository root (directory that contains `pyproject.toml`):

```bash
cd "/path/to/Thesis HW Codesign"
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,gui]"
```

---

## Setup — Windows (PowerShell)

```powershell
cd "C:\path\to\Thesis HW Codesign"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,gui]"
```

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

---

## Dashboard (optional)

```bash
hwa-dashboard
```

Opens the Streamlit lab UI (typically `http://localhost:8501`). Equivalent:

```bash
python -m streamlit run src/hwa_gui/Home.py
```

---

## CLI reference

Console scripts are defined in `pyproject.toml`; after install they are on `PATH` inside the venv.

| Phase | Command | Notes |
| ----- | ------- | ----- |
| **1** Baseline | `hwa-train-baseline --out-dir results/run_baseline` | Writes `best.pt`, `metrics.json` (`fp32_test_accuracy`, `int4_ptq_test_accuracy`, parity error). |
| **2** Noisy eval | `hwa-eval-noisy --checkpoint …/best.pt --out …/noisy_eval.json` | Middle-bar input for thesis plot. |
| **2** Γ sweep | `hwa-sweep-phase2 --checkpoint …/best.pt --out-dir results/phase2_sweep` | CSV/JSON under `out-dir`. |
| **3** HWA | `hwa-train-hwa --out-dir results/run_hwa --gamma 0.02 --alpha 3.0` | Best noisy-validation checkpoint. |
| **3** Grid | `hwa-sweep-hwa --out-dir results/sweep_hwa` | γ × α grid (long run). |
| **4** Distill | `hwa-train-distill --out-dir results/run_distill` | Teacher + noisy student. |
| **Fig** Thesis bars | `hwa-plot-thesis --baseline-dir … --hwa-checkpoint …/best.pt [--noisy-eval-json …]` | Uses `int4_ptq_test_accuracy` with fallback to legacy `int8_ptq_test_accuracy`. |
| **Fig** Parasitic | `hwa-plot-parasitic --out results/figures/parasitic_sweep.png` | MOM-oriented sweep **0–20%** (`--pdk-marker` optional). |

**Phase 5 (hardware CSV):**

```bash
hwa-train-hwa --noise-mode csv --noise-profile path/to/mc_profile.csv --out-dir results/run_hwa_mc ...
```

The loader expects columns such as `input_code`, `ideal_output`, `mean_output`, `sigma` (see `src/hwa_cim/noise.py`). Training currently summarizes **`sigma`** via **`sigma_mean`** for injection scale — tighten this when richer per-code statistics are needed.

---

## Outputs

| Path | Contents |
| ---- | -------- |
| `results/run_baseline/` | FP32 checkpoint, `metrics.json`, optional `noisy_eval.json` |
| `results/run_hwa/` | HWA checkpoint, `metrics.json` |
| `results/run_distill/` | Teacher/student checkpoints, `metrics.json` |
| `results/phase2_sweep/` | `gamma_sweep.csv`, `.json` |
| `results/figures/` | `thesis_bars.png`, `parasitic_sweep.png` |

If Matplotlib warns about a non-writable config dir (e.g. CI), set:

```bash
export MPLCONFIGDIR="$PWD/results/.mplconfig"
mkdir -p "$MPLCONFIGDIR"
```

(`results/.mplconfig` is gitignored.)

---

## Hardware track (short)

Layout targets **UMC 65nm** SRAM CiM / **C-2C** ladder with **MOMCAPS_SY_MMKF** defaults in software. **PEX** netlists are for **Virtuoso/Spectre MC** — this repo consumes **simulation-derived tables** (CSV), not raw PEX text, unless you add tooling. See **Phase 5** and **layout scope** in `HWA_Training_Pipeline_Plan.md`.

---

## Tests

```bash
pytest
```

(`python -m pytest` works on all platforms; `pythonpath` is set in `pyproject.toml`.)

---

## License

MIT — see `pyproject.toml`.
