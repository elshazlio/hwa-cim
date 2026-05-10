# HWA-CiM: Hardware-Aware Training (MNIST)

Implements the pipeline described in `HWA_Training_Pipeline_Plan.md`: FP32/INT4 baseline, C-2C MAC parity, parasitic/noise models (MOM ladder defaults), HWA training, distillation, and a Phase-5 CSV hook for Monte Carlo noise profiles.

## Requirements

- **Python 3.10–3.12** (matches `requires-python` and stable PyTorch wheels).
- **MNIST** downloads automatically on first train run into `data/` (ignored by git).

## Setup — macOS / Linux

From the repository root (folder that contains `pyproject.toml`):

```bash
cd "/path/to/Thesis HW Codesign"
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,gui]"
```

## Setup — Windows (PowerShell)

From the repository root:

```powershell
cd "C:\path\to\Thesis HW Codesign"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,gui]"
```

If execution policy blocks activation, run once: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

**CMD:** use `\.venv\Scripts\activate.bat` instead of `Activate.ps1`.

## Run — dashboard (optional GUI)

With the venv **activated**, from the repo root:

```bash
hwa-dashboard
```

Same command on Windows after PowerShell activation. Streamlit prints a local URL (usually `http://localhost:8501`).

Equivalent without the console script:

```bash
python -m streamlit run src/hwa_gui/Home.py
```

## Commands (CLI)

| Phase | Command |
|-------|---------|
| 1 Baseline | `hwa-train-baseline --out-dir results/run_baseline` |
| 2 Noisy eval | `hwa-eval-noisy --checkpoint results/run_baseline/best.pt --gamma 0.02 --seeds 10` |
| 2 Gamma sweep | `hwa-sweep-phase2 --checkpoint results/run_baseline/best.pt` |
| 3 HWA train | `hwa-train-hwa --out-dir results/run_hwa --gamma 0.02 --alpha 3.0` |
| 3 Sweep | `hwa-sweep-hwa --out-dir results/sweep_hwa` |
| 4 Distill | `hwa-train-distill --out-dir results/run_distill --teacher-epochs 30` |
| Thesis chart | `hwa-plot-thesis --baseline-dir results/run_baseline --hwa-checkpoint results/run_hwa/best.pt --noisy-eval-json results/run_baseline/noisy_eval.json --out results/figures/thesis_bars.png` |
| Parasitic sweep | `hwa-plot-parasitic --out results/figures/parasitic_sweep.png` |

Run `hwa-eval-noisy` on the Phase 1 checkpoint first to produce `noisy_eval.json` for the middle bar.

## Outputs

- Checkpoints, `metrics.json`, CSV summaries under `results/`
- Figures under `results/figures/`
- Phase 5: CSV columns `input_code,ideal_output,mean_output,sigma` (optional `CSNR_dB`); use `hwa-train-hwa --noise-mode csv --noise-profile path/to/profile.csv`

## Tests

With venv activated and repo root as cwd:

```bash
pytest
```

Or: `python -m pytest` (macOS, Linux, or Windows).
