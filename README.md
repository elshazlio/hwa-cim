# HWA-CiM: Hardware-Aware Training (MNIST)

Implements the pipeline described in `HWA_Training_Pipeline_Plan.md`: FP32/INT8 baseline, C-2C MAC parity, parasitic/noise models, HWA training, distillation, and a Phase-5 CSV hook for Monte Carlo noise profiles.

## Setup

Use **Python 3.10–3.12** (PyTorch wheels). Example:

```bash
cd "Thesis HW Codesign"
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,gui]"
```

**GUI (optional):** after installing with `gui`, run `hwa-dashboard` to open the lab dashboard in your browser.

## Commands

| Phase | Command |
|-------|---------|
| 1 Baseline | `hwa-train-baseline --out-dir results/run_baseline` |
| 2 Noisy eval | `hwa-eval-noisy --checkpoint results/run_baseline/best.pt --gamma 0.02 --seeds 10` |
| 2 Gamma sweep | `hwa-sweep-phase2 --checkpoint results/run_baseline/best.pt` |
| 3 HWA train | `hwa-train-hwa --out-dir results/run_hwa --gamma 0.02 --alpha 3.0` |
| 3 Sweep | `hwa-sweep-hwa --baseline-checkpoint results/run_baseline/best.pt --out-dir results/sweep_hwa` |
| 4 Distill | `hwa-train-distill --out-dir results/run_distill --teacher-epochs 30` |
| Thesis chart | `hwa-plot-thesis --baseline-dir results/run_baseline --hwa-checkpoint results/run_hwa/best.pt --noisy-eval-json results/run_baseline/noisy_eval.json --out results/figures/thesis_bars.png` |
| Parasitic sweep | `hwa-plot-parasitic --out results/figures/parasitic_sweep.png` |

Run `hwa-eval-noisy` on the Phase 1 checkpoint first to produce `noisy_eval.json` for the middle bar.

## Outputs

- Checkpoints, `metrics.json`, CSV summaries under `results/`
- Figures under `results/figures/`
- Phase 5: CSV columns `input_code,ideal_output,mean_output,sigma` (optional `CSNR_dB`); use `hwa-train-hwa --noise-mode csv --noise-profile path/to/profile.csv`

## Tests

```bash
pytest
```
