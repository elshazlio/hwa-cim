# Curated Plot Outputs

Presentation-ready plots are grouped here. Raw/provenance outputs are kept in their original run folders.

## 01_maestro_pex

- `01_oa_charge_overlay_zoom.png` — no-PEX vs PEX /OA_Charge with zoomed sampling window.
- `02_oa_charge_delta_zoom.png` — PEX minus no-PEX voltage delta.
- `03_sampled_calibration_summary.png` — sampled calibration point and relative gain.

Source data: `results/maestro_pex/maestro_pex_summary.csv`, `stuff_from_cadence/nopex1.csv`, `stuff_from_cadence/pex1.csv`.

## 02_hwa_context

- `01_hwa_vs_pex_context.png` — HWA vs PEX-cal context (main bars + HWA zoom inset + PEX scaling). Regenerate: `python -c "from hwa_cim.plots import run_hwa_pex_context_plot; run_hwa_pex_context_plot()"`.
- `02_noisy_mnist_pex_three_bar.png` — **FP32 (clean)** → **INT4+noise (no HWA)** → **INT4+noise (HWA, PEX cal. only)**. Regenerate: `python -c "from hwa_cim.plots import run_noisy_mnist_pex_three_bar; run_noisy_mnist_pex_three_bar()"`.

## Key numbers

- Sample time: 200.25 ns
- no-PEX /OA_Charge: 0.8274 V
- PEX /OA_Charge: 0.7554 V
- Relative gain: 0.9130x
- HWA schematic noisy accuracy: 97.09%
- HWA PEX-calibrated noisy accuracy: 97.02%

Do not caption these as Monte Carlo noise-profile results or power results.

## 03_surrogate_mc

Phase 4.5 — **Surrogate Monte Carlo with user-defined Gaussian parametric variation** (not UMC-certified MC, not final Phase 5).

- `01_sensitivity_spread_bars.png` — dvth0 grid vs MOM cap grid spread at 200.25 ns.
- `02_mom_cap_sweep.png` — `/OA_Charge` vs `umc_mc_d_c1_vp` (series by `d_cox_vp`).
- `03_pvt_pex_corner_bars.png` — no-PEX vs PEX per PVT corner (deterministic, not MC σ).

Regenerate parser artifacts:

```bash
hwa-surrogate-mc --all-defaults
```

Regenerate plots:

```bash
hwa-plot-surrogate-mc
```

Source data: `stuff_from_cadence/manual_mc_*.csv`, `no_pex_oa_only_3_corners.csv`, `with_pex_oa_only_3_corners.csv`.
