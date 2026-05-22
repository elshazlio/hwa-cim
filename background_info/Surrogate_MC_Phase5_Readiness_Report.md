# Surrogate Monte Carlo Findings and Phase-5 Readiness

Date: 2026-05-23  
Scope: UMC 65 nm SRAM-CIM `/OA_Charge` Maestro/VIVA exports, manual MC bypass, PEX/corner evidence, and software handoff for `hwa-cim`.

## Executive Summary

The current Cadence work produced a defensible **surrogate statistical variation** dataset, but it is **not foundry-certified UMC Monte Carlo** and it is **not yet true Phase 5** under the repo's current rules.

Recommended thesis label:

> **Surrogate Monte Carlo / user-defined Gaussian parametric variation at TT, with sigma estimated from PDK FF/SS corner deltas.**

Recommended phase label:

> **Phase 4.5 / Phase 5S (surrogate)**, not final Phase 5.

True Phase 5 remains defined by AgDR-0003 and `background_info/HWA_CIM_Required_Changes.md`: a per-code, per-pattern statistical CSV with `input_code`, `weight_population`, `ideal_output`, `mean_output`, `sigma`, and optionally `CSNR_dB`, produced from many statistical samples on the relevant extracted testbench.

The new results are still valuable. They establish:

- The manual wrapper flow can run PDK delta-parameter sweeps when native `mc_sp_*` Monte Carlo is blocked.
- The `/OA_Charge` inference node is **far more sensitive to MOM main-cap variation** than to the four tested MOS threshold knobs.
- The old 3-corner PEX/no-PEX exports remain useful as deterministic PVT/post-layout evidence, separate from statistical sigma.
- The software should support a **separate surrogate-profile mode** with clear provenance rather than rebranding surrogate data as official Phase 5 MC.

## Files and Evidence

Primary Cadence exports now in the repo:

- `stuff_from_cadence/manual_mc_4_var_1.csv`
  - Manual grid over four threshold-voltage delta variables:
    - `umc_mc_dvth0_n_splvt`
    - `umc_mc_dvth0_p_splvt`
    - `umc_mc_dvth0_n_sphvt`
    - `umc_mc_dvth0_p_sphvt`
  - Grid values: `[-sigma, 0, +sigma]`
  - Total points: `3^4 = 81`
- `stuff_from_cadence/manual_mc_2_var_cap_1.csv`
  - Manual grid over MOM capacitor delta variables:
    - `umc_mc_d_c1_vp`
    - `umc_mc_d_cox_vp`
  - Grid values: `[-0.067, 0, +0.067]`
  - Total points: `3^2 = 9`
- `stuff_from_cadence/no_pex_oa_only_3_corners.csv`
  - Nominal, FF, SS, and FNSP no-PEX `/OA_Charge`
- `stuff_from_cadence/with_pex_oa_only_3_corners.csv`
  - Nominal, FF, SS, and FNSP PEX `/OA_Charge`

Supporting flow/docs:

- `monte_carlo_debugging/umc65_manual_mc.scs`
- `monte_carlo_debugging/ADE_MAESTRO_MANUAL_MC.md`
- `monte_carlo_debugging/user_defined_statistical_variation.md`
- `docs/agdr/AgDR-0003-yaml-calibration-and-csv-noise.md`
- `docs/agdr/AgDR-0004-maestro-pex-calibration-path.md`

## What Was Actually Run

Native UMC Monte Carlo sections (`mc_sp_*`, encrypted `*_statistical_p`) were not used. Instead, the wrapper loads TT model sections and remaps PDK delta parameters to ADE global variables:

```spectre
include "l65sp_v132_mc.lib.scs" section=tt_sp_lvt11
include "l65sp_v132_mc.lib.scs" section=tt_sp_hvt11
include "l65sp_v132_mc.lib.scs" section=tt_65_momcaps
```

Then ADE sweeps or statistically varies variables such as:

- `umc_mc_dvth0_*` -> MOS threshold delta parameters
- `umc_mc_du0_*` -> mobility delta parameters
- `umc_mc_d_c1_vp`, `umc_mc_d_cox_vp` -> MOM capacitance delta parameters

The sigma values are **educated estimates** from PDK corner delta values, using the assumption:

```text
sigma ~= abs(FF/SS corner delta) / 3
```

That makes the flow a **surrogate Monte Carlo / statistical parametric variation** flow. It is mathematically MC-like when random sampling is used, but it is not UMC-certified MC unless the foundry confirms the sigma model.

## Measured Findings at 200.25 ns

The following values were sampled from `/OA_Charge` at `200.25 ns`, consistent with the existing Maestro PEX calibration point.

| Dataset | Points | Min (V) | Max (V) | Mean (V) | Std (V) | Spread (V) |
|---|---:|---:|---:|---:|---:|---:|
| 4-variable `dvth0` grid | 81 | 0.754847 | 0.756009 | 0.755440 | 0.000354 | 0.001162 |
| 2-variable MOM cap grid | 9 | 0.742910 | 0.766716 | 0.755016 | 0.009715 | 0.023806 |
| no-PEX 4-corner PVT | 4 | 0.720369 | 0.901262 | 0.818865 | 0.064461 | 0.180893 |
| PEX 4-corner PVT | 4 | 0.652208 | 0.789592 | 0.737273 | 0.051273 | 0.137384 |

### Interpretation

The cap grid spread is about:

```text
0.023806 / 0.001162 ~= 20.5x
```

larger than the four-variable threshold-voltage grid spread at the same sample point.

This supports a strong, thesis-useful finding:

> For this charge-domain C-2C inference testbench, `/OA_Charge` is much more sensitive to MOM capacitor variation, especially the main capacitance term `d_c1_vp`, than to the tested SRAM/precharge MOS threshold-voltage deltas.

This is physically plausible. `/OA_Charge` is a charge-sharing node whose value is primarily set by capacitor ratios. MOS threshold shifts mainly affect switch/access behavior unless they disturb timing, stored state, or charge transfer enough to become limiting.

## Detailed Cap Sensitivity

From `manual_mc_2_var_cap_1.csv`, the visible trend is:

- `umc_mc_d_c1_vp = -0.067` -> `/OA_Charge` around `0.743-0.749 V`
- `umc_mc_d_c1_vp = 0` -> `/OA_Charge` around `0.755-0.762 V`
- `umc_mc_d_c1_vp = +0.067` -> `/OA_Charge` around `0.767-0.774 V`

At fixed `d_c1_vp`, `d_cox_vp` changes the output much less than `d_c1_vp`.

Software/plot recommendation:

- Plot `/OA_Charge` vs `d_c1_vp` with different colors for `d_cox_vp`.
- If time is short, run or use a clean `d_c1_vp`-only sweep:

```text
umc_mc_d_c1_vp = -0.067, -0.0335, 0, 0.0335, 0.067
umc_mc_d_cox_vp = 0
```

This gives a simple sensitivity curve suitable for slides.

## Value of the 3-Corner PEX/No-PEX Runs

The 3-corner runs are useful, but they answer a different question from surrogate MC.

They are **deterministic PVT/post-layout characterization**, not statistical mismatch.

At 200.25 ns:

| Corner | no-PEX (V) | PEX (V) | PEX/no-PEX | PEX delta (V) |
|---|---:|---:|---:|---:|
| nominal | 0.827435 | 0.755421 | 0.912967 | -0.072014 |
| FF / high VDD / low temp | 0.901262 | 0.789592 | 0.876096 | -0.111670 |
| SS / low VDD / high temp | 0.720369 | 0.652208 | 0.905381 | -0.068161 |
| FNSP | 0.826396 | 0.751872 | 0.909821 | -0.074523 |

Useful claims:

- PEX reduces the sampled `/OA_Charge` relative to no-PEX by about `9-12%`, depending on corner.
- FF/high-VDD/low-temperature gives the largest sampled `/OA_Charge`; SS/low-VDD/high-temperature gives the smallest.
- PEX reduces the absolute PVT spread in these exports:
  - no-PEX spread: `180.9 mV`
  - PEX spread: `137.4 mV`
- These results are strong evidence for **post-layout/PVT sensitivity** and can support a plot or thesis section independent of MC.

Do not claim:

- "The 3-corner spread is Monte Carlo sigma."
- "PEX corner results replace statistical Phase 5."
- "HWA recovered PEX loss" unless a dedicated PEX-only accuracy evaluation is generated and clearly labeled.

Potential software use:

- Extend `src/hwa_cim/maestro_pex.py` so `corner_runs` can ingest the wide 4-corner VIVA CSVs directly.
- Generate a `pex_corner_proxy` artifact only if it is labeled as non-statistical corner proxy. This matches AgDR-0004.
- Add a PVT/PEX plot:
  - no-PEX vs PEX bars per corner
  - relative gain per corner
  - optional "PVT spread before/after PEX" summary

## Are We in Phase 5 Now?

No, not under the existing repo rules.

Current Phase 5 definition requires:

```csv
input_code,weight_population,ideal_output,mean_output,sigma,CSNR_dB
```

with many statistical samples per input code and weight-population class. The current surrogate exports are:

- one or a few stimulus/testbench conditions
- manual grids, not random samples with foundry-certified sigma
- `/OA_Charge` waveform families, not code-indexed noise tables

Best classification:

| Name | Appropriate? | Notes |
|---|---|---|
| Phase 5 | No | Reserved for real or at least complete code-indexed statistical profile |
| Phase 5S / surrogate Phase 5 | Yes | If the software explicitly labels provenance and limitations |
| Phase 4.5 | Yes | Best conservative label for thesis/status docs |
| UMC Monte Carlo | No | Implies native foundry MC decks |
| Surrogate Monte Carlo | Yes | Best concise thesis label |
| User-defined Gaussian parametric variation | Yes | Most technically precise |

Recommended wording:

> Native UMC Monte Carlo could not be executed reliably in the available environment, so a surrogate statistical variation flow was implemented. The flow uses TT models and user-defined Gaussian or grid perturbations of exposed PDK delta parameters. Sigma values are estimated from FF/SS corner deltas assuming the corner values approximate three-sigma process limits. These results are reported as surrogate Monte Carlo / parametric statistical variation, not foundry-certified UMC Monte Carlo.

## What Software Should Do Next

Before implementing these changes, write a new AgDR because this changes training/evaluation semantics and hardware profile provenance.

Suggested AgDR title:

```text
AgDR-0005-surrogate-monte-carlo-profile-mode.md
```

Suggested decision:

> Add a separate surrogate MC profile path for user-defined Gaussian/parametric delta-parameter sweeps, distinct from true `--noise-mode csv` Monte Carlo and distinct from deterministic PEX/corner proxy.

### 1. Add a Surrogate Sweep Parser

Create a parser that reads wide VIVA CSVs like:

- `manual_mc_4_var_1.csv`
- `manual_mc_2_var_cap_1.csv`

It should:

- Find all `/OA_Charge ... X/Y` column pairs.
- Parse sweep variables from headers, e.g. `umc_mc_d_c1_vp=0.067`.
- Sample each trace at `sample_time_ns` (default `200.25 ns`).
- Emit:
  - per-point CSV: `marker,signal,sample_time_ns,variable_group,<sweep variables>,sampled_v`
  - summary JSON/CSV: mean, std, min, max, spread, number of points, provenance warning

Suggested CLI:

```bash
hwa-surrogate-mc \
  --csv stuff_from_cadence/manual_mc_2_var_cap_1.csv \
  --signal /OA_Charge \
  --sample-time-ns 200.25 \
  --out-dir results/surrogate_mc/cap_sweep
```

### 2. Keep True MC CSV Semantics Protected

Do not silently feed surrogate data through the existing Phase 5 `--noise-mode csv` path as if it were foundry MC.

Options:

- Add `hardware_profile_mode = surrogate_mc`.
- Add explicit metadata:
  - `profile_kind = user_defined_surrogate_mc`
  - `profile_is_statistical = true`
  - `profile_is_foundry_certified = false`
  - `sigma_source = corner_delta_div_3`
  - `profile_warning = Surrogate MC; not UMC-certified Monte Carlo`
- Keep `monte_carlo_csv` reserved for true Phase 5 CSVs.

If the software uses surrogate sigma for HWA training, the run/plot title must include "surrogate".

### 3. Generate a Phase-5S Profile, Not a Final Phase-5 Profile

For current data, the software can generate a limited profile like:

```csv
marker,signal,sample_time_ns,variable_group,n_points,mean_output,sigma_output,min_output,max_output,profile_kind
cap_sweep,/OA_Charge,200.25,mom_cap_grid,9,0.755016,0.009715,0.742910,0.766716,user_defined_surrogate_mc
dvth0_sweep,/OA_Charge,200.25,dvth0_grid,81,0.755440,0.000354,0.754847,0.756009,user_defined_surrogate_mc
```

This is useful for thesis figures and software stress tests, but it is not enough for the existing `NoiseProfileCSV` schema because it lacks:

- `input_code`
- `weight_population`
- multiple code/pattern stimuli
- foundry-certified statistical sampling

To approach true Phase 5, rerun the surrogate flow over the verified stimulus set:

- For each `input_code` or each test vector marker
- For each relevant weight-population class
- Prefer random Gaussian draws if the UI can produce samples; otherwise use a documented grid
- Export each waveform family
- Use the parser to compute `mean_output` and `sigma` per marker/code

This would be defensible as **Phase 5S**, and it could drive an explicitly labeled HWA experiment.

### 4. Add Plots

Minimum presentation plots:

1. **Surrogate sensitivity bars**
   - `dvth0` spread vs MOM cap spread at 200.25 ns
   - Expected conclusion: MOM cap variation dominates.
2. **MOM cap sweep plot**
   - `/OA_Charge` vs `d_c1_vp`
   - Separate series or labels for `d_cox_vp`
3. **PEX/PVT corner plot**
   - no-PEX vs PEX per corner
   - relative gain per corner
4. **HWA context plot**
   - FP32 clean
   - INT4/noise no HWA
   - INT4/noise HWA synthetic or surrogate-labeled

Do not mix surrogate MC, PEX deterministic gain, and PVT corners into a single unlabeled "MC" plot.

### 5. Update GUI / Metrics Provenance

If the Streamlit GUI is used, add or update hardware-profile modes:

- Synthetic
- Maestro PEX deterministic
- PEX/PVT corner proxy
- Surrogate MC
- Foundry/statistical MC CSV

Every metrics JSON should include:

```json
{
  "hardware_profile_mode": "surrogate_mc",
  "profile_kind": "user_defined_surrogate_mc",
  "profile_is_statistical": true,
  "profile_is_foundry_certified": false,
  "sigma_source": "PDK FF/SS corner delta divided by 3",
  "profile_warning": "Surrogate MC; not UMC-certified Monte Carlo"
}
```

## Thesis Claims: Safe vs Unsafe

Safe claims:

- A surrogate MC flow was implemented because native UMC MC was blocked.
- The flow perturbs physically meaningful PDK delta parameters.
- `d_c1_vp` / main MOM capacitance variation dominates the sampled `/OA_Charge` sensitivity in the current testbench.
- PEX causes a deterministic gain reduction on `/OA_Charge` of roughly `9-12%` depending on corner.
- PVT corner spread is larger than the surrogate mismatch grids tested so far.
- HWA remains valuable as a methodology; the software is now ready to consume a true Phase 5 profile when available.

Unsafe claims:

- "We ran UMC Monte Carlo."
- "The surrogate sigma is the official foundry mismatch sigma."
- "The 3-corner PVT spread is MC sigma."
- "This fully completes Phase 5."
- "PEX calibration improved MNIST accuracy" without a dedicated, correctly labeled evaluation.

## Recommended Immediate Next Steps

1. Preserve the current Cadence exports under `stuff_from_cadence/`.
2. Write AgDR-0005 before changing training/evaluation code.
3. Implement `hwa-surrogate-mc` parser and summary outputs.
4. Generate two plots:
   - `dvth0` vs MOM cap sensitivity
   - no-PEX vs PEX across the 4 corners
5. Decide whether to run additional Cadence exports:
   - Recommended: `d_c1_vp`-only 5-point sweep for a cleaner slope plot.
   - Optional: surrogate runs for multiple input vectors/codes to build Phase 5S profile.
   - Avoid: huge Cartesian sweeps across all variables.
6. Only call the final software experiment "Phase 5" if it receives a code-indexed statistical profile; otherwise call it "Phase 5S" or "Phase 4.5 surrogate MC".

## One-Sentence Status for the Next Agent

We have not completed true Phase 5, but we now have enough surrogate statistical and PEX/PVT evidence to implement a clearly labeled **Phase 5S / surrogate MC** software path, generate thesis-grade sensitivity plots, and keep the real Phase 5 CSV contract intact for future foundry-certified Monte Carlo.
