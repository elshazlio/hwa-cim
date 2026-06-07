# User-defined statistical variation (Option C)

Use in ADE Assembler: **User Defined Statistical Variation Setup** (enable design variable statistical variation).

**Mean:** There is no Mean column. Set each `umc_mc_`* **Global Variable** default to `0`; Gaussian sampling is around that nominal.

**σ source:** Educated guess from PDK `l65sp_v132_mc.lib.scs` FF/SS corner deltas on `dvth0_`*, `du0_*`, `dtoxe_*`, `d_c1_vp`, `d_cox_vp`, assuming corner ≈ **3σ** (not certified UMC mismatch σ).

**Thesis label:** *User-defined Gaussian parametric variation with σ estimated from PDK corner deltas; not certified UMC Monte Carlo.*

---

## Minimum run (start here)

Enable only these four. Suggested sample count: **20–50** (set in Maestro/run options, not in this table).


| Name                   | Distribution Type | Std     | Range (N) | Global var default (mean) |
| ---------------------- | ----------------- | ------- | --------- | ------------------------- |
| `umc_mc_dvth0_n_splvt` | Gaussian          | `0.009` | `3`       | `0`                       |
| `umc_mc_dvth0_p_splvt` | Gaussian          | `0.006` | `3`       | `0`                       |
| `umc_mc_dvth0_n_sphvt` | Gaussian          | `0.006` | `3`       | `0`                       |
| `umc_mc_dvth0_p_sphvt` | Gaussian          | `0.006` | `3`       | `0`                       |


Std units: **volts** (e.g. `0.009` = 9 mV).

### Corner rationale (SPLVT / SPHVT dvth0)


| Parameter          | PDK corner deltas (approx.)  | σ = |Δ| / 3 |
| ------------------ | ---------------------------- | ----------- |
| `dvth0_n_11_splvt` | +25.3 mV (ss), −29.0 mV (ff) | **0.009** V |
| `dvth0_p_11_splvt` | −16.9 mV (ss), +19.1 mV (ff) | **0.006** V |
| `dvth0_n_11_sphvt` | ±17–18 mV                    | **0.006** V |
| `dvth0_p_11_sphvt` | ±16–17 mV                    | **0.006** V |


---

## Extended run (after minimum passes)

Add mobility, tox, and MOM cap deltas. Same **Range (N) = 3**, global defaults **0**.


| Name                   | Distribution Type | Std       | Range (N) |
| ---------------------- | ----------------- | --------- | --------- |
| `umc_mc_du0_n_splvt`   | Gaussian          | `0.0004`  | `3`       |
| `umc_mc_du0_p_splvt`   | Gaussian          | `0.00015` | `3`       |
| `umc_mc_du0_n_sphvt`   | Gaussian          | `0.0006`  | `3`       |
| `umc_mc_du0_p_sphvt`   | Gaussian          | `0.0004`  | `3`       |
| `umc_mc_dtoxe_n_splvt` | Gaussian          | `3.3e-11` | `3`       |
| `umc_mc_dtoxe_p_splvt` | Gaussian          | `3.3e-11` | `3`       |
| `umc_mc_dtoxe_n_sphvt` | Gaussian          | `3.3e-11` | `3`       |
| `umc_mc_dtoxe_p_sphvt` | Gaussian          | `3.3e-11` | `3`       |
| `umc_mc_d_c1_vp`       | Gaussian          | `0.067`   | `3`       |
| `umc_mc_d_cox_vp`      | Gaussian          | `0.067`   | `3`       |


MOM: PDK `ff_65_momcaps_corner` / `ss_65_momcaps_corner` use `d_c1_vp` / `d_cox_vp` = **±0.20** → σ ≈ **0.067** (~6.7% relative cap variation at 1σ).

Optional geometry deltas (wrapper supports them; not in minimum table):


| Name                   | Std (corner/3 guess) | Range (N) |
| ---------------------- | -------------------- | --------- |
| `umc_mc_dtoxp_n_splvt` | `3.3e-11`            | `3`       |
| `umc_mc_dtoxp_p_splvt` | `3.3e-11`            | `3`       |
| `umc_mc_dxl_n_splvt`   | `5.3e-10`            | `3`       |
| `umc_mc_dxl_p_splvt`   | `1.2e-9`             | `3`       |
| `umc_mc_dxw_n_splvt`   | `1.0e-9`             | `3`       |
| `umc_mc_dxw_p_splvt`   | `1.4e-9`             | `3`       |


---

## Do not use as σ

- FF/SS **full corner** `dvth0` values directly as 1σ (e.g. ±25 mV) — those are global PVT corners, not per-lot mismatch.
- Values from `manual_mc_sweep.csv` — that file is for **parametric point import** (Option B), not this dialog.

---

## Related files

- Wrapper: `umc65_manual_mc.scs`
- Flow: `ADE_MAESTRO_MANUAL_MC.md` (Option C)
- PDK reference copy: `l65sp_v132_mc.lib.scs` (corner sections `*_corner`)

