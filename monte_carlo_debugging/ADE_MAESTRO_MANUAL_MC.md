# Manual MC bypass in ADE Assembler (UMC L65SP v132)

Use this flow when native PDK Monte Carlo fails (`SFE-4` / `SFE-7` on `mc_sp_*` + encrypted `*_statistical_p`), but **TT simulation already works** on your testbench.

| Item | Location |
|------|----------|
| Model wrapper | `monte_carlo_debugging/umc65_manual_mc.scs` |
| Sweep CSV template | `monte_carlo_debugging/manual_mc_sweep_template.csv` |
| Option C σ tables | `user_defined_statistical_variation.md`, `*_minimum.csv`, `*_extended.csv` |
| PDK reference copy | `monte_carlo_debugging/l65sp_v132_mc.lib.scs` (do not replace installed PDK) |

**Use ADE Assembler** (not legacy ADE Explorer) for parametric sweeps and Maestro plans. Explorer can run single simulations, but Assembler + Maestro is the flow that matches batch thesis work.

---

## What this does (and does not)

| Does | Does not |
|------|----------|
| Runs Spectre with **TT** MOSFET / MOM models | Replace UMC-certified `mc_sp_*` Monte Carlo |
| Sweeps PDK **delta parameters** (`dvth0_*`, `du0_*`, …) via ADE variables | Decrypt `statistical_p` or recover official σ automatically |
| Works in **Parametric Analysis** and **Maestro** plans | Give **per-instance** mismatch (unless you add that yourself) |

Document thesis results as **“parametric mismatch sweep at TT using PDK delta parameters”**, not “UMC Monte Carlo,” until UMC confirms equivalence or provides σ.

---

## Prerequisites

1. **Virtuoso IC6** with UMC65 Mixed-Mode Low-K PDK loaded (`l65sp_v132`).
2. **Working TT sim** on the SRAM testbench (no `mc_sp_lvt11` in model setup).
3. Spectre can resolve `l65sp_v132_mc.lib.scs` from the PDK tree (normal when the PDK is configured in Cadence).
4. Copy or link `umc65_manual_mc.scs` to a path Spectre can read (repo copy is fine if you add that directory to the include path — see Step 2).

---

## Step 0 — One-time: confirm TT-only sim passes

1. Open the schematic testbench cell used for SRAM verification.
2. **Launch → ADE Assembler** (or **Launch → ADE XL** then open/create Assembler view).
3. **Setup → Model Libraries** (or **Setup → Simulator/Analyzer Setup → Model Libraries**).
4. Ensure model entries look like **TT**, not MC, for example:
   - `l65sp_v132_mc.lib.scs` → section **`tt_sp_lvt11`** (or nominal `l65sp_v132.lib.scs` → `tt_sp_lvt11` if you use the non-MC card)
   - Same for **`tt_sp_hvt11`** and **`tt_65_momcaps`** if the bench uses SPLVT + SPHVT + MOM
5. **Remove** any lines with:
   - `mc_sp_lvt11`
   - `mc_sp_hvt11`
   - `mc_65_momcaps`
6. **Simulation → Run** (single point). Confirm **no SFE-4** and expected TT results.

If this fails, fix the base TT setup before continuing.

---

## Step 1 — Install the wrapper on the Cadence side

On the machine that runs Spectre (your lab server):

```bash
# Example: copy into a stable path next to your testbench project
cp "/path/to/hwa-cim/monte_carlo_debugging/umc65_manual_mc.scs" \
   "$HOME/umc65_manual_mc/umc65_manual_mc.scs"
```

Note the **absolute path** — you will paste it into Model Libraries.

Optional: add the directory to Spectre search path in the testbench:

- **Setup → Environment → Include / Definition Files**, or  
- `simulatorOptions` / `include` in your existing ADE env configuration.

---

## Step 2 — Point Model Libraries at the wrapper only

In **ADE Assembler** for this testbench:

1. **Setup → Model Libraries**.
2. **Disable or delete** all `mc_sp_*` section includes.
3. **Disable or delete** duplicate `tt_sp_*` lines if the wrapper already includes them.
4. **Add one line** (adjust path):

   ```
   /home/<you>/umc65_manual_mc/umc65_manual_mc.scs
   ```

   Section column: leave **empty** (the wrapper file is self-contained).

5. **Apply** / **OK**.

The wrapper sets `manualMcOptions options redefinedparams=ignore` so ADE `umc_mc_*` remaps can override PDK defaults already declared inside `tt_sp_*` (otherwise Spectre stops with **SFE-59**).

The wrapper internally does:

```spectre
include "l65sp_v132_mc.lib.scs" section=tt_sp_lvt11
include "l65sp_v132_mc.lib.scs" section=tt_sp_hvt11
include "l65sp_v132_mc.lib.scs" section=tt_65_momcaps
```

So you must **not** also list those three sections separately in Model Libraries (avoid double-include).

### Post-layout vs pre-layout

`tt_sp_lvt11` / `tt_sp_hvt11` pull in `cctg_pre_simu_*` (pre-layout contact-to-gate cap, `cctgflag=1`).

- **Pre-layout schematic sim:** leave as-is.  
- **Post-layout / PEX sim:** edit `umc65_manual_mc.scs` locally to use corner sections **without** `cctg_pre_simu`, or ask UMC flow for post-layout TT sections. Do not mix pre-layout `cctg` with extracted layout without intent.

---

## Step 3 — Create ADE design variables (sweep knobs)

In Assembler **Setup → Design Variables** (sometimes **Variables** in the toolbar), add each variable with **default 0**:

### Minimum set (smoke test — 4 variables)

| Variable name | Maps to PDK parameter | Default |
|---------------|----------------------|---------|
| `umc_mc_dvth0_n_splvt` | `dvth0_n_11_splvt` | `0` |
| `umc_mc_dvth0_p_splvt` | `dvth0_p_11_splvt` | `0` |
| `umc_mc_dvth0_n_sphvt` | `dvth0_n_11_sphvt` | `0` |
| `umc_mc_dvth0_p_sphvt` | `dvth0_p_11_sphvt` | `0` |

### Recommended set (matches `umc65_manual_mc.scs`)

Also add (default `0`):

- `umc_mc_du0_n_splvt`, `umc_mc_du0_p_splvt`
- `umc_mc_dtoxe_n_splvt`, `umc_mc_dtoxe_p_splvt`
- `umc_mc_dtoxp_n_splvt`, `umc_mc_dtoxp_p_splvt`
- `umc_mc_dxl_n_splvt`, `umc_mc_dxl_p_splvt`
- `umc_mc_dxw_n_splvt`, `umc_mc_dxw_p_splvt`
- `umc_mc_du0_n_sphvt`, `umc_mc_du0_p_sphvt`
- `umc_mc_dtoxe_n_sphvt`, `umc_mc_dtoxe_p_sphvt`
- `umc_mc_d_c1_vp`, `umc_mc_d_cox_vp`

Names must match the wrapper **exactly** (case-sensitive).

---

## Step 4 — Single-run sanity check (TT, all zeros)

1. Leave all `umc_mc_*` variables at `0`.
2. **Simulation → Run** (or green Run button).
3. Confirm:
   - Netlist/elaboration succeeds
   - No `SFE-4` / `SFE-7`
   - Results match your earlier TT baseline (small numerical noise is OK)

If Spectre reports **undefined variable** `umc_mc_*`, the wrapper is loaded but variables were not created in Step 3.

---

## Step 5 — Enable parametric sweep (Assembler)

Do **not** enable the analysis type **Monte Carlo** in Spectre.

1. **Analyses** → choose your analysis (e.g. **tran** for read/write, or **dc** for SNM).
2. Open **Parametric Analysis** (toolbar icon **Px**, or **Tools → Parametric Analysis**, or **Simulation → Parametric Analysis** depending on IC6 subversion).
3. In the parametric window:
   - **Add** variables from the design variable list (e.g. `umc_mc_dvth0_n_splvt`).
   - Choose sweep mode:

### Option A — Manual grid (quick test)

| Variable | Sweep |
|----------|--------|
| `umc_mc_dvth0_n_splvt` | `-0.003`, `0`, `0.003` |
| `umc_mc_dvth0_p_splvt` | `-0.002`, `0`, `0.002` |

Use placeholder sigmas until UMC provides mismatch σ. **Do not** use FF/SS corner values (±25 mV) as 1σ — those are global corners.

### Option B — Import CSV (many points)

1. Use `manual_mc_sweep.csv` for the prepared 54-point sweep. It includes all 20 `umc_mc_*` variables used by `umc65_manual_mc.scs`.
   - Keep `manual_mc_sweep_template.csv` only as the tiny example / fallback.
   - These are placeholder parametric draws for debugging the flow, not certified UMC Monte Carlo sigma values.
2. In Parametric Analysis: **File → Import** or **Load from file** (wording varies).
3. Map columns to design variables.
4. Set **Run mode** to **Points** or **All combinations** as needed:
   - **Points** = each row is one simulation (recommended for MC-style draws).
   - **Cartesian product** = full grid (grows fast).

### Option C — User-defined statistical variation

In **User Defined Statistical Variation Setup** (enable design variable statistical variation):

- **Mean:** no separate column — keep each `umc_mc_*` global variable at **`0`**.
- **Std / Range (N):** copy from [`user_defined_statistical_variation.md`](user_defined_statistical_variation.md) or the CSV companions:
  - Minimum (4 variables): [`user_defined_statistical_variation_minimum.csv`](user_defined_statistical_variation_minimum.csv)
  - Extended (14 variables): [`user_defined_statistical_variation_extended.csv`](user_defined_statistical_variation_extended.csv)

σ values are **educated guesses** from PDK FF/SS corner deltas ÷ 3, not certified UMC mismatch statistics. Label thesis results accordingly.

4. **Apply** parametric setup.
5. **Simulation → Run Parametric** (or Run with parametric enabled).

Results appear as `parametric=N` folders under the PSF database, or in **Results → Parametric Plot**.

---

## Step 6 — Maestro batch (optional, for many points)

Use Maestro when you need parallel runs or regression-style plans.

1. In Assembler: **Tools → Maestro** (or **Window → Maestro**), or start **Maestro** from the CIW and **Import** the ADE testbench.
2. **Create Plan** → attach your SRAM testbench configuration.
3. Verify the plan inherits:
   - Model library: `umc65_manual_mc.scs` only  
   - Design variables from Step 3  
   - Parametric definition from Step 5  
4. **Setup → Run Options** → set parallelism (hosts/cores) per lab policy.
5. **Run Plan**.
6. Monitor **Job Monitor** for failed points; open any failing point’s `spectre.out` for errors.

Maestro does not change the physics — it only schedules multiple parametric points.

---

## Step 7 — Save outputs for thesis / Python pipeline

1. **Outputs** → ensure saved signals (e.g. `/OA_Charge`, bitlines, SNM probes) are checked **before** the parametric run.
2. After the run:
   - **Results → Save** or export **CSV** via **VIVA** (right-click waveform → **Export**), or  
   - Use your existing Maestro/VIVA export script for `nopex`/`pex`-style CSVs.
3. Label files clearly, e.g. `manual_mc_point_042.csv`, and keep a **manifest** mapping point index → (`umc_mc_dvth0_n_splvt`, …).

For **hwa-cim** Phase 5 statistical MC, use only exports that include `input_code`, `ideal_output`, `mean_output`, `sigma` per code — this parametric path produces **per-run waveforms**, not automatic code-indexed MC tables unless your testbench generates them.

---

## Step 8 — Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `SFE-4` / `SFE-7` returns | Still including `mc_sp_*` or `statistical_p` | Remove from Model Libraries; wrapper only |
| `undefined parameter` / variable | Missing design variable in ADE | Add all `umc_mc_*` names in Step 3 |
| Duplicate model definitions | Wrapper **and** separate `tt_sp_*` in Model Libraries | Keep wrapper only |
| **`SFE-59` parameter already defined** (`dvth0_n_11_splvt`, …) | `tt_sp_*` declares deltas; wrapper remaps them | Copy updated `umc65_manual_mc.scs` (has `parameters redefinedparams=ignore` before remaps). Re-run Step 4. |
| TT baseline shifted with all zeros | Double-include or wrong section | Single path to `umc65_manual_mc.scs` |
| Parametric does not run | MC analysis enabled instead of parametric | Disable Spectre **montecarlo** analysis |
| Results identical for all points | Variables not wired to parametric set | Re-import CSV; check **Points** mode |
| `l65sp_v132_mc.lib.scs` not found | PDK path not on include path | Fix PDK install / `cds.lib` / `include` env |

---

## Checklist (printable)

- [ ] TT-only sim works without `mc_sp_*`
- [ ] `umc65_manual_mc.scs` on disk with correct absolute path in Model Libraries
- [ ] No duplicate `tt_sp_*` includes in Model Libraries
- [ ] All `umc_mc_*` design variables created (default 0)
- [ ] Single run at zero passes
- [ ] Parametric Analysis enabled (not Spectre Monte Carlo)
- [ ] Sweep CSV or grid uses **UMC σ**, not FF/SS corners
- [ ] Outputs saved / exported with point index manifest
- [ ] Thesis text uses correct nomenclature (parametric sweep vs UMC MC)

---

## Software handoff (Phase 4.5)

After exporting wide VIVA CSVs to `stuff_from_cadence/`:

```bash
hwa-surrogate-mc --all-defaults
hwa-plot-surrogate-mc
```

This produces **Phase 4.5 — Surrogate Monte Carlo (user-defined Gaussian parametric variation)** summaries and plots under `results/surrogate_mc/` and `results/plots/03_surrogate_mc/`. These are **not** the Phase 5 `input_code` noise profile; do not label as UMC-certified Monte Carlo. See [AgDR-0005](../docs/agdr/AgDR-0005-surrogate-monte-carlo-profile-mode.md).

---

## Related repo docs

- [AgDR-0004](../docs/agdr/AgDR-0004-maestro-pex-calibration-path.md) — Maestro PEX path (deterministic layout), separate from statistical MC.
- [AgDR-0005](../docs/agdr/AgDR-0005-surrogate-monte-carlo-profile-mode.md) — Phase 4.5 surrogate MC profile mode.
- Email to UMC support — certified Spectre version and official `mc_sp_*` procedure remain the long-term fix.

---

## File reference: wrapper mapping

```text
umc_mc_dvth0_n_splvt  →  dvth0_n_11_splvt   (SPLVT NMOS)
umc_mc_dvth0_p_splvt  →  dvth0_p_11_splvt   (SPLVT PMOS)
umc_mc_dvth0_n_sphvt  →  dvth0_n_11_sphvt   (SPHVT NMOS, precharge)
umc_mc_dvth0_p_sphvt  →  dvth0_p_11_sphvt   (SPHVT PMOS)
umc_mc_d_c1_vp        →  d_c1_vp            (MOM)
umc_mc_d_cox_vp       →  d_cox_vp           (MOM)
```

Full TT corner parameter lists are in `l65sp_v132_mc.lib.scs` sections `tt_sp_lvt11_corner`, `tt_sp_hvt11_corner`, and `tt_65_momcaps_corner`.
