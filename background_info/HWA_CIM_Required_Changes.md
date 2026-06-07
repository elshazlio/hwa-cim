# HWA-CIM Repository — Required Changes Per Verification Roadmap

**Companion to:** `CIM_SRAM_Verification_Software_Roadmap.md`  
**Target repository:** [`github.com/elshazlio/hwa-cim`](https://github.com/elshazlio/hwa-cim) (Python package: `hwa-cim`)  
**Repository scope (current):** HWA training pipeline (Phases 1–5 of AFM methodology)  
**Document version:** 1.0  
**Last updated:** 2026-05-11  
**Status:** Pending implementation; layout phase in parallel

---

## 1. Executive Summary

After full review of the `hwa-cim` source code (`src/hwa_cim/*.py`), I identified **12 required changes** across 5 source files plus 3 new files. The changes fall into 4 priority tiers:

| Priority | Count | Description |
|---|---|---|
| 🔴 **Critical (blocks thesis claim)** | 3 changes | Hardware-aware MAC currently uses purely ideal math; missing gain-pattern dependence + offset |
| 🟠 **Important (affects Phase 5 + PTQ baseline)** | 5 changes | CSV schema and noise injection need refinement; INT4 PTQ should reflect hardware |
| 🟡 **Moderate (improves accuracy)** | 4 changes | Documentation, plot calibration, parity test handling |
| 🟢 **Optional (best practice)** | 3 changes | Regression fixtures, shared config, MIM/MOM cleanup |

**Headline finding:** The current `c2c.py` `c2c_mac()` function performs an **ideal INT4 matmul** identical to `nn.Linear`. This is mathematically correct but **misses the dominant non-ideality** of the verified hardware: a gain-pattern-dependent transfer function (G_eff = 0.62 for sparse weights, 0.44 for dense weights) and a residual ~50 mV offset. The HWA training loop currently learns to be robust against **synthetic Gaussian weight noise only**, not against the actual systematic gain shift that the hardware exhibits. This is the gap most critical to close before Phase 5 hardware integration.

---

## 2. File-by-File Required Changes

### 2.1 `src/hwa_cim/c2c.py` — Major refactor required 🔴

**Three changes needed:**

#### Change C-1: Add hardware calibration constants 🔴 Critical

After line 21 (`from hwa_cim.quantization import dequantize_int4_matmul`), add:

```python
# Hardware calibration constants from schematic-level verification
# (CIM_SRAM_Verification_Software_Roadmap.md Section 3.3)
# These should be moved to calibration.yaml in a follow-up commit.

# Gain factor extracted from 4-row charge-shared MAC across 8 verified test vectors
G_EFF_SPARSE = 0.62  # Weight population <= 4 bits set total across all rows
G_EFF_DENSE = 0.44   # Weight population >= 12 bits set total
OFFSET_DENSE = 50e-3 # mV residual after auto-reset (NMOS gated by OR(EN, WE))

# Standalone C-2C ladder (without integrated S/H load)
G_EFF_STANDALONE = 0.831  # Reference value, not used in integrated forward pass

# Sparse/dense regime boundaries (in bit population units, total across 4 rows of weights)
POPULATION_SPARSE_MAX = 4
POPULATION_DENSE_MIN = 12
```

**Rationale:** These numbers are the empirical result of schematic verification (Section 3.3 of the roadmap). They define the actual hardware transfer function. Currently the code has no awareness of them.

#### Change C-2: Add gain-pattern computation function 🔴 Critical

After the constants block above, add:

```python
def compute_g_eff(weights_q: torch.Tensor) -> torch.Tensor:
    """
    Compute the effective gain factor G_eff based on weight bit population
    per output channel (per row of the weight matrix).

    Returns a tensor of shape [out_features] with G_eff for each output.

    Mapping: each output row in `weights_q` represents one tile's 4 weights.
    Population = total `1` bits across the row's 4-bit values.

    Sparse regime (pop <= 4):  G_eff = 0.62
    Dense regime (pop >= 12):  G_eff = 0.44
    Mid regime: linear interpolation
    """
    # Count bits per row (per output channel)
    # weights_q is int8 storing INT4 values in [-8, 7]; convert to magnitude bit pattern
    abs_w = weights_q.abs().to(torch.int32)  # [out, in]
    # popcount per element (4-bit values, so at most 4 bits per element)
    pop_per_elem = (
        (abs_w & 1) +
        ((abs_w >> 1) & 1) +
        ((abs_w >> 2) & 1) +
        ((abs_w >> 3) & 1)
    )
    # Sum across input dimension to get total population per output row
    # In real hardware, MAC operates on tiles of 4 inputs at a time. For software
    # modeling, we approximate using per-row total population scaled to 4-tile size.
    n_tiles = max(1, weights_q.shape[1] // 4)
    pop_per_tile = pop_per_elem.float().sum(dim=1) / n_tiles  # avg pop per tile

    # Piecewise linear interpolation
    g_eff = torch.full_like(pop_per_tile, G_EFF_SPARSE)
    sparse_mask = pop_per_tile <= POPULATION_SPARSE_MAX
    dense_mask = pop_per_tile >= POPULATION_DENSE_MIN
    mid_mask = ~(sparse_mask | dense_mask)

    g_eff[sparse_mask] = G_EFF_SPARSE
    g_eff[dense_mask] = G_EFF_DENSE
    # Linear interpolation in mid regime
    f = (pop_per_tile[mid_mask] - POPULATION_SPARSE_MAX) / (
        POPULATION_DENSE_MIN - POPULATION_SPARSE_MAX
    )
    g_eff[mid_mask] = G_EFF_SPARSE + f * (G_EFF_DENSE - G_EFF_SPARSE)

    return g_eff


def compute_offset(weights_q: torch.Tensor) -> torch.Tensor:
    """
    Compute the residual offset (post auto-reset) for each output row.
    Scales with the same dense regime as G_eff.
    """
    abs_w = weights_q.abs().to(torch.int32)
    pop_per_elem = (
        (abs_w & 1) +
        ((abs_w >> 1) & 1) +
        ((abs_w >> 2) & 1) +
        ((abs_w >> 3) & 1)
    )
    n_tiles = max(1, weights_q.shape[1] // 4)
    pop_per_tile = pop_per_elem.float().sum(dim=1) / n_tiles

    offset = torch.zeros_like(pop_per_tile)
    sparse_mask = pop_per_tile <= POPULATION_SPARSE_MAX
    dense_mask = pop_per_tile >= POPULATION_DENSE_MIN
    mid_mask = ~(sparse_mask | dense_mask)

    offset[sparse_mask] = 0.0
    offset[dense_mask] = OFFSET_DENSE
    f = (pop_per_tile[mid_mask] - POPULATION_SPARSE_MAX) / (
        POPULATION_DENSE_MIN - POPULATION_SPARSE_MAX
    )
    offset[mid_mask] = f * OFFSET_DENSE

    return offset
```

**Caveat for software-side documentation:** The hardware computes G_eff based on the 4 weights physically located in a single tile. The full layer's weight matrix is tiled in hardware, but the software model here approximates by averaging across the whole row. This is a first-order approximation that captures the sparse-vs-dense behavior; for higher fidelity, partition the weight matrix into 4-input tiles explicitly and apply G_eff per tile (deferred to Phase 5 refinement).

**Important representation note (signed INT4 vs hardware unsigned 4-bit):** The schematic verification used unsigned 4-bit weights stored in SRAM (values 0–15). The HWA code uses signed INT4 (`symmetric_quantize_int4` produces values in [−8, 7] stored as int8). For the population-counting calibration to map correctly, we count bits of the **magnitude** (absolute value): for example, signed weight −7 maps to magnitude 7 (binary `0111`, popcount 3), and weight −8 maps to magnitude 8 (`1000`, popcount 1).

This is a simplification — the actual hardware behavior depends on the exact bit pattern stored in the SRAM cells, which may include sign bit encoding logic not yet specified. If the integrated SoC encodes signed weights as sign-magnitude (sign bit + 3-bit magnitude), the popcount semantics change accordingly. **For Phase 3 HWA training, the magnitude-based approximation is acceptable** because the training loop will learn to be robust against the resulting gain pattern. For Phase 5, when real Monte Carlo data is available, the CSV-based noise profile bypasses this approximation entirely.

#### Change C-3: Modify `c2c_mac` to apply calibration 🔴 Critical

Replace the existing `c2c_mac` function (lines 26–42) with:

```python
def c2c_mac(
    weights_q: torch.Tensor,
    activations_q: torch.Tensor,
    scale_w: torch.Tensor,
    scale_x: torch.Tensor,
    bias: torch.Tensor | None = None,
    shift_x: torch.Tensor | None = None,
    hardware_aware: bool = False,
) -> torch.Tensor:
    """
    C-2C MAC with optional hardware-aware calibration.

    weights_q: [out, in] signed INT4 in int8 storage
    activations_q: [batch, in] INT4/uint4 in int8 storage
    hardware_aware: if True, apply gain-pattern dependence + offset
                    measured at schematic level (default False for backwards compat
                    and `parity_linear_vs_c2c` test).

    Returns: [batch, out] dequantized MAC output.
    """
    y = dequantize_int4_matmul(activations_q, weights_q, scale_x, scale_w, bias, shift_x)

    if hardware_aware:
        g_eff = compute_g_eff(weights_q)       # [out]
        offset = compute_offset(weights_q)      # [out]
        # Broadcast across batch dimension
        y = y * g_eff.unsqueeze(0) + offset.unsqueeze(0)

    return y
```

**Why a flag, not always-on:** The `parity_linear_vs_c2c` test in `evaluate.py` checks that `c2c_mac` matches `nn.Linear` to floating-point precision. That test relies on the ideal math. Keeping `hardware_aware=False` as default preserves that test; the noisy/HWA training path will pass `hardware_aware=True` explicitly.

#### Change C-4: Update `C2CLadderWithParasitics` operating point 🟡 Moderate

The current parasitic_ratio sweep (lines 82–112) has no annotation for the actual operating point. Schematic verification shows the integrated macro operates around G_eff ≈ 0.62 for sparse weights, which corresponds to a parasitic_ratio of approximately **0.17** (computed from G_standalone=0.831 → G_integrated=0.62 → 25% additional loss from S/H + interconnect parasitics, broken across 4 ladder bits).

Update the `transfer` method docstring and add an operating point constant:

```python
# Approximate operating point of the integrated 4-bit ladder per schematic verification.
# 0.831 standalone → 0.62 integrated sparse → equivalent parasitic_ratio ~0.17.
# This is a heuristic; replace with calibrated value when G-05 SPICE model
# for MOMCAPS_SY_MMKF is available.
INTEGRATED_OPERATING_POINT = 0.17
```

Then in `plot_parasitic_sweep` (plots.py, see Change P-1), use this as the default PDK marker instead of 0.10.

---

### 2.2 `src/hwa_cim/layers.py` — Wire hardware-aware flag through 🔴 Critical

#### Change L-1: Pass `hardware_aware=True` in NoisyQuantLinear forward 🔴 Critical

The current `NoisyQuantLinear.forward()` uses `F.linear(x, w_noisy, b)` (line 67) — this is ideal matmul. It should use `c2c_mac` with `hardware_aware=True`.

But there's a complication: `NoisyQuantLinear` works on float weights (with STE), not the INT4 quantized weights that `c2c_mac` expects. The HWA training currently happens entirely in float-domain with STE backprop.

**Recommended solution:** Add a post-MAC gain shift to `NoisyQuantLinear.forward()` that emulates the hardware calibration without requiring integer quantization in the forward pass:

```python
# After: y = F.linear(x, w_noisy, b)
# Before: if self.use_adc: y = adc_quantize_ste(y, self.adc_bits)

if self.hardware_aware:
    # Approximate per-output G_eff based on the float weight magnitude pattern
    # (analog of bit population for fake-quant INT4)
    w_normalized = w_noisy / (w_noisy.detach().abs().max() + 1e-8)
    avg_magnitude = w_normalized.abs().mean(dim=1)  # [out]
    # Map avg_magnitude to G_eff via the same piecewise rule
    # avg_magnitude in [0, 1]; high magnitude ≈ "dense", low ≈ "sparse"
    from hwa_cim.c2c import G_EFF_SPARSE, G_EFF_DENSE, OFFSET_DENSE
    # Threshold at 0.4 (mid magnitude) ↔ population threshold of ~8 bits
    g_eff = torch.where(
        avg_magnitude < 0.25, torch.tensor(G_EFF_SPARSE, device=y.device),
        torch.where(
            avg_magnitude > 0.6, torch.tensor(G_EFF_DENSE, device=y.device),
            G_EFF_SPARSE + (avg_magnitude - 0.25) / 0.35 * (G_EFF_DENSE - G_EFF_SPARSE)
        )
    )
    offset = torch.where(
        avg_magnitude < 0.25, torch.tensor(0.0, device=y.device),
        torch.where(
            avg_magnitude > 0.6, torch.tensor(OFFSET_DENSE, device=y.device),
            (avg_magnitude - 0.25) / 0.35 * OFFSET_DENSE
        )
    )
    y = y * g_eff.unsqueeze(0) + offset.unsqueeze(0)
```

And add `hardware_aware` to the `__init__` signature:

```python
def __init__(
    self,
    in_features: int,
    out_features: int,
    bias: bool = True,
    *,
    gamma: float = 0.02,
    alpha_clip: float = 3.0,
    use_adc: bool = True,
    adc_bits: int = 4,
    noise_mode: str = "synthetic",
    sigma_global: float | None = None,
    hardware_aware: bool = True,  # NEW — default True per thesis intent
) -> None:
    ...
    self.hardware_aware = hardware_aware
```

#### Change L-2: Propagate `hardware_aware` through `NoisyMicroMLP` 🔴 Critical

In `models.py`, `NoisyMicroMLP.__init__` should accept and pass `hardware_aware` to each `NoisyQuantLinear`. Add to its signature:

```python
hardware_aware: bool = True,
```

And pass it through to each of `self.fc1`, `self.fc2`, `self.fc3` instantiation.

---

### 2.3 `src/hwa_cim/noise.py` — Refine Phase 5 CSV ingestion 🟠 Important

#### Change N-1: Per-code noise injection instead of global sigma_mean 🟠 Important

The current `noise_scale_for_forward()` returns `profile.sigma_mean` — a single scalar for the entire training run. This loses critical information: hardware noise is **code-dependent** (smaller signals have proportionally larger noise contribution).

Replace with a per-tensor lookup or interpolation:

```python
def noisy_forward_from_profile(
    ideal_output: torch.Tensor,
    profile: NoiseProfileCSV,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Apply per-code noise from a Phase 5 Monte Carlo profile.

    ideal_output: [batch, out_features] tensor from ideal MAC
    profile: loaded NoiseProfileCSV with per-code sigma values

    For each output value, find the closest code in profile.input_code,
    look up corresponding sigma, and add Gaussian noise scaled by it.
    """
    # Quantize output to integer codes (assumes 4-bit ADC range)
    code_count = len(profile.input_code)
    out_range = (max(profile.ideal_output) - min(profile.ideal_output))
    if out_range < 1e-8:
        return ideal_output

    codes = torch.tensor(profile.input_code, device=ideal_output.device, dtype=torch.int32)
    sigmas = torch.tensor(profile.sigma, device=ideal_output.device, dtype=ideal_output.dtype)

    # Map ideal_output to nearest code
    normalized = (ideal_output - min(profile.ideal_output)) / out_range
    nearest_idx = (normalized * (code_count - 1)).round().clamp(0, code_count - 1).long()

    # Look up sigma per output element
    sigma_per_elem = sigmas[nearest_idx]
    noise = torch.randn(
        ideal_output.shape,
        device=ideal_output.device,
        dtype=ideal_output.dtype,
        generator=generator,
    )
    return ideal_output + sigma_per_elem * noise
```

This function should be called from `NoisyQuantLinear.forward()` when `noise_mode == 'csv'` and a profile is available.

#### Change N-2: Extend CSV schema to include weight population context 🟠 Important

The current schema `[input_code, ideal_output, mean_output, sigma, CSNR_dB]` captures one row's noise but not the gain-pattern dependence. Suggested extended schema:

```csv
input_code,weight_population,ideal_output,mean_output,sigma,CSNR_dB,G_eff_measured,offset_measured
0,0,0.000,0.050,0.012,15.2,0.620,0.000
1,4,0.069,0.122,0.015,14.8,0.612,0.045
...
```

Where `weight_population` is the bit population of the test weight vector used in that Monte Carlo run. This lets the software model the full transfer function — both the gain shift AND the residual noise — rather than treating them as one lumped sigma.

Update `NoiseProfileCSV.load()` to handle these new columns (gracefully fall back if absent — for backwards compatibility).

#### Change N-3: Document the simulation methodology that produces this CSV 🟡 Moderate

Add a docstring section to `NoiseProfileCSV` explaining:
- What Cadence test produces this CSV (Spectre Monte Carlo on PEX netlist)
- How many MC iterations are recommended (≥ 100, ≥ 1000 for production)
- Whether ADC code or raw V_OA voltage is being captured
- The relationship to the verification roadmap's `predict_OA()` function

This is the integration boundary — the spec should be self-documenting.

---

### 2.4 `src/hwa_cim/evaluate.py` — Update `parity_linear_vs_c2c` and `forward_int4_mlp` 🟠 Important

**Code-discovery note:** After tracing usage, `c2c_mac` is only called in two places:
1. `forward_int4_mlp` (lines 48, 52) — used by `accuracy_int4` for INT4 PTQ baseline
2. `parity_linear_vs_c2c` (line 87) — the parity test

The HWA training path (`NoisyQuantLinear.forward`) uses `F.linear` directly and **never touches** `c2c_mac`. This means:
- The parity test should keep `hardware_aware=False` to verify the ideal math.
- The INT4 PTQ accuracy in `forward_int4_mlp` SHOULD use `hardware_aware=True` so the baseline reflects what the hardware actually does at INT4 precision.
- The HWA training gain shift must be implemented inside `NoisyQuantLinear.forward` (Change L-1 above), NOT through `c2c_mac`.

#### Change E-1: Use `hardware_aware=False` flag explicitly in parity test 🟡 Moderate

After Change C-3, the parity test will fail unless it explicitly requests ideal mode. Update line 96 of `evaluate.py`:

```python
def parity_linear_vs_c2c(device: torch.device = torch.device("cpu")) -> float:
    """Max error between dequant linear and c2c_mac (ideal mode), should be ~float noise."""
    ...
    y_mac = c2c_mac(w_q, x_q, sw, sx, lin.bias, shift_x=shift, hardware_aware=False)
    # The hardware_aware=False is explicit; this test verifies the math, not hardware behavior.
    ...
```

Also add a NEW parity test (`parity_c2c_against_verified_vectors`) that tests the hardware-aware path against the 8 verified vectors from the roadmap's Appendix A:

```python
@torch.no_grad()
def parity_c2c_against_verified_vectors(device: torch.device = torch.device("cpu")) -> dict:
    """
    Run the 8 verified test vectors through c2c_mac with hardware_aware=True
    and compare against measured V_OA values from schematic verification.

    Returns a dict with per-vector error in mV; max error should be < 50 mV
    (within the documented calibration uncertainty).
    """
    test_vectors = [
        # (W, IA, expected_V_OA_mV)
        ([0, 0, 0, 0],    [0, 0, 0, 0],     48.40),
        ([0, 0, 0, 0],    [15, 15, 15, 15], 49.59),
        ([1, 2, 4, 8],    [15, 0, 0, 0],    9.53),
        ([1, 2, 4, 8],    [0, 0, 0, 15],    84.08),
        ([1, 2, 4, 8],    [15, 15, 15, 15], 147.79),
        ([15, 15, 15, 15], [3, 3, 3, 3],    268.39),
        ([15, 15, 15, 15], [7, 7, 7, 7],    380.48),
        ([15, 15, 15, 15], [15, 15, 15, 15], 603.24),
    ]

    results = {}
    for i, (W, IA, V_OA_meas) in enumerate(test_vectors, start=1):
        # Convert to INT4 tensors and run through c2c_mac
        w_q = torch.tensor([W], dtype=torch.int8, device=device)  # [1, 4]
        x_q = torch.tensor([IA], dtype=torch.int8, device=device)  # [1, 4]
        # Use unit scales for direct mV interpretation
        sw = torch.tensor(1.0, device=device)
        sx = torch.tensor(1.0, device=device)
        y_mac = c2c_mac(w_q, x_q, sw, sx, None, None, hardware_aware=True)
        # Compare (with appropriate unit conversion)
        v_oa_predicted_mV = float(y_mac.item()) * 1.1 / 1024 * 1000  # rough conversion
        error_mV = abs(V_OA_meas - v_oa_predicted_mV)
        results[f"V{i}"] = {
            "V_OA_measured_mV": V_OA_meas,
            "V_OA_predicted_mV": v_oa_predicted_mV,
            "error_mV": error_mV,
        }
    return results
```

This becomes a regression test ensuring the hardware-aware model stays calibrated to the schematic-verified data.

#### Change E-2: Pass `hardware_aware=True` in `forward_int4_mlp` 🟠 Important

Update `forward_int4_mlp` (lines 39–54) so the INT4 PTQ accuracy reflects what the hardware actually does:

```python
@torch.no_grad()
def forward_int4_mlp(model: MicroMLP, x: torch.Tensor, hardware_aware: bool = True) -> torch.Tensor:
    """Layerwise INT4 weight + uint4 activation MAC + ReLU (PTQ-style, hardware-calibrated)."""
    x = model.flatten(x)
    for fc, relu in [(model.fc1, model.relu1), (model.fc2, model.relu2)]:
        w_q, sw = symmetric_quantize_int4(fc.weight)
        x_q, sx, shift = quantize_uint4(x)
        x = c2c_mac(w_q, x_q, sw, sx, fc.bias, shift_x=shift, hardware_aware=hardware_aware)
        x = relu(x)
    w3_q, sw3 = symmetric_quantize_int4(model.fc3.weight)
    x_q, sx, shift = quantize_uint4(x)
    logits = c2c_mac(w3_q, x_q, sw3, sx, model.fc3.bias, shift_x=shift, hardware_aware=hardware_aware)
    return logits
```

**Expected impact:** The `int4_ptq_test_accuracy` metric in `train_baseline.py` output will change after this — it'll drop somewhat because the model wasn't trained for the gain-pattern dependence. This drop is itself a meaningful number for the thesis (it quantifies the "PTQ-only without HWA" performance on hardware).

If you want to preserve the pre-change number for reporting purposes, ALSO save it with the ideal flag:

```python
metrics = {
    "fp32_test_accuracy": fp32_acc,
    "int4_ptq_test_accuracy_ideal": float(accuracy_int4(model, test_loader, dev, hardware_aware=False)),
    "int4_ptq_test_accuracy_hardware": float(accuracy_int4(model, test_loader, dev, hardware_aware=True)),
    ...
}
```

This gives you both numbers for the thesis chapter.

---

### 2.5 `src/hwa_cim/plots.py` — Calibration markers 🟡 Moderate

#### Change P-1: Update PDK marker default to verified operating point 🟡 Moderate

Line 113: `pdk_marker: float = 0.10` → `pdk_marker: float = 0.17`

(This is the integrated operating point per Change C-4.)

#### Change P-2: Add verified-vector overlay to parasitic sweep plot 🟢 Optional

In `plot_parasitic_sweep()`, after computing the metric curve, add a horizontal line marking the measured INL (±0.41 LSB worst case from standalone) and the integrated INL bound. This visually anchors the simulation to verified data.

```python
plt.axhline(0.41 / 16, color="C2", linestyle=":", alpha=0.5,
            label="Standalone INL (verified, 0.41 LSB)")
```

---

### 2.6 `src/hwa_cim/config.py` — Add hardware calibration knobs 🟠 Important

The current `HWAConfig` has `gamma_weight`, `alpha_clip`, `adc_bits`, `parasitic_ratio`. Missing knobs:

```python
@dataclass
class HWAConfig:
    """Noise-aware training knobs (Phases 2–3)."""
    gamma_weight: float = 0.02
    alpha_clip: float = 3.0
    adc_bits: int = 4
    use_adc: bool = True
    parasitic_ratio: float = 0.17  # Updated default to integrated operating point
    noise_mode: str = "synthetic"
    noise_profile_csv: Optional[Path] = None
    hardware_aware: bool = True  # NEW — apply calibrated gain/offset
    g_eff_sparse: float = 0.62   # NEW — calibration constant (rarely overridden)
    g_eff_dense: float = 0.44    # NEW — calibration constant
    offset_dense: float = 50e-3  # NEW — calibration constant (V)
```

Plumb these through to the `NoisyQuantLinear` constructor wherever it's instantiated.

---

## 3. New Files to Create

### 3.1 `calibration.yaml` (repo root) 🟢 Optional but recommended

Shared configuration with the verification roadmap. Contents:

```yaml
# Hardware calibration constants for HWA-CIM
# Source: CIM_SRAM_Verification_Software_Roadmap.md Section 3
# Updated when post-PEX re-extraction completes (currently pre-layout)

version: 1.0
last_calibrated: 2026-05-09
calibration_source: schematic_simulation
calibration_status: pre_layout

ladder:
  c_unit_F: 15.1e-15        # MOMCAPS_SY_MMKF
  c_series_F: 30.2e-15
  n_bits: 4
  g_eff_standalone: 0.831
  inl_worst_LSB: 0.41

integrated_macro:
  g_eff_sparse: 0.62
  g_eff_dense: 0.44
  offset_dense_V: 50e-3
  population_sparse_max: 4
  population_dense_min: 12
  inter_program_residual_V: 50e-3

dac:
  resolution_bits: 4
  v_ref_V: 1.1
  lsb_V: 0.06875

adc:
  resolution_bits: 4
  conversion_cycles: 6
  v_ref_V: 1.1

simulation:
  process: UMC65nm
  variant: G-4M
  vdd_V: 1.1
  clock_period_ns: 50
  clock_frequency_MHz: 20

phase5_status:
  monte_carlo_pending: true
  pex_netlist_pending: true
  csv_format_locked: true
```

Then create `src/hwa_cim/calibration.py` to load this YAML:

```python
"""Load calibration.yaml; provides typed accessors."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Calibration:
    g_eff_sparse: float
    g_eff_dense: float
    offset_dense_V: float
    population_sparse_max: int
    population_dense_min: int
    g_eff_standalone: float
    inl_worst_LSB: float
    inter_program_residual_V: float
    
    @classmethod
    def load(cls, path: Path = Path("calibration.yaml")) -> "Calibration":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(
            g_eff_sparse=cfg["integrated_macro"]["g_eff_sparse"],
            g_eff_dense=cfg["integrated_macro"]["g_eff_dense"],
            offset_dense_V=cfg["integrated_macro"]["offset_dense_V"],
            population_sparse_max=cfg["integrated_macro"]["population_sparse_max"],
            population_dense_min=cfg["integrated_macro"]["population_dense_min"],
            g_eff_standalone=cfg["ladder"]["g_eff_standalone"],
            inl_worst_LSB=cfg["ladder"]["inl_worst_LSB"],
            inter_program_residual_V=cfg["integrated_macro"]["inter_program_residual_V"],
        )
```

Then in `c2c.py`, replace hardcoded constants with calls to `Calibration.load()`. This means **a single source of truth** between the verification software roadmap and the HWA training pipeline.

### 3.2 `tests/test_verified_vectors.py` 🟢 Optional but strong regression guard

```python
"""Regression tests against 8 verified schematic-level test vectors."""
import pytest
import torch
from hwa_cim.evaluate import parity_c2c_against_verified_vectors


def test_8_verified_vectors_within_calibration_uncertainty():
    """All 8 vectors should fall within 50 mV of measured V_OA."""
    results = parity_c2c_against_verified_vectors()
    for vec, data in results.items():
        assert data["error_mV"] < 50, (
            f"Vector {vec}: error {data['error_mV']:.1f} mV exceeds 50 mV tolerance. "
            f"Predicted {data['V_OA_predicted_mV']:.1f}, measured {data['V_OA_measured_mV']:.1f}"
        )


def test_parity_ideal_mode_matches_linear():
    """Ideal mode of c2c_mac must match nn.Linear within float precision."""
    from hwa_cim.evaluate import parity_linear_vs_c2c
    err = parity_linear_vs_c2c()
    assert err < 1e-4, f"Parity test failed: {err}"
```

### 3.3 `docs/CALIBRATION_HISTORY.md` 🟢 Optional but good for traceability

A versioned log of calibration changes:

```markdown
# Calibration History

## v1.0 — 2026-05-09 (pre-layout, schematic verification)
Source: CIM_SRAM_Verification_Software_Roadmap.md
- G_eff_sparse = 0.62 (verified across WP-B test vectors)
- G_eff_dense = 0.44 (verified across WP-C test vectors)
- Offset_dense = 50 mV (post auto-reset switch)

## v2.0 — TBD (post-PEX, post-MC)
- Replace synthetic noise with measured σ per output code
- Update G_eff based on parasitic-extracted netlist behavior
- ...
```

---

## 4. Limitations to Document (Mirroring Verification Roadmap)

Add a new section to the repo README under "Hardware Considerations" or similar:

```markdown
## Hardware Calibration & Known Limitations

This pipeline models a CIM-SRAM macro that exhibits **gain-pattern-dependent
behavior** — the analog transfer function depends on the cumulative bit
population of the weight tile being multiplied:

| Weight bit population (per 4-row tile) | G_eff  | Offset (mV) |
| --- | --- | --- |
| ≤ 4 (sparse)  | 0.62  | 0    |
| ≥ 12 (dense)  | 0.44  | 50   |
| 5–11 (mixed)  | interp | interp |

Set `hardware_aware=True` on `NoisyQuantLinear` (default for training) to
apply this transfer function. For pure software baselines (e.g., parity
checks), pass `hardware_aware=False`.

Other documented limitations carried over from schematic verification:
- L-1: G_eff is **systematic**, not random — HWA training must learn it via
  the modified forward pass, not via Gaussian noise injection alone.
- L-2: Residual ~50 mV offset is included in the dense regime, post auto-reset.
- L-3: SAR ADC phase alignment limitation (`hwa_cim` consumes the analog
  output before ADC; the SAR alignment issue affects the integration test
  harness, not the HWA training math directly).
- **L-NEW (per Wang Sec IV-A):** Transfer-curve **overlaps** (multiple input codes producing
  similar output voltages) CAN be compensated by HWA training (which is a learned
  form of digital pre-distortion). Transfer-curve **gaps** (missing intermediate
  output codes) CANNOT be compensated by any software technique. If post-layout
  MC reveals gaps in the transfer curve, this is a hardware-level redesign issue,
  not a software fix. Honest thesis framing: HWA training compensates for systematic
  nonlinearity and Gaussian-distributed mismatch noise, but cannot fix missing codes.
- L-9 through L-13 (layout-pending): See verification roadmap.
```

---

## 5. Phase 5 Integration Specification

### What Cadence delivers to `hwa-cim` (Phase 5 hand-off)

A single CSV file at `noise_profiles/mc_profile_<rev>.csv`:

```csv
input_code,weight_population,ideal_output,mean_output,sigma,CSNR_dB
0,0,0.000,0.050,0.012,15.2
0,4,0.000,0.000,0.005,42.1
1,4,0.069,0.061,0.010,14.8
...
```

**How to produce this CSV (Cadence side):**

1. Pick the verified 8-vector test set as your Monte Carlo input stimuli.
2. For each weight-population class (0, 4, 12, 16) and each input code (0..15), run Monte Carlo with N ≥ 100 iterations.
3. Record output voltage for each iteration.
4. Compute mean and σ per (weight_population, input_code) pair.
5. Compute CSNR = 10·log10(mean(DMAC²) / mean((DMAC − AMAC)²)).
6. Export as the CSV above.

### What `hwa-cim` does with the CSV

1. `NoiseProfileCSV.load(path)` parses it.
2. `hwa-train-hwa --noise-mode csv --noise-profile path/to/mc_profile.csv` activates per-code noise injection.
3. `noisy_forward_from_profile()` applies the per-code sigma per output element.
4. Phase 5 deliverable: the thesis bar chart with **real-noise** comparison.

### Layout Team Deliverables — Explicit Contract

For the software to consume layout-extracted data, the layout phase must produce **all** of the following:

| # | Deliverable | Format | Purpose | Required for |
|---|---|---|---|---|
| D-1 | PEX-extracted Spectre netlist of the integrated 4-row macro | `.sp` or `.scs` | MC simulation substrate | All MC analysis |
| D-2 | Layout DRC/LVS clean confirmation report | Text/PDF | Confirm netlist validity | Reviewer trust |
| D-3 | Monte Carlo CSV per schema above (Section 5) | `.csv` | Drives `--noise-mode csv` in HWA training | Phase 5 |
| D-4 | Single-run PEX simulation results re-running the verified 8-vector test | CSV with `[marker, V_OA_measured, V_OA_predicted, delta_mV]` | Re-calibration of G_eff/offset after parasitics | Verification roadmap Phase 2.1 |
| D-5 | Extracted unit cap value (post-parasitic) | Single number (fF) | Updates `c_unit` in calibration.yaml | Documentation |
| D-6 | Extracted 2C inter-branch cap value | Single number (fF) | Updates `c_series` in calibration.yaml | Documentation |
| D-7 | (Optional) Temperature corner MC at TT/SS/FF × {0, 27, 85} °C | CSV variant | PVT robustness claim (per Wang Fig 27) | Optional thesis enhancement |
| D-8 | (Optional) V_DD corner sweep ±10% | CSV variant | Voltage robustness | Optional thesis enhancement |

**Numerical targets (from Wang et al. JSSC 2023):**
- Uncompensated 1σ MAC error: **≤ 0.89%** (Wang's measured value on 22nm FinFET; UMC 65nm may exceed this)
- Pre-distortion-compensated 1σ MAC error: **≤ 0.5%** (Wang's measured)
- Transfer curve INL: **≤ 0.5 LSB** at 4-bit (≤ 34 mV) — pre-layout achieves 0.41 LSB standalone
- These are *targets*, not pass/fail criteria for the thesis; the thesis claim is comparative (HWA improves accuracy by X%) not absolute (HWA achieves Wang's numbers).

**Array size — LOCKED: 4×4 (decision date: 2026-05-11):**
The layout phase will implement the 4×4 array matching the schematic-verified build. The thesis presentation slide referring to 8×8 should be updated to reflect 4×4 (a one-line edit). This decision is final due to time constraints and is justified academically — the 4×4 build is sufficient to demonstrate the C-2C charge-domain MAC architecture, the HWA training methodology, and the hardware-software co-design loop. Scaling to 8×8 is well-defined future work.

**MC test plan size for 4×4 (locked):**
- 4 weight-population classes × 16 input codes × 100 MC iterations = **6,400 MC runs per corner**
- At TT corner alone (single PVT point): ~6,400 runs (feasible within a few days of Spectre runtime)
- If PVT included: 6,400 × 9 corners = ~57,600 runs (still feasible but plan ahead)

---

## 6. Testing Plan (After Changes)

```bash
# Validate baseline still passes (ideal path)
pytest tests/test_parity.py -v

# Validate hardware-aware regression
pytest tests/test_verified_vectors.py -v

# Re-run Phase 1 baseline (should produce identical INT4 accuracy as before)
hwa-train-baseline --out-dir results/run_baseline_v2

# Run Phase 2 noisy eval with new hardware-aware path
hwa-eval-noisy --checkpoint results/run_baseline_v2/best.pt \
               --out results/run_baseline_v2/noisy_eval.json --gamma 0.02

# Run Phase 3 HWA training; this should now learn against systematic gain shift
hwa-train-hwa --out-dir results/run_hwa_v2 --gamma 0.02 --alpha 3.0

# Verify thesis 3-bar plot regenerates correctly
hwa-plot-thesis --baseline-dir results/run_baseline_v2 \
                --hwa-checkpoint results/run_hwa_v2/best.pt \
                --noisy-eval-json results/run_baseline_v2/noisy_eval.json
```

**Acceptance:**
- INT4 PTQ accuracy in baseline = within 1% of pre-change value (the math hasn't changed for ideal mode).
- HWA noisy accuracy = improvement of ≥ 5% over Phase 2 noisy eval (this is the central thesis claim).
- Verified vectors regression test: all 8 within 50 mV of measured.

---

## 7. Implementation Order (Recommended)

Suggested order to minimize breakage and maximize learning per increment:

| # | Change | Why this order |
|---|---|---|
| 1 | C-1: Add constants (c2c.py) | Pure additions, no behavior change |
| 2 | C-2: Add compute_g_eff, compute_offset functions | Pure additions |
| 3 | C-3: Add hardware_aware flag to c2c_mac | Defaults to False = no behavior change yet |
| 4 | E-1: Update parity test (explicit hardware_aware=False) | Compatibility |
| 5 | E-2: Update forward_int4_mlp to use hardware_aware=True | INT4 PTQ now reflects hardware |
| 6 | N-1, N-2: Refine CSV ingestion | Independent of forward path changes |
| 7 | L-1, L-2: Wire hardware_aware through layers + models | Now HWA training behavior changes |
| 8 | Test: re-run baseline + capture new int4_ptq_test_accuracy_hardware metric | Validate baseline |
| 9 | Test: run Phase 2 with hardware_aware=True | First behavior-change validation |
| 10 | Test: run Phase 3 HWA training; document delta vs old baseline | Capture central thesis result |
| 11 | P-1, P-2: Update plots with verified operating point | Polish; doesn't affect training |
| 12 | Calibration YAML + module | Single source of truth |
| 13 | Documentation, README updates, CALIBRATION_HISTORY.md | Finalize |

---

## 8. Open Questions for User

Before implementing, please confirm:

1. ~~Array size resolution~~ — **RESOLVED 2026-05-11: 4×4 locked.** Update slide 44 of thesis presentation to reflect 4×4 instead of 8×8.

2. **Do you want the calibration constants in `calibration.yaml` (Change 3.1) or hardcoded in `c2c.py`?** YAML is cleaner long-term but adds a dependency on PyYAML. Hardcoded is simpler.

3. **Should the per-row-population approximation in `compute_g_eff()` (Change C-2) use the average bit count, or should we partition the weight matrix into explicit 4-input tiles?** The former is simpler; the latter is more accurate but requires reshape logic. I recommend starting with the average for Phase 3 and refining to explicit tiles in Phase 5 alongside MC integration.

4. **For `parity_c2c_against_verified_vectors()` (Change E-1)**, the unit conversion in my draft uses `* 1.1 / 1024 * 1000` to go from "INT4 MAC accumulator units" to "mV". This depends on assumptions about what unit `c2c_mac` returns. Please double-check this conversion when you implement, or send me the actual return scale and I'll refine.

5. **Should I generate any of these as actual code files** (e.g., `calibration.py`, `tests/test_verified_vectors.py`) ready for `git add`, or do you prefer to type them in yourself from the spec above? I can produce them if you want a copy-paste-ready PR.

6. **PVT corners — required or optional?** Wang's paper reports INL/DNL across temperature (0–85°C, Fig 27a) and supply voltage (Fig 27b). My MD treats these as optional (D-7, D-8). If your thesis examiner expects PVT, escalate to required. At 4×4 scale with PVT, total MC runs are ~57,600 (still feasible).

---

*Last updated: 2026-05-11 — comprehensive change list for hwa-cim repository derived from full review of src/hwa_cim/*.py (c2c.py, noise.py, layers.py, models.py, evaluate.py, train_baseline.py, train_hwa.py, train_distill.py, plots.py, config.py, data.py, quantization.py, utils_io.py); identified 12 specific changes across 5 source files plus 3 new files; prioritized as 3 critical, 5 important, 4 moderate, 3 optional; preserves baseline behavior via hardware_aware flag default for backward compatibility; integration boundary with verification roadmap documented for Phase 5; **updated 2026-05-11 evening:** added Wang's gap-vs-overlap caveat (L-NEW), Wang's numerical error targets (0.89% / 0.5%), explicit Layout Team Deliverables contract (D-1 through D-8), and PVT corner question after project knowledge verification; **updated 2026-05-11 final:** 4×4 array size LOCKED per user decision (time-constrained, academically justified); slide 44 of thesis presentation needs one-line update from 8×8 to 4×4; MC test plan size locked at 6,400 runs per corner.*
