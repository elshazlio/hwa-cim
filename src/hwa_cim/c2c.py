"""
Ideal C-2C MAC and parasitic ladder model (thesis / Wang-style abstraction).

Ideal MAC matches INT4 dequantized matmul used with nn.Linear + quant helpers.
Default ladder is 4-bit (k=4); default caps reference MOMCAPS_SY_MMKF extractions.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from hwa_cim.quantization import dequantize_int4_matmul

# Schematic-level integrated macro calibration (pre-layout). See thesis verification notes.
G_EFF_SPARSE = 0.62
G_EFF_DENSE = 0.44
OFFSET_DENSE_V = 50e-3
POPULATION_SPARSE_MAX = 4
POPULATION_DENSE_MIN = 12


def _popcount_low4(abs_w: torch.Tensor) -> torch.Tensor:
    """Popcount of magnitude in lower 4 bits (INT4 abs range 0..8)."""
    abs_w = abs_w.to(torch.int32)
    return (
        (abs_w & 1)
        + ((abs_w >> 1) & 1)
        + ((abs_w >> 2) & 1)
        + ((abs_w >> 3) & 1)
    )


def _pop_per_output_row(weights_q: torch.Tensor) -> torch.Tensor:
    """Average total popcount per 4-input tile along each output row [out]."""
    abs_w = weights_q.abs().to(torch.int32)
    pop_per_elem = _popcount_low4(abs_w).to(torch.float32)
    n_tiles = max(1, weights_q.shape[1] // 4)
    return pop_per_elem.sum(dim=1) / float(n_tiles)


def compute_g_eff(
    weights_q: torch.Tensor,
    *,
    g_eff_sparse: float | None = None,
    g_eff_dense: float | None = None,
    population_sparse_max: int | None = None,
    population_dense_min: int | None = None,
) -> torch.Tensor:
    """
    Effective gain per output row from INT4 weight bit population (first-order tile model).

    weights_q: [out_features, in_features] int8 in [-8, 7].
    Returns float tensor [out_features] on same device as weights_q.
    """
    gs = float(G_EFF_SPARSE if g_eff_sparse is None else g_eff_sparse)
    gd = float(G_EFF_DENSE if g_eff_dense is None else g_eff_dense)
    p_lo = int(POPULATION_SPARSE_MAX if population_sparse_max is None else population_sparse_max)
    p_hi = int(POPULATION_DENSE_MIN if population_dense_min is None else population_dense_min)

    pop = _pop_per_output_row(weights_q)
    device, dtype = weights_q.device, torch.float32
    g_eff = torch.empty(pop.shape, device=device, dtype=dtype)
    sparse = pop <= p_lo
    dense = pop >= p_hi
    mid = ~(sparse | dense)
    g_eff[sparse] = gs
    g_eff[dense] = gd
    denom = float(p_hi - p_lo)
    if denom > 0:
        f = (pop[mid] - float(p_lo)) / denom
        g_eff[mid] = gs + f * (gd - gs)
    else:
        g_eff[mid] = gs
    return g_eff


def compute_offset(
    weights_q: torch.Tensor,
    *,
    offset_dense_v: float | None = None,
    population_sparse_max: int | None = None,
    population_dense_min: int | None = None,
) -> torch.Tensor:
    """Residual offset per output row (dense regime), volts. Shape [out_features]."""
    off_d = float(OFFSET_DENSE_V if offset_dense_v is None else offset_dense_v)
    p_lo = int(POPULATION_SPARSE_MAX if population_sparse_max is None else population_sparse_max)
    p_hi = int(POPULATION_DENSE_MIN if population_dense_min is None else population_dense_min)

    pop = _pop_per_output_row(weights_q)
    device, dtype = weights_q.device, torch.float32
    offset = torch.zeros(pop.shape, device=device, dtype=dtype)
    sparse = pop <= p_lo
    dense = pop >= p_hi
    mid = ~(sparse | dense)
    offset[sparse] = 0.0
    offset[dense] = off_d
    denom = float(p_hi - p_lo)
    if denom > 0:
        f = (pop[mid] - float(p_lo)) / denom
        offset[mid] = f * off_d
    else:
        offset[mid] = 0.0
    return offset


def c2c_mac(
    weights_q: torch.Tensor,
    activations_q: torch.Tensor,
    scale_w: torch.Tensor,
    scale_x: torch.Tensor,
    bias: torch.Tensor | None = None,
    shift_x: torch.Tensor | None = None,
    *,
    hardware_aware: bool = False,
    g_eff_sparse: float | None = None,
    g_eff_dense: float | None = None,
    offset_dense_v: float | None = None,
    population_sparse_max: int | None = None,
    population_dense_min: int | None = None,
) -> torch.Tensor:
    """
    Noiseless C-2C MAC: int32 accumulation + dequantization.

    weights_q: [out_features, in_features] signed INT4 in int8 storage
    activations_q: [batch, in_features] signed INT4 or uint4 [0..15] in int8 storage
    shift_x: optional scalar from quantize_uint4 (activations).
    hardware_aware: if True, apply schematic gain/offset vs weight population per row.
    """
    y = dequantize_int4_matmul(activations_q, weights_q, scale_x, scale_w, bias, shift_x)
    if hardware_aware:
        g_eff = compute_g_eff(
            weights_q,
            g_eff_sparse=g_eff_sparse,
            g_eff_dense=g_eff_dense,
            population_sparse_max=population_sparse_max,
            population_dense_min=population_dense_min,
        ).to(dtype=y.dtype)
        offset = compute_offset(
            weights_q,
            offset_dense_v=offset_dense_v,
            population_sparse_max=population_sparse_max,
            population_dense_min=population_dense_min,
        ).to(dtype=y.dtype)
        y = y * g_eff.unsqueeze(0) + offset.unsqueeze(0)
    return y


def ladder_ideal_output(bits: torch.Tensor, v_ref: float = 1.0) -> torch.Tensor:
    """
    bits: bool or 0/1 tensor [..., n_bits]; bit i weight 2^i (LSB at i=0).
    """
    if bits.dtype != torch.bool:
        bits = bits > 0.5
    n = bits.shape[-1]
    powers = torch.tensor([2.0**i for i in range(n)], device=bits.device, dtype=torch.float32)
    denom = 2.0**n - 1.0 + 1e-8
    return v_ref * (bits.to(torch.float32) * powers).sum(dim=-1) / denom


# MOMCAPS_SY_MMKF characterized targets (UMC 65nm FDK), farads
C_UNIT_MOM_DEFAULT = 15.1e-15
C_SERIES_MOM_DEFAULT = 30.2e-15


class C2CLadderWithParasitics:
    """
    Binary-weighted ladder output with parasitic ratio (MOM-oriented sweep 0–20%).

    Ideal unit caps C, 2C (defaults from MOM PCell extractions); scalar model distorts
    effective bit weights vs. Wang Fig. 11. c_unit / c_series are stored for calibration hooks.
    """

    def __init__(
        self,
        n_bits: int = 4,
        parasitic_ratio: float = 0.0,
        *,
        c_unit: float = C_UNIT_MOM_DEFAULT,
        c_series: float = C_SERIES_MOM_DEFAULT,
    ) -> None:
        self.n_bits = n_bits
        self.parasitic_ratio = float(np.clip(parasitic_ratio, 0.0, 0.2))
        self.c_unit = float(c_unit)
        self.c_series = float(c_series)

    def transfer(self, code: int, v_ref: float = 1.0) -> float:
        """Digital code 0..2^n-1 to analog output (normalized full-scale)."""
        bits = np.array([(code >> i) & 1 for i in range(self.n_bits)], dtype=np.float64)
        r = self.parasitic_ratio
        weights = np.array([(2.0**i) * (1.0 - 0.8 * r * (i + 1) / self.n_bits) for i in range(self.n_bits)])
        raw = float(np.dot(bits, weights))
        ideal = float(np.dot(bits, np.array([2.0**i for i in range(self.n_bits)])))
        denom = (2.0**self.n_bits) - 1.0
        if ideal < 1e-12:
            return 0.0
        return v_ref * (raw / denom) * (denom / ((2.0**self.n_bits - 1) * (1.0 - 0.3 * r)))

    def sweep_codes(self) -> Tuple[np.ndarray, np.ndarray]:
        nlev = 2**self.n_bits
        codes = np.arange(nlev, dtype=np.int32)
        outs = np.array([self.transfer(int(c)) for c in codes])
        return codes, outs


def ladder_nonlinearity_metric(parasitic_ratio: float) -> float:
    """Max deviation from ideal straight line (normalized codes vs output) for thesis plots."""
    lad = C2CLadderWithParasitics(n_bits=4, parasitic_ratio=parasitic_ratio)
    codes, outs = lad.sweep_codes()
    ideal = codes.astype(np.float64) / float(2**lad.n_bits - 1)
    return float(np.max(np.abs(outs - ideal)))
