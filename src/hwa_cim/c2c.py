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


def c2c_mac(
    weights_q: torch.Tensor,
    activations_q: torch.Tensor,
    scale_w: torch.Tensor,
    scale_x: torch.Tensor,
    bias: torch.Tensor | None = None,
    shift_x: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Noiseless C-2C MAC: int32 accumulation + dequantization.

    weights_q: [out_features, in_features] signed INT4 in int8 storage
    activations_q: [batch, in_features] signed INT4 or uint4 [0..15] in int8 storage
    shift_x: optional scalar from quantize_uint4 (activations).
    """
    return dequantize_int4_matmul(activations_q, weights_q, scale_x, scale_w, bias, shift_x)


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
