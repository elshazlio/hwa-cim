"""
Ideal C-2C MAC and parasitic ladder model (thesis / Wang-style abstraction).

Ideal MAC matches int8 dequantized matmul used with nn.Linear + symmetric quant.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from hwa_cim.quantization import dequantize_int8_matmul


def c2c_mac(
    weights_int8: torch.Tensor,
    activations_int8: torch.Tensor,
    scale_w: torch.Tensor,
    scale_x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Noiseless C-2C MAC: int32 accumulation + dequantization.

    weights_int8: [out_features, in_features]
    activations_int8: [batch, in_features]
    """
    return dequantize_int8_matmul(activations_int8, weights_int8, scale_x, scale_w, bias)


def ladder_ideal_output(bits: torch.Tensor, v_ref: float = 1.0) -> torch.Tensor:
    """
    bits: bool or 0/1 tensor [..., 8] MSB..LSB or LSB..MSB — we assume bit i has weight 2^i.
    """
    if bits.dtype != torch.bool:
        bits = bits > 0.5
    powers = torch.tensor([2.0**i for i in range(bits.shape[-1])], device=bits.device, dtype=torch.float32)
    return v_ref * (bits.to(torch.float32) * powers).sum(dim=-1) / (2.0 ** (bits.shape[-1]) - 1 + 1e-8)


class C2CLadderWithParasitics:
    """
    Models binary-weighted ladder output with bottom-plate parasitic ratio.

    Ideal unit caps C, 2C; parasitic p·C on bottom plate reduces effective 2:1 ratio.
    We use a simplified scalar model: each bit i effective weight = 2^i * (1 - eps * r)
    with nonlinear term r = parasitic_ratio in [0, 0.5].
    """

    def __init__(self, n_bits: int = 8, parasitic_ratio: float = 0.0) -> None:
        self.n_bits = n_bits
        self.parasitic_ratio = float(np.clip(parasitic_ratio, 0.0, 0.5))

    def transfer(self, code: int, v_ref: float = 1.0) -> float:
        """Digital code 0..255 to analog output (normalized 0..1)."""
        bits = np.array([(code >> i) & 1 for i in range(self.n_bits)], dtype=np.float64)
        r = self.parasitic_ratio
        # Effective bit weights shrink and couple: qualitative Wang Fig. 11 behavior
        weights = np.array([(2.0**i) * (1.0 - 0.8 * r * (i + 1) / self.n_bits) for i in range(self.n_bits)])
        raw = float(np.dot(bits, weights))
        ideal = float(np.dot(bits, np.array([2.0**i for i in range(self.n_bits)])))
        denom = (2.0**self.n_bits) - 1.0
        # Normalize like ideal full-scale
        if ideal < 1e-12:
            return 0.0
        return v_ref * (raw / denom) * (denom / ((2.0**self.n_bits - 1) * (1.0 - 0.3 * r)))

    def sweep_codes(self) -> Tuple[np.ndarray, np.ndarray]:
        codes = np.arange(256, dtype=np.int32)
        outs = np.array([self.transfer(int(c)) for c in codes])
        return codes, outs


def ladder_nonlinearity_metric(parasitic_ratio: float) -> float:
    """Max deviation from ideal straight line (normalized codes vs output) for thesis plots."""
    lad = C2CLadderWithParasitics(n_bits=8, parasitic_ratio=parasitic_ratio)
    codes, outs = lad.sweep_codes()
    ideal = codes.astype(np.float64) / 255.0
    return float(np.max(np.abs(outs - ideal)))
