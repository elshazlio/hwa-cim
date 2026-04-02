"""Noisy quantized linear layers, ADC quantization, STE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hwa_cim.noise import additive_weight_noise, additive_weight_noise_sigma
from hwa_cim.quantization import clip_weights_by_std, fake_quantize_int8_ste


def adc_quantize_ste(y: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Uniform ADC quantization with STE; per-tensor dynamic range."""
    levels = float(2**bits - 1)
    y_min = y.detach().amin()
    y_max = y.detach().amax()
    rng = (y_max - y_min).clamp(min=1e-8)
    scale = rng / levels
    y_q = ((y - y_min) / scale).round().clamp(0, levels)
    y_dq = y_q * scale + y_min
    return y + (y_dq - y).detach()


class NoisyQuantLinear(nn.Module):
    """
    Linear with optional weight clipping, fake int8 quant, AFM-style noise, optional ADC on output.
    Training: STE through quant + ADC; noise active if gamma > 0 or csv sigma.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        gamma: float = 0.02,
        alpha_clip: float = 3.0,
        use_adc: bool = True,
        adc_bits: int = 8,
        noise_mode: str = "synthetic",
        sigma_global: float | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gamma = gamma
        self.alpha_clip = alpha_clip
        self.use_adc = use_adc
        self.adc_bits = adc_bits
        self.noise_mode = noise_mode
        self.sigma_global = sigma_global
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.linear.weight
        b = self.linear.bias
        w = clip_weights_by_std(w, self.alpha_clip)
        w_q = fake_quantize_int8_ste(w)
        if self.training and (
            self.gamma > 0
            or (self.sigma_global is not None and self.sigma_global > 0)
        ):
            if self.noise_mode == "csv" and self.sigma_global is not None and self.sigma_global > 0:
                w_noisy = additive_weight_noise_sigma(w_q, self.sigma_global)
            else:
                w_noisy = additive_weight_noise(w_q, self.gamma)
        else:
            w_noisy = w_q
        y = F.linear(x, w_noisy, b)
        if self.use_adc:
            y = adc_quantize_ste(y, self.adc_bits)
        return y
