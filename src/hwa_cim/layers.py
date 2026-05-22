"""Noisy quantized linear layers, ADC quantization, STE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hwa_cim.c2c import compute_g_eff, compute_offset
from hwa_cim.noise import (
    NoiseProfileCSV,
    additive_weight_noise,
    additive_weight_noise_sigma,
    noisy_forward_from_profile,
)
from hwa_cim.quantization import clip_weights_by_std, fake_quantize_int4_ste, symmetric_quantize_int4


def adc_quantize_ste(y: torch.Tensor, bits: int = 4) -> torch.Tensor:
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
    Linear with optional weight clipping, fake INT4 quant, AFM-style noise, optional ADC on output.
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
        adc_bits: int = 4,
        noise_mode: str = "synthetic",
        sigma_global: float | None = None,
        noise_profile: NoiseProfileCSV | None = None,
        hardware_aware: bool = False,
        g_eff_sparse: float | None = None,
        g_eff_dense: float | None = None,
        offset_dense_v: float | None = None,
        population_sparse_max: int | None = None,
        population_dense_min: int | None = None,
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
        self.noise_profile = noise_profile
        self.hardware_aware = hardware_aware
        self._g_eff_sparse = g_eff_sparse
        self._g_eff_dense = g_eff_dense
        self._offset_dense_v = offset_dense_v
        self._population_sparse_max = population_sparse_max
        self._population_dense_min = population_dense_min
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.linear.weight
        b = self.linear.bias
        w = clip_weights_by_std(w, self.alpha_clip)
        w_q = fake_quantize_int4_ste(w)
        if self.training and (
            self.gamma > 0
            or (self.sigma_global is not None and self.sigma_global > 0)
            or (self.noise_mode == "csv" and self.noise_profile is not None)
        ):
            if self.noise_mode == "csv" and self.noise_profile is not None:
                w_noisy = self.noise_profile.apply_weight_noise(w_q)
            elif self.noise_mode == "csv" and self.sigma_global is not None and self.sigma_global > 0:
                w_noisy = additive_weight_noise_sigma(w_q, self.sigma_global)
            else:
                w_noisy = additive_weight_noise(w_q, self.gamma)
        else:
            w_noisy = w_q
        y = F.linear(x, w_noisy, b)
        if self.training and self.noise_mode == "csv" and self.noise_profile is not None:
            y = noisy_forward_from_profile(y, self.noise_profile)
        if self.hardware_aware:
            w_pop_q, _ = symmetric_quantize_int4(w)
            g_eff = compute_g_eff(
                w_pop_q,
                g_eff_sparse=self._g_eff_sparse,
                g_eff_dense=self._g_eff_dense,
                population_sparse_max=self._population_sparse_max,
                population_dense_min=self._population_dense_min,
            ).to(dtype=y.dtype, device=y.device)
            offset = compute_offset(
                w_pop_q,
                offset_dense_v=self._offset_dense_v,
                population_sparse_max=self._population_sparse_max,
                population_dense_min=self._population_dense_min,
            ).to(dtype=y.dtype, device=y.device)
            y = y * g_eff.unsqueeze(0) + offset.unsqueeze(0)
        if self.use_adc:
            y = adc_quantize_ste(y, self.adc_bits)
        return y
