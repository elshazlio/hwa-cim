"""
Noise injection (AFM-style) and Phase-5 Monte Carlo CSV profile adapter.

Synthetic: W_noisy = W + gamma * max(|W|) * tau, tau ~ N(0, I)

CSV schema (header required):
    input_code,ideal_output,mean_output,sigma,CSNR_dB

Optional columns (case-insensitive, backward compatible if absent):
    weight_population, g_eff_measured, offset_measured

**Cadence boundary:** Export from Spectre Monte Carlo on the PEX netlist (or pre-PEX
schematic for bring-up). Recommended ≥100 MC iterations for exploration, ≥1000 for
thesis statistics. Rows are indexed by **ADC/digital code** (or normalized ideal output);
raw V_OA columns should be converted to the same units as training before load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import torch


def additive_weight_noise(
    weight: torch.Tensor,
    gamma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Gaussian noise scaled by gamma * max(|W|)."""
    wmax = weight.detach().abs().max().clamp(min=1e-8)
    noise = torch.randn(weight.shape, device=weight.device, dtype=weight.dtype, generator=generator)
    return weight + (gamma * wmax) * noise


def additive_weight_noise_sigma(
    weight: torch.Tensor,
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Noise with explicit sigma (Phase 5 hook)."""
    noise = torch.randn(weight.shape, device=weight.device, dtype=weight.dtype, generator=generator)
    return weight + sigma * noise


def additive_relative_output_noise(
    y: torch.Tensor,
    sigma_rel: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Gaussian output noise scaled by ``sigma_rel * max(|y|)`` (Cadence-informed stress).

    Active in training mode only (caller responsibility). Models analog output uncertainty
    after MAC shaping, not per-code weight mismatch.
    """
    if sigma_rel <= 0:
        return y
    scale = y.detach().abs().max().clamp(min=1e-8)
    noise = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=generator)
    return y + (sigma_rel * scale) * noise


def _nearest_code_indices(values: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Map scalar tensor elements to nearest index in ``codes`` (1-D, sorted ascending)."""
    flat = values.reshape(-1)
    diff = (flat.unsqueeze(1) - codes.unsqueeze(0)).abs()
    return diff.argmin(dim=1).reshape(values.shape)


@dataclass
class NoiseProfileCSV:
    """Parsed Monte Carlo noise profile for integration / reporting."""

    path: Path
    input_code: list[int]
    ideal_output: list[float]
    mean_output: list[float]
    sigma: list[float]
    csnr_db: Optional[list[float]] = None
    weight_population: Optional[list[int]] = None
    g_eff_measured: Optional[list[float]] = None
    offset_measured: Optional[list[float]] = None
    sigma_mean: float = 0.0
    sigma_max: float = 0.0
    _codes_tensor: Optional[torch.Tensor] = field(default=None, repr=False)
    _sigmas_tensor: Optional[torch.Tensor] = field(default=None, repr=False)

    @classmethod
    def load(cls, path: Path) -> "NoiseProfileCSV":
        path = Path(path)
        df = pd.read_csv(path)
        cols = {c.lower().strip(): c for c in df.columns}

        def col(*names: str) -> str:
            for n in names:
                if n in cols:
                    return cols[n]
            raise ValueError(f"CSV {path} missing one of {names}, got {list(df.columns)}")

        def col_optional(*names: str) -> Optional[str]:
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        ic = col("input_code", "code", "digital_code")
        io = col("ideal_output", "ideal")
        mo = col("mean_output", "mean")
        sg = col("sigma", "std")
        csnr_col = col_optional("csnr_db", "csnr")
        wp_col = col_optional("weight_population", "population", "pop")
        ge_col = col_optional("g_eff_measured", "g_eff")
        off_col = col_optional("offset_measured", "offset")

        sig = df[sg].astype(float).tolist()
        csnr_list = df[csnr_col].astype(float).tolist() if csnr_col else None
        wp_list = df[wp_col].astype(int).tolist() if wp_col else None
        ge_list = df[ge_col].astype(float).tolist() if ge_col else None
        off_list = df[off_col].astype(float).tolist() if off_col else None

        return cls(
            path=path,
            input_code=df[ic].astype(int).tolist(),
            ideal_output=df[io].astype(float).tolist(),
            mean_output=df[mo].astype(float).tolist(),
            sigma=sig,
            csnr_db=csnr_list,
            weight_population=wp_list,
            g_eff_measured=ge_list,
            offset_measured=off_list,
            sigma_mean=float(pd.Series(sig).mean()),
            sigma_max=float(pd.Series(sig).max()),
        )

    def codes_tensor(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self._codes_tensor is None or self._codes_tensor.device != device:
            self._codes_tensor = torch.tensor(self.input_code, device=device, dtype=dtype)
        return self._codes_tensor

    def sigmas_tensor(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if self._sigmas_tensor is None or self._sigmas_tensor.device != device:
            self._sigmas_tensor = torch.tensor(self.sigma, device=device, dtype=dtype)
        return self._sigmas_tensor

    def sigma_at_code(self, code: int) -> float:
        """Nearest-row σ for an integer code (falls back to ``sigma_mean`` if empty)."""
        if not self.input_code:
            return self.sigma_mean
        idx = min(
            range(len(self.input_code)),
            key=lambda i: abs(self.input_code[i] - code),
        )
        return float(self.sigma[idx])

    def sigma_map_for_weights(self, weight_q: torch.Tensor) -> torch.Tensor:
        """
        Per-element σ from profile by mapping |INT4 weight| to nearest ``input_code``.

        weight_q: int8 storage in [-8, 7].
        """
        device, dtype = weight_q.device, torch.float32
        codes = self.codes_tensor(device, dtype)
        sigmas = self.sigmas_tensor(device, dtype)
        mag = weight_q.abs().to(dtype)
        idx = _nearest_code_indices(mag, codes)
        return sigmas[idx]

    def apply_weight_noise(
        self,
        weight_q: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Per-element Gaussian noise using code-dependent σ."""
        sigma_map = self.sigma_map_for_weights(weight_q)
        noise = torch.randn(
            weight_q.shape, device=weight_q.device, dtype=weight_q.dtype, generator=generator
        )
        return weight_q + sigma_map.to(dtype=weight_q.dtype) * noise

    def apply_output_noise(
        self,
        ideal_output: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Per-element output noise: map activations to nearest profile code via ``ideal_output`` axis.
        """
        device = ideal_output.device
        dtype = ideal_output.dtype
        codes = self.codes_tensor(device, torch.float32)
        sigmas = self.sigmas_tensor(device, dtype)
        ideal = torch.tensor(self.ideal_output, device=device, dtype=torch.float32)
        out_min = float(ideal.min().item())
        out_max = float(ideal.max().item())
        out_range = out_max - out_min
        if out_range < 1e-8:
            return ideal_output
        normalized = (ideal_output.to(torch.float32) - out_min) / out_range
        code_span = max(float(codes.max() - codes.min()), 1e-8)
        pseudo_code = codes.min() + normalized * code_span
        idx = _nearest_code_indices(pseudo_code, ideal)
        sigma_per = sigmas[idx]
        noise = torch.randn(ideal_output.shape, device=device, dtype=dtype, generator=generator)
        return ideal_output + sigma_per * noise


def noise_scale_for_forward(
    gamma: float,
    profile: Optional[NoiseProfileCSV],
    mode: str,
) -> tuple[float, str]:
    """
    Returns (effective_gamma_or_sigma_scale, description).
    In csv mode, use mean sigma normalized by a reference (caller passes scaled gamma).
    """
    if mode == "csv" and profile is not None:
        return profile.sigma_mean, "csv_sigma_mean"
    return gamma, "synthetic_gamma"


def noisy_forward_from_profile(
    ideal_output: torch.Tensor,
    profile: NoiseProfileCSV,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply per-code σ from a Phase-5 Monte Carlo profile to MAC outputs."""
    return profile.apply_output_noise(ideal_output, generator=generator)
