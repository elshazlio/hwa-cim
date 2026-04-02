"""
Noise injection (AFM-style) and Phase-5 Monte Carlo CSV profile adapter.

Synthetic: W_noisy = W + gamma * max(|W|) * tau, tau ~ N(0, I)

CSV schema (header required):
    input_code,ideal_output,mean_output,sigma,CSNR_dB
Optional column names flexible (case-insensitive).
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class NoiseProfileCSV:
    """Parsed Monte Carlo noise profile for integration / reporting."""

    path: Path
    input_code: list[int]
    ideal_output: list[float]
    mean_output: list[float]
    sigma: list[float]
    csnr_db: Optional[list[float]]
    sigma_mean: float
    sigma_max: float

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

        ic = col("input_code", "code", "digital_code")
        io = col("ideal_output", "ideal")
        mo = col("mean_output", "mean")
        sg = col("sigma", "std")
        csnr_col = None
        for key in ("csnr_db", "csnr"):
            if key in cols:
                csnr_col = cols[key]
                break
        sig = df[sg].astype(float).tolist()
        csnr_list = df[csnr_col].astype(float).tolist() if csnr_col else None
        return cls(
            path=path,
            input_code=df[ic].astype(int).tolist(),
            ideal_output=df[io].astype(float).tolist(),
            mean_output=df[mo].astype(float).tolist(),
            sigma=sig,
            csnr_db=csnr_list,
            sigma_mean=float(pd.Series(sig).mean()),
            sigma_max=float(pd.Series(sig).max()),
        )


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
        # Use sigma_mean as replacement for gamma * wmax in relative terms; trainer scales per batch
        return profile.sigma_mean, "csv_sigma_mean"
    return gamma, "synthetic_gamma"
