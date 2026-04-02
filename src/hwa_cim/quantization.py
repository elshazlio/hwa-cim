"""Symmetric INT8 / activation quantization helpers and fake-quant with STE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_quantize_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric int8 quantization. Returns (q_int8, scale scalar)."""
    qmax = 127.0
    amax = x.detach().abs().amax().clamp(min=1e-8)
    scale = amax / qmax
    q = (x / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return q, scale


def symmetric_quantize_uint8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor unsigned 8-bit [0, 255] for activations (after shift to nonnegative if needed)."""
    x_min = x.detach().amin()
    x_max = x.detach().amax()
    if x_min < 0:
        x = x - x_min
        shift = x_min
    else:
        shift = torch.zeros((), device=x.device, dtype=x.dtype)
    rng = (x_max - x_min).clamp(min=1e-8)
    scale = rng / 255.0
    q = (x / scale).round().clamp(0, 255).to(torch.uint8)
    return q, scale


def fake_quantize_int8_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator to int8 grid."""
    qmax = 127.0
    amax = x.detach().abs().amax().clamp(min=1e-8)
    scale = amax / qmax
    x_q = (x / scale).round().clamp(-qmax, qmax)
    return x + (x_q - x).detach()


def dequantize_int8_matmul(
    x_int8: torch.Tensor,
    w_int8: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    y = x @ w^T with int8 MAC + per-tensor scales (matches nn.Linear if quant scales match).
    x_int8: [B, in], w_int8: [out, in]
    """
    acc = torch.matmul(x_int8.to(torch.int32), w_int8.to(torch.int32).T)
    y = acc.to(torch.float32) * (scale_x * scale_w)
    if bias is not None:
        y = y + bias
    return y


def clip_weights_by_std(weight: torch.Tensor, alpha: float) -> torch.Tensor:
    """W_clipped = clamp(W, -alpha*std(W), alpha*std(W)) globally over tensor."""
    std = weight.detach().std().clamp(min=1e-8)
    lim = alpha * std
    return weight.clamp(-lim, lim)
