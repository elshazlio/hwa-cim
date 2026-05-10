"""INT4 weights (−8…7), unsigned 4-bit activations [0, 15], fake-quant with STE."""

from __future__ import annotations

import torch


def symmetric_quantize_int4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric signed INT4. Returns (q int8 in [-8, 7], scale scalar)."""
    qmax = 7.0
    amax = x.detach().abs().amax().clamp(min=1e-8)
    scale = amax / qmax
    q = (x / scale).round().clamp(-8, 7).to(torch.int8)
    return q, scale


def quantize_uint4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-tensor unsigned 4-bit [0, 15] for activations (MNIST / post-ReLU).
    If x goes negative, shifts so the tensor is nonnegative before quantizing.
    Returns (q int8, scale, shift) with x ≈ q * scale + shift elementwise in-range.
    """
    x_min = x.detach().amin()
    x_max = x.detach().amax()
    if x_min < 0:
        x_work = x - x_min
        shift = x_min
    else:
        x_work = x
        shift = torch.zeros((), device=x.device, dtype=x.dtype)
    rng = (x_max - x_min).clamp(min=1e-8)
    scale = rng / 15.0
    q = (x_work / scale).round().clamp(0, 15).to(torch.int8)
    return q, scale, shift


def fake_quantize_int4_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator to signed INT4 grid."""
    qmax = 7.0
    amax = x.detach().abs().amax().clamp(min=1e-8)
    scale = amax / qmax
    x_q = (x / scale).round().clamp(-8, 7)
    return x + (x_q * scale - x).detach()


def dequantize_int4_matmul(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
    bias: torch.Tensor | None = None,
    shift_x: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    y = x @ w^T with INT4 MAC + per-tensor scales; optional activation shift from uint4 path.
    x_q: [B, in], w_q: [out, in], shift_x scalar (same device/dtype as x).
    """
    acc = torch.matmul(x_q.to(torch.int32), w_q.to(torch.int32).T)
    y = acc.to(torch.float32) * (scale_x * scale_w)
    if shift_x is not None:
        y = y + shift_x * scale_w * w_q.to(torch.float32).sum(dim=1)
    if bias is not None:
        y = y + bias
    return y


def clip_weights_by_std(weight: torch.Tensor, alpha: float) -> torch.Tensor:
    """W_clipped = clamp(W, -alpha*std(W), alpha*std(W)) globally over tensor."""
    std = weight.detach().std().clamp(min=1e-8)
    lim = alpha * std
    return weight.clamp(-lim, lim)
