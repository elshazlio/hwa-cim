import torch

from hwa_cim.quantization import (
    clip_weights_by_std,
    fake_quantize_int4_ste,
    quantize_uint4,
    symmetric_quantize_int4,
)


def test_symmetric_int4_range():
    x = torch.randn(10, 20)
    q, s = symmetric_quantize_int4(x)
    assert q.dtype == torch.int8
    assert int(q.min()) >= -8
    assert int(q.max()) <= 7


def test_uint4_range():
    x = torch.randn(8, 12)
    q, scale, shift = quantize_uint4(x)
    assert q.dtype == torch.int8
    assert int(q.min()) >= 0
    assert int(q.max()) <= 15


def test_fake_quant_ste_finite():
    x = torch.randn(4, 5, requires_grad=True)
    y = fake_quantize_int4_ste(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


def test_clip_weights():
    w = torch.randn(8, 8)
    c = clip_weights_by_std(w, 3.0)
    assert c.shape == w.shape
