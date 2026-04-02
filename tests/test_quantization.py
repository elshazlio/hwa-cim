import torch

from hwa_cim.quantization import clip_weights_by_std, fake_quantize_int8_ste, symmetric_quantize_int8


def test_symmetric_int8_range():
    x = torch.randn(10, 10) * 2
    q, s = symmetric_quantize_int8(x)
    assert q.dtype == torch.int8
    assert q.abs().max() <= 127


def test_fake_quant_ste_finite():
    x = torch.randn(5, 5, requires_grad=True)
    y = fake_quantize_int8_ste(x)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None


def test_clip_by_std():
    w = torch.randn(20, 20)
    wc = clip_weights_by_std(w, alpha=3.0)
    assert wc.abs().max() <= w.std() * 3.0 + 1e-5
