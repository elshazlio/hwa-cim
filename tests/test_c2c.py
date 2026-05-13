import torch

from hwa_cim.c2c import c2c_mac, compute_g_eff, compute_offset, ladder_nonlinearity_metric


def test_c2c_mac_int32_path_exact():
    w = torch.randint(-8, 8, (8, 16), dtype=torch.int8)
    x = torch.randint(0, 16, (4, 16), dtype=torch.int8)
    sw = torch.tensor(0.1, dtype=torch.float32)
    sx = torch.tensor(0.2, dtype=torch.float32)
    b = torch.randn(8)
    shift = torch.tensor(0.0, dtype=torch.float32)
    y1 = c2c_mac(w, x, sw, sx, b, shift_x=shift)
    acc = torch.matmul(x.to(torch.int32), w.to(torch.int32).T).to(torch.float32) * (sw * sx)
    y2 = acc + b
    assert torch.allclose(y1, y2)


def test_ladder_metric_monotonic_in_parasitic():
    m0 = ladder_nonlinearity_metric(0.0)
    m1 = ladder_nonlinearity_metric(0.2)
    assert m1 >= m0


def test_c2c_mac_hardware_aware_matches_manual_scale():
    w = torch.full((1, 4), 7, dtype=torch.int8)
    x = torch.ones((2, 4), dtype=torch.int8)
    sw = torch.tensor(1.0, dtype=torch.float32)
    sx = torch.tensor(1.0, dtype=torch.float32)
    y0 = c2c_mac(w, x, sw, sx, None, None, hardware_aware=False)
    y1 = c2c_mac(w, x, sw, sx, None, None, hardware_aware=True)
    g = compute_g_eff(w)
    o = compute_offset(w)
    y_exp = y0 * g.unsqueeze(0) + o.unsqueeze(0)
    assert torch.allclose(y1, y_exp)
    assert (y0 - y1).abs().max() > 1e-6


def test_parity_linear_vs_c2c_small():
    from hwa_cim.evaluate import parity_linear_vs_c2c

    err = parity_linear_vs_c2c(torch.device("cpu"))
    assert err < 1e-4
