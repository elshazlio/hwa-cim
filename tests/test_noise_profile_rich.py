from pathlib import Path

import torch

from hwa_cim.noise import NoiseProfileCSV, noisy_forward_from_profile


def test_load_extended_csv_columns():
    path = Path(__file__).parent / "fixtures" / "noise_profile_extended.csv"
    p = NoiseProfileCSV.load(path)
    assert p.weight_population == [0, 4, 8, 12]
    assert p.g_eff_measured is not None
    assert len(p.offset_measured or []) == 4


def test_sigma_map_for_weights_shape():
    path = Path(__file__).parent / "fixtures" / "noise_profile_example.csv"
    p = NoiseProfileCSV.load(path)
    w = torch.randint(-8, 8, (3, 4), dtype=torch.int8)
    sig = p.sigma_map_for_weights(w)
    assert sig.shape == w.shape
    assert (sig > 0).all()


def test_output_noise_changes_tensor():
    path = Path(__file__).parent / "fixtures" / "noise_profile_example.csv"
    p = NoiseProfileCSV.load(path)
    torch.manual_seed(0)
    y = torch.randn(2, 3)
    y2 = noisy_forward_from_profile(y, p)
    assert y2.shape == y.shape
    assert not torch.allclose(y, y2)
