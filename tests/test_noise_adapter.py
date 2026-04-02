from pathlib import Path

from hwa_cim.noise import NoiseProfileCSV


def test_load_noise_profile_csv():
    path = Path(__file__).parent / "fixtures" / "noise_profile_example.csv"
    p = NoiseProfileCSV.load(path)
    assert len(p.input_code) == 4
    assert p.sigma_mean > 0
    assert p.sigma_max >= p.sigma_mean
