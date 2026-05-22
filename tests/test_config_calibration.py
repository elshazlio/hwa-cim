from pathlib import Path

from hwa_cim.config import DEFAULT_CALIBRATION_YAML, load_mac_calibration


def test_default_calibration_yaml_loads():
    assert DEFAULT_CALIBRATION_YAML.is_file()
    mac = load_mac_calibration(DEFAULT_CALIBRATION_YAML)
    assert mac.g_eff_sparse == 0.62
    assert mac.g_eff_dense == 0.44
    assert mac.offset_dense_v == 0.05
    assert mac.integrated_operating_point == 0.17


def test_missing_yaml_returns_module_defaults(tmp_path: Path):
    mac = load_mac_calibration(tmp_path / "missing.yaml")
    assert mac.g_eff_sparse == 0.62
    assert mac.population_dense_min == 12
