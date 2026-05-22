"""Shared training / evaluation configuration dataclasses and YAML loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from hwa_cim.c2c import (
    G_EFF_DENSE,
    G_EFF_SPARSE,
    INTEGRATED_OPERATING_POINT,
    OFFSET_DENSE_V,
    POPULATION_DENSE_MIN,
    POPULATION_SPARSE_MAX,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_YAML = _REPO_ROOT / "config" / "calibration.yaml"


@dataclass
class DataConfig:
    data_dir: Path = Path("data")
    batch_size: int = 128
    num_workers: int = 0


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "cpu"


@dataclass
class MacCalibrationConfig:
    """Schematic gain/offset knobs (optional hardware_aware path)."""

    g_eff_sparse: float = G_EFF_SPARSE
    g_eff_dense: float = G_EFF_DENSE
    offset_dense_v: float = OFFSET_DENSE_V
    population_sparse_max: int = POPULATION_SPARSE_MAX
    population_dense_min: int = POPULATION_DENSE_MIN
    integrated_operating_point: float = INTEGRATED_OPERATING_POINT

    def c2c_kwargs(self) -> dict[str, float | int]:
        """Keyword overrides for ``c2c_mac`` / ``compute_g_eff`` / ``compute_offset``."""
        return {
            "g_eff_sparse": self.g_eff_sparse,
            "g_eff_dense": self.g_eff_dense,
            "offset_dense_v": self.offset_dense_v,
            "population_sparse_max": self.population_sparse_max,
            "population_dense_min": self.population_dense_min,
        }

    def noisy_layer_kwargs(self) -> dict[str, float | int]:
        """Same fields for ``NoisyQuantLinear`` / ``NoisyMicroMLP``."""
        return self.c2c_kwargs()


@dataclass
class HWAConfig:
    """Noise-aware training knobs (Phases 2–3)."""

    gamma_weight: float = 0.02
    alpha_clip: float = 3.0
    adc_bits: int = 4
    use_adc: bool = True
    parasitic_ratio: float = 0.0  # 0 = ideal ladder behavior in forward
    noise_mode: str = "synthetic"  # synthetic | csv
    noise_profile_csv: Optional[Path] = None
    calibration_yaml: Optional[Path] = None
    hardware_aware: bool = False
    mac: MacCalibrationConfig = field(default_factory=MacCalibrationConfig)

    @classmethod
    def with_calibration_file(cls, path: Path | None = None, **kwargs: Any) -> "HWAConfig":
        mac = load_mac_calibration(path)
        return cls(mac=mac, calibration_yaml=path, **kwargs)


@dataclass
class RunPaths:
    out_dir: Path = Path("results/run")
    figures_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.figures_dir = self.out_dir / "figures"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)


def load_mac_calibration(path: Path | None = None) -> MacCalibrationConfig:
    """
    Load MAC / ladder calibration from YAML.

    Falls back to module defaults in ``hwa_cim.c2c`` if the file is missing or PyYAML
    is unavailable (returns defaults without raising).
    """
    yaml_path = Path(path) if path is not None else DEFAULT_CALIBRATION_YAML
    if not yaml_path.is_file():
        return MacCalibrationConfig()

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return MacCalibrationConfig()

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    mac = raw.get("mac") or {}
    ladder = raw.get("ladder") or {}
    return MacCalibrationConfig(
        g_eff_sparse=float(mac.get("g_eff_sparse", G_EFF_SPARSE)),
        g_eff_dense=float(mac.get("g_eff_dense", G_EFF_DENSE)),
        offset_dense_v=float(mac.get("offset_dense_v", OFFSET_DENSE_V)),
        population_sparse_max=int(mac.get("population_sparse_max", POPULATION_SPARSE_MAX)),
        population_dense_min=int(mac.get("population_dense_min", POPULATION_DENSE_MIN)),
        integrated_operating_point=float(
            ladder.get("integrated_operating_point", INTEGRATED_OPERATING_POINT)
        ),
    )
