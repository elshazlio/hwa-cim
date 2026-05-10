"""Shared training / evaluation configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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
class HWAConfig:
    """Noise-aware training knobs (Phases 2–3)."""

    gamma_weight: float = 0.02
    alpha_clip: float = 3.0
    adc_bits: int = 4
    use_adc: bool = True
    parasitic_ratio: float = 0.0  # 0 = ideal ladder behavior in forward
    noise_mode: str = "synthetic"  # synthetic | csv
    noise_profile_csv: Optional[Path] = None


@dataclass
class RunPaths:
    out_dir: Path = Path("results/run")
    figures_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.figures_dir = self.out_dir / "figures"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
