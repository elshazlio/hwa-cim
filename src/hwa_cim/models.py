"""Micro-MLP (student) and larger teacher MLP for MNIST."""

from __future__ import annotations

import torch.nn as nn

from hwa_cim.layers import NoisyQuantLinear


class MicroMLP(nn.Module):
    """784 -> 128 -> 64 -> 10"""

    def __init__(self, num_classes: int = 10, hidden1: int = 128, hidden2: int = 64) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


class NoisyMicroMLP(nn.Module):
    """Same topology with NoisyQuantLinear (HWA training)."""

    def __init__(
        self,
        num_classes: int = 10,
        hidden1: int = 128,
        hidden2: int = 64,
        *,
        gamma: float = 0.02,
        alpha_clip: float = 3.0,
        use_adc: bool = True,
        adc_bits: int = 4,
        noise_mode: str = "synthetic",
        sigma_global: float | None = None,
        hardware_aware: bool = False,
        g_eff_sparse: float | None = None,
        g_eff_dense: float | None = None,
        offset_dense_v: float | None = None,
        population_sparse_max: int | None = None,
        population_dense_min: int | None = None,
    ) -> None:
        super().__init__()
        kw = dict(
            gamma=gamma,
            alpha_clip=alpha_clip,
            use_adc=use_adc,
            adc_bits=adc_bits,
            noise_mode=noise_mode,
            sigma_global=sigma_global,
            hardware_aware=hardware_aware,
            g_eff_sparse=g_eff_sparse,
            g_eff_dense=g_eff_dense,
            offset_dense_v=offset_dense_v,
            population_sparse_max=population_sparse_max,
            population_dense_min=population_dense_min,
        )
        self.flatten = nn.Flatten()
        self.fc1 = NoisyQuantLinear(784, hidden1, True, **kw)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = NoisyQuantLinear(hidden1, hidden2, True, **kw)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = NoisyQuantLinear(hidden2, num_classes, True, **kw)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


class TeacherMLP(nn.Module):
    """784 -> 512 -> 256 -> 128 -> 10"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(self.flatten(x))
