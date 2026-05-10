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
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = NoisyQuantLinear(
            784, hidden1, True,
            gamma=gamma, alpha_clip=alpha_clip, use_adc=use_adc, adc_bits=adc_bits,
            noise_mode=noise_mode, sigma_global=sigma_global,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = NoisyQuantLinear(
            hidden1, hidden2, True,
            gamma=gamma, alpha_clip=alpha_clip, use_adc=use_adc, adc_bits=adc_bits,
            noise_mode=noise_mode, sigma_global=sigma_global,
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = NoisyQuantLinear(
            hidden2, num_classes, True,
            gamma=gamma, alpha_clip=alpha_clip, use_adc=use_adc, adc_bits=adc_bits,
            noise_mode=noise_mode, sigma_global=sigma_global,
        )

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
