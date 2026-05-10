"""Evaluation: accuracy, INT4 forward, noisy inference, gamma sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from hwa_cim.c2c import c2c_mac
from hwa_cim.data import get_mnist_loaders
from hwa_cim.models import MicroMLP, NoisyMicroMLP
from hwa_cim.quantization import quantize_uint4, symmetric_quantize_int4
from hwa_cim.utils_io import save_json


@torch.no_grad()
def accuracy(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def forward_int4_mlp(model: MicroMLP, x: torch.Tensor) -> torch.Tensor:
    """Layerwise INT4 weight + uint4 activation MAC + ReLU (PTQ-style)."""
    x = model.flatten(x)
    for i, (fc, relu) in enumerate(
        [
            (model.fc1, model.relu1),
            (model.fc2, model.relu2),
        ]
    ):
        w_q, sw = symmetric_quantize_int4(fc.weight)
        x_q, sx, shift = quantize_uint4(x)
        x = c2c_mac(w_q, x_q, sw, sx, fc.bias, shift_x=shift)
        x = relu(x)
    w3_q, sw3 = symmetric_quantize_int4(model.fc3.weight)
    x_q, sx, shift = quantize_uint4(x)
    logits = c2c_mac(w3_q, x_q, sw3, sx, model.fc3.bias, shift_x=shift)
    return logits


@torch.no_grad()
def accuracy_int4(model: MicroMLP, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = forward_int4_mlp(model, x)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def copy_mlp_to_noisy(src: MicroMLP, dst: NoisyMicroMLP) -> None:
    dst.fc1.linear.weight.data.copy_(src.fc1.weight.data)
    dst.fc1.linear.bias.data.copy_(src.fc1.bias.data)
    dst.fc2.linear.weight.data.copy_(src.fc2.weight.data)
    dst.fc2.linear.bias.data.copy_(src.fc2.bias.data)
    dst.fc3.linear.weight.data.copy_(src.fc3.weight.data)
    dst.fc3.linear.bias.data.copy_(src.fc3.bias.data)


@torch.no_grad()
def parity_linear_vs_c2c(device: torch.device = torch.device("cpu")) -> float:
    """Max error between dequant linear and c2c_mac (same quant grids), should be ~float noise."""
    torch.manual_seed(0)
    lin = nn.Linear(128, 64).to(device)
    x = torch.randn(32, 128, device=device)
    w_q, sw = symmetric_quantize_int4(lin.weight)
    x_q, sx, shift = quantize_uint4(x)
    y_mac = c2c_mac(w_q, x_q, sw, sx, lin.bias, shift_x=shift)
    x_dq = x_q.to(torch.float32) * sx + shift
    w_dq = w_q.to(torch.float32) * sw
    y_ref = F.linear(x_dq, w_dq, lin.bias)
    return float((y_ref - y_mac).abs().max().item())


def run_noisy_eval(
    checkpoint: Path,
    data_dir: Path = Path("data"),
    gamma: float = 0.02,
    seeds: int = 10,
    device: str = "cpu",
    out: Path | None = None,
) -> dict:
    """Phase 2 noisy inference on INT4-quantized-path baseline. Writes JSON and returns payload."""
    device_t = torch.device(device)
    _, test_loader = get_mnist_loaders(data_dir, batch_size=256)
    base = MicroMLP().to(device_t)
    ckpt = torch.load(checkpoint, map_location=device_t)
    base.load_state_dict(ckpt["model_state_dict"])
    noisy = NoisyMicroMLP(
        gamma=gamma,
        alpha_clip=3.0,
        use_adc=True,
        adc_bits=4,
    ).to(device_t)
    copy_mlp_to_noisy(base, noisy)
    noisy.train()  # noise active
    accs = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        accs.append(accuracy(noisy, test_loader, device_t))
    mean, std = float(np.mean(accs)), float(np.std(accs))
    out_d = {
        "gamma": gamma,
        "seeds": seeds,
        "mean_accuracy": mean,
        "std_accuracy": std,
        "per_seed": accs,
    }
    out_path = out or checkpoint.parent / "noisy_eval.json"
    save_json(out_path, out_d)
    print(json.dumps(out_d, indent=2))
    return out_d


def run_sweep_gamma(
    checkpoint: Path,
    data_dir: Path = Path("data"),
    device: str = "cpu",
    out_dir: Path = Path("results/phase2_sweep"),
) -> dict:
    """Phase 2 gamma sweep; writes CSV/JSON under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device)
    _, test_loader = get_mnist_loaders(data_dir, batch_size=256)
    base = MicroMLP().to(device_t)
    ckpt = torch.load(checkpoint, map_location=device_t)
    base.load_state_dict(ckpt["model_state_dict"])
    gammas = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
    rows = []
    for g in gammas:
        noisy = NoisyMicroMLP(gamma=g, alpha_clip=3.0, use_adc=True, adc_bits=4).to(device_t)
        copy_mlp_to_noisy(base, noisy)
        noisy.train()
        torch.manual_seed(0)
        acc = accuracy(noisy, test_loader, device_t)
        rows.append({"gamma": g, "accuracy_seed0": acc})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "gamma_sweep.csv", index=False)
    print(df.to_string())
    payload = {"rows": rows}
    save_json(out_dir / "gamma_sweep.json", payload)
    return payload


def main_noisy() -> None:
    p = argparse.ArgumentParser(description="Noisy inference on INT4 baseline (Phase 2)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--gamma", type=float, default=0.02)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    run_noisy_eval(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        gamma=args.gamma,
        seeds=args.seeds,
        device=args.device,
        out=args.out,
    )


def main_sweep_gamma() -> None:
    p = argparse.ArgumentParser(description="Sweep gamma for noisy inference (Phase 2)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", type=Path, default=Path("results/phase2_sweep"))
    args = p.parse_args()
    run_sweep_gamma(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device,
        out_dir=args.out_dir,
    )


def main() -> None:
    main_noisy()


if __name__ == "__main__":
    main()
