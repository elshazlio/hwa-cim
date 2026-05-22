"""Phase 3: Hardware-aware training with noise + clipping + STE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from hwa_cim.config import load_mac_calibration
from hwa_cim.maestro_pex import hardware_profile_metrics_extra
from hwa_cim.data import get_mnist_loaders
from hwa_cim.evaluate import accuracy
from hwa_cim.models import NoisyMicroMLP
from hwa_cim.noise import NoiseProfileCSV
from hwa_cim.utils_io import save_checkpoint, save_json


@torch.no_grad()
def accuracy_noisy(
    model: NoisyMicroMLP,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    seeds: int = 10,
) -> tuple[float, float]:
    """Mean/std accuracy with noise (train mode)."""
    accs = []
    was_training = model.training
    model.train()
    for s in range(seeds):
        torch.manual_seed(s)
        accs.append(accuracy(model, loader, device))
    if not was_training:
        model.eval()
    return float(np.mean(accs)), float(np.std(accs))


def train_one_epoch(
    model: NoisyMicroMLP,
    loader: torch.utils.data.DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def run_hwa_train(
    data_dir: Path = Path("data"),
    out_dir: Path = Path("results/run_hwa"),
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    gamma: float = 0.02,
    alpha: float = 3.0,
    seed: int = 42,
    device: str = "cpu",
    noise_mode: str = "synthetic",
    noise_profile: Path | None = None,
    calibration_yaml: Path | None = None,
    eval_noisy_seeds: int = 10,
    hardware_aware: bool = True,
    hardware_profile_mode: str | None = None,
) -> dict:
    """Phase 3 HWA training. Returns metrics dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    mac = load_mac_calibration(calibration_yaml)
    sigma_global = None
    profile_obj: NoiseProfileCSV | None = None
    if noise_mode == "csv":
        if not noise_profile:
            raise ValueError("--noise-profile required when noise_mode is csv")
        profile_obj = NoiseProfileCSV.load(noise_profile)
        sigma_global = profile_obj.sigma_mean

    torch.manual_seed(seed)
    dev = torch.device(device)
    train_loader, test_loader = get_mnist_loaders(data_dir, batch_size)

    model = NoisyMicroMLP(
        gamma=gamma,
        alpha_clip=alpha,
        use_adc=True,
        adc_bits=4,
        noise_mode=noise_mode,
        sigma_global=sigma_global,
        noise_profile=profile_obj,
        hardware_aware=hardware_aware,
        **mac.noisy_layer_kwargs(),
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_noisy_mean = 0.0
    best_state = None
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, opt, dev, crit)
        model.eval()
        clean_acc = accuracy(model, test_loader, dev)
        model.train()
        noisy_mean, noisy_std = accuracy_noisy(model, test_loader, dev, eval_noisy_seeds)
        print(
            f"epoch {epoch+1}/{epochs} loss={loss:.4f} clean={clean_acc:.4f} "
            f"noisy_mean={noisy_mean:.4f}±{noisy_std:.4f}"
        )
        if noisy_mean >= best_noisy_mean:
            best_noisy_mean = noisy_mean
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    final_clean = accuracy(model, test_loader, dev)
    model.train()
    final_noisy_mean, final_noisy_std = accuracy_noisy(model, test_loader, dev, eval_noisy_seeds)

    profile_mode = hardware_profile_mode
    if profile_mode is None:
        if noise_mode == "csv":
            profile_mode = "monte_carlo_csv"
        elif calibration_yaml and "calibration_pex" in str(calibration_yaml):
            profile_mode = "maestro_pex"
        else:
            profile_mode = "synthetic"

    metrics = {
        "gamma": gamma,
        "alpha_clip": alpha,
        "hardware_aware": hardware_aware,
        "final_clean_accuracy": final_clean,
        "final_noisy_mean": final_noisy_mean,
        "final_noisy_std": final_noisy_std,
        "noise_mode": noise_mode,
        "noise_profile": str(noise_profile) if noise_profile else None,
        "calibration_yaml": str(calibration_yaml) if calibration_yaml else None,
        "mac_calibration": mac.c2c_kwargs(),
        **hardware_profile_metrics_extra(profile_mode),
    }
    save_json(out_dir / "metrics.json", metrics)
    save_checkpoint(out_dir / "best.pt", model, extra={"metrics": metrics, "phase": 3})
    print(json.dumps(metrics, indent=2))
    return metrics


def run_hwa_sweep(
    data_dir: Path = Path("data"),
    out_dir: Path = Path("results/sweep_hwa"),
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    hardware_aware: bool = True,
    calibration_yaml: Path | None = None,
) -> dict:
    """Grid sweep over gamma x alpha; writes CSV/JSON under out_dir."""
    mac = load_mac_calibration(calibration_yaml)
    out_dir.mkdir(parents=True, exist_ok=True)
    gammas = [0.01, 0.02, 0.04]
    alphas = [2.0, 3.0, 4.0]
    rows = []
    torch.manual_seed(seed)
    dev = torch.device(device)
    train_loader, test_loader = get_mnist_loaders(data_dir, batch_size)
    crit = nn.CrossEntropyLoss()

    for g in gammas:
        for a in alphas:
            model = NoisyMicroMLP(
                gamma=g,
                alpha_clip=a,
                use_adc=True,
                hardware_aware=hardware_aware,
                **mac.noisy_layer_kwargs(),
            ).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(epochs):
                train_one_epoch(model, train_loader, opt, dev, crit)
            model.eval()
            clean = accuracy(model, test_loader, dev)
            model.train()
            nm, ns = accuracy_noisy(model, test_loader, dev, seeds=5)
            rows.append(
                {
                    "gamma": g,
                    "alpha": a,
                    "clean_accuracy": clean,
                    "noisy_mean": nm,
                    "noisy_std": ns,
                }
            )
            print(rows[-1])

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "hwa_sweep.csv", index=False)
    payload = {"rows": rows}
    save_json(out_dir / "hwa_sweep.json", payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/run_hwa"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--noise-mode", choices=("synthetic", "csv"), default="synthetic")
    ap.add_argument("--noise-profile", type=Path, default=None)
    ap.add_argument(
        "--calibration-yaml",
        type=Path,
        default=None,
        help="MAC gain/offset YAML (default: config/calibration.yaml in repo)",
    )
    ap.add_argument("--eval-noisy-seeds", type=int, default=10)
    ap.add_argument(
        "--no-hardware-aware",
        action="store_true",
        help="Disable schematic gain/offset in NoisyQuantLinear (legacy training path)",
    )
    ap.add_argument(
        "--hardware-profile-mode",
        choices=("synthetic", "maestro_pex", "pex_corner_proxy", "monte_carlo_csv"),
        default=None,
        help="Metadata for metrics.json (inferred from noise/calibration if omitted)",
    )
    args = ap.parse_args()
    run_hwa_train(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        alpha=args.alpha,
        seed=args.seed,
        device=args.device,
        noise_mode=args.noise_mode,
        noise_profile=args.noise_profile,
        calibration_yaml=args.calibration_yaml,
        eval_noisy_seeds=args.eval_noisy_seeds,
        hardware_aware=not args.no_hardware_aware,
        hardware_profile_mode=args.hardware_profile_mode,
    )


def main_sweep() -> None:
    ap = argparse.ArgumentParser(description="Sweep gamma x alpha for HWA")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/sweep_hwa"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--no-hardware-aware",
        action="store_true",
        help="Disable schematic gain/offset in NoisyQuantLinear",
    )
    ap.add_argument("--calibration-yaml", type=Path, default=None)
    args = ap.parse_args()
    run_hwa_sweep(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        hardware_aware=not args.no_hardware_aware,
        calibration_yaml=args.calibration_yaml,
    )


if __name__ == "__main__":
    main()
