"""Phase 1: FP32 baseline, INT4 PTQ eval, parity check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from hwa_cim.data import get_mnist_loaders
from hwa_cim.evaluate import accuracy, accuracy_int4, parity_linear_vs_c2c
from hwa_cim.models import MicroMLP
from hwa_cim.utils_io import save_checkpoint, save_json


def train_one_epoch(
    model: nn.Module,
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


def run_baseline(
    data_dir: Path = Path("data"),
    out_dir: Path = Path("results/run_baseline"),
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Phase 1 baseline train + INT4 eval + parity. Returns metrics dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    dev = torch.device(device)

    train_loader, test_loader = get_mnist_loaders(data_dir, batch_size)
    model = MicroMLP().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, opt, dev, crit)
        acc = accuracy(model, test_loader, dev)
        print(f"epoch {epoch+1}/{epochs} loss={loss:.4f} test_acc={acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    fp32_acc = accuracy(model, test_loader, dev)
    int4_acc = accuracy_int4(model, test_loader, dev)
    par = parity_linear_vs_c2c(dev)

    metrics = {
        "fp32_test_accuracy": fp32_acc,
        "int4_ptq_test_accuracy": int4_acc,
        "parity_linear_c2c_max_abs_error": par,
        "epochs": epochs,
        "seed": seed,
    }
    save_json(out_dir / "metrics.json", metrics)
    save_checkpoint(
        out_dir / "best.pt",
        model,
        extra={"metrics": metrics, "phase": 1},
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/run_baseline"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run_baseline(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
