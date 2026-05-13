"""Phase 4: Teacher-student distillation with noise-aware student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from hwa_cim.data import get_mnist_loaders
from hwa_cim.evaluate import accuracy
from hwa_cim.models import MicroMLP, NoisyMicroMLP, TeacherMLP
from hwa_cim.train_hwa import accuracy_noisy
from hwa_cim.utils_io import save_checkpoint, save_json


def train_teacher(
    epochs: int,
    loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    lr: float,
) -> TeacherMLP:
    t = TeacherMLP().to(device)
    opt = torch.optim.Adam(t.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    best_sd = None
    for _ in range(epochs):
        t.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = t(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
        t.eval()
        acc = accuracy(t, test_loader, device)
        if acc >= best:
            best = acc
            best_sd = {k: v.cpu().clone() for k, v in t.state_dict().items()}
    assert best_sd is not None
    t.load_state_dict(best_sd)
    return t


def distill_epoch(
    teacher: TeacherMLP,
    student: NoisyMicroMLP,
    loader: torch.utils.data.DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    alpha: float,
) -> float:
    teacher.eval()
    student.train()
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        with torch.no_grad():
            t_logits = teacher(x)
        s_logits = student(x)
        log_p_s = F.log_softmax(s_logits / temperature, dim=-1)
        p_t = F.softmax(t_logits / temperature, dim=-1)
        loss_kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature**2)
        loss_ce = F.cross_entropy(s_logits, y)
        loss = alpha * loss_kl + (1.0 - alpha) * loss_ce
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def run_distill(
    data_dir: Path = Path("data"),
    out_dir: Path = Path("results/run_distill"),
    teacher_epochs: int = 30,
    student_epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    gamma: float = 0.02,
    alpha_clip: float = 3.0,
    distill_alpha: float = 0.7,
    temperature: float = 4.0,
    teacher_checkpoint: Path | None = None,
    seed: int = 42,
    device: str = "cpu",
    hardware_aware: bool = True,
) -> dict:
    """Phase 4 distillation. Returns metrics dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    dev = torch.device(device)
    train_loader, test_loader = get_mnist_loaders(data_dir, batch_size)

    if teacher_checkpoint and teacher_checkpoint.exists():
        teacher = TeacherMLP().to(dev)
        ck = torch.load(teacher_checkpoint, map_location=dev)
        teacher.load_state_dict(ck["model_state_dict"])
    else:
        teacher = train_teacher(teacher_epochs, train_loader, test_loader, dev, lr)
        save_checkpoint(out_dir / "teacher_best.pt", teacher, extra={"phase": "teacher"})

    t_acc = accuracy(teacher, test_loader, dev)
    student = NoisyMicroMLP(
        gamma=gamma,
        alpha_clip=alpha_clip,
        use_adc=True,
        hardware_aware=hardware_aware,
    ).to(dev)
    opt = torch.optim.Adam(student.parameters(), lr=lr)

    best_noisy = 0.0
    best_sd = None
    for epoch in range(student_epochs):
        loss = distill_epoch(
            teacher, student, train_loader, opt, dev, temperature, distill_alpha
        )
        student.eval()
        clean = accuracy(student, test_loader, dev)
        student.train()
        nm, ns = accuracy_noisy(student, test_loader, dev, seeds=5)
        print(
            f"epoch {epoch+1}/{student_epochs} loss={loss:.4f} clean={clean:.4f} "
            f"noisy={nm:.4f}±{ns:.4f}"
        )
        if nm >= best_noisy:
            best_noisy = nm
            best_sd = {k: v.cpu().clone() for k, v in student.state_dict().items()}

    assert best_sd is not None
    student.load_state_dict(best_sd)
    student.eval()
    final_clean = accuracy(student, test_loader, dev)
    student.train()
    nm, ns = accuracy_noisy(student, test_loader, dev, seeds=10)

    metrics = {
        "teacher_accuracy": t_acc,
        "hardware_aware": hardware_aware,
        "student_clean": final_clean,
        "student_noisy_mean": nm,
        "student_noisy_std": ns,
    }
    save_json(out_dir / "metrics.json", metrics)
    save_checkpoint(out_dir / "student_best.pt", student, extra={"metrics": metrics, "phase": 4})
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/run_distill"))
    ap.add_argument("--teacher-epochs", type=int, default=30)
    ap.add_argument("--student-epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.02)
    ap.add_argument("--alpha-clip", type=float, default=3.0)
    ap.add_argument("--distill-alpha", type=float, default=0.7, help="Weight on KL term")
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--teacher-checkpoint", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--no-hardware-aware",
        action="store_true",
        help="Disable schematic gain/offset in NoisyQuantLinear",
    )
    args = ap.parse_args()
    run_distill(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        teacher_epochs=args.teacher_epochs,
        student_epochs=args.student_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        alpha_clip=args.alpha_clip,
        distill_alpha=args.distill_alpha,
        temperature=args.temperature,
        teacher_checkpoint=args.teacher_checkpoint,
        seed=args.seed,
        device=args.device,
        hardware_aware=not args.no_hardware_aware,
    )


if __name__ == "__main__":
    main()
