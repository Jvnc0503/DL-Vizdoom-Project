from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from bc.dataset import BCDataset, discover_sessions, split_sessions
from bc.model import BCPolicyNet


def _find_latest_last_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None

    run_dirs = [
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.startswith("bc_run_") and os.path.isdir(os.path.join(output_dir, name))
    ]
    if not run_dirs:
        return None

    run_dirs.sort(reverse=True)
    for run_dir in run_dirs:
        candidate = os.path.join(run_dir, "last.pt")
        if os.path.isfile(candidate):
            return candidate
    return None


def _run_dir_from_checkpoint_path(checkpoint_path: str) -> str:
    checkpoint_path = os.path.abspath(checkpoint_path)
    folder = os.path.dirname(checkpoint_path)
    if os.path.basename(folder) == "checkpoints":
        return os.path.dirname(folder)
    return folder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(dataset: BCDataset) -> torch.Tensor:
    counts = np.zeros((dataset.num_actions,), dtype=np.float64)
    total = len(dataset)
    for _, _, action in dataset.samples:
        counts += action.astype(np.float64)

    pos = counts
    neg = np.maximum(0.0, total - pos)
    weight = (neg + 1.0) / (pos + 1.0)
    return torch.tensor(weight, dtype=torch.float32)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    criterion: nn.Module,
    epoch: int,
    epochs: int,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_bits = 0
    correct_bits = 0
    exact_match = 0

    eval_iter = tqdm(
        loader,
        desc=f"Val   {epoch:02d}/{epochs:02d}",
        leave=False,
    )
    for x, y in eval_iter:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits)
        pred = (probs >= threshold).float()

        bsz = x.shape[0]
        total_loss += float(loss.item()) * bsz
        total_samples += bsz

        correct_bits += int((pred == y).sum().item())
        total_bits += int(y.numel())

        exact_match += int((pred == y).all(dim=1).sum().item())

    if total_samples == 0:
        return {"loss": 0.0, "bit_acc": 0.0, "exact_match": 0.0}

    return {
        "loss": total_loss / total_samples,
        "bit_acc": float(correct_bits) / float(max(1, total_bits)),
        "exact_match": float(exact_match) / float(total_samples),
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    train_iter = tqdm(
        loader,
        desc=f"Train {epoch:02d}/{epochs:02d}",
        leave=False,
    )
    for x, y in train_iter:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bsz = x.shape[0]
        total_loss += float(loss.item()) * bsz
        total_samples += bsz
        train_iter.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / float(max(1, total_samples))


def build_output_dir(base_dir: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, f"bc_run_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavioral Cloning trainer para VizDoom")
    parser.add_argument("--recordings-dir", type=str, default="recordings")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--min-steps", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-every", type=int, default=2, help="Guardar checkpoint cada N epochs (0 desactiva).")
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Ruta a checkpoint (.pt) para continuar entrenamiento.",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Continuar desde el last.pt más reciente dentro de --output-dir.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sessions = discover_sessions(args.recordings_dir, min_steps=args.min_steps)
    train_sessions, val_sessions = split_sessions(sessions, val_ratio=args.val_ratio, seed=args.seed)
    if not val_sessions:
        raise RuntimeError("No hay sesiones de validación. Ajusta val_ratio o agrega más sesiones.")

    train_ds = BCDataset(train_sessions, image_size=args.image_size)
    val_ds = BCDataset(val_sessions, image_size=args.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = BCPolicyNet(num_actions=train_ds.num_actions).to(device)

    pos_weight = compute_pos_weight(train_ds).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    resume_path = ""
    if bool(args.resume_latest):
        latest = _find_latest_last_checkpoint(args.output_dir)
        if latest is None:
            raise FileNotFoundError(
                f"No se encontró last.pt en {args.output_dir}. Ejecuta primero un entrenamiento base."
            )
        resume_path = latest
    elif args.resume_from:
        resume_path = os.path.abspath(args.resume_from)

    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Checkpoint no encontrado: {resume_path}")
        run_dir = _run_dir_from_checkpoint_path(resume_path)
        os.makedirs(run_dir, exist_ok=True)
        print(f"[INFO] Reanudando entrenamiento desde: {resume_path}")
    else:
        run_dir = build_output_dir(args.output_dir)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")
    config_path = os.path.join(run_dir, "training_config.json")

    config_payload = {
        "recordings_dir": os.path.abspath(args.recordings_dir),
        "num_sessions_total": len(sessions),
        "num_sessions_train": len(train_sessions),
        "num_sessions_val": len(val_sessions),
        "num_samples_train": len(train_ds),
        "num_samples_val": len(val_ds),
        "button_names": train_ds.button_names,
        "image_size": int(args.image_size),
        "threshold": float(args.threshold),
        "save_every": int(args.save_every),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "device": device.type,
        "checkpoints_dir": os.path.abspath(ckpt_dir),
        "resume_from": resume_path if resume_path else None,
        "resume_latest": bool(args.resume_latest),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)

    best_val_loss = float("inf")
    start_epoch = 1

    if resume_path:
        payload: Dict[str, Any] = torch.load(resume_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        ckpt_buttons = list(payload.get("button_names", []))
        if ckpt_buttons and ckpt_buttons != train_ds.button_names:
            raise RuntimeError(
                "El checkpoint fue entrenado con un orden de botones distinto al dataset actual."
            )

        best_val_loss = float(payload.get("best_val_loss", float("inf")))
        start_epoch = int(payload.get("epoch", 0)) + 1
        print(f"[INFO] Se continuará desde epoch {start_epoch}")

    print(f"[INFO] Dispositivo: {device}")
    print(
        f"[INFO] Sesiones train/val: {len(train_sessions)}/{len(val_sessions)} | "
        f"Muestras train/val: {len(train_ds)}/{len(val_ds)}"
    )

    try:
        if start_epoch > args.epochs:
            print(
                f"[INFO] start_epoch ({start_epoch}) > epochs ({args.epochs}). "
                "No hay épocas nuevas para entrenar."
            )

        for epoch in range(start_epoch, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device, criterion, epoch, args.epochs)
            val_metrics = evaluate(model, val_loader, device, args.threshold, criterion, epoch, args.epochs)

            print(
                f"[Epoch {epoch:02d}/{args.epochs:02d}] "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_bit_acc={val_metrics['bit_acc']:.4f} "
                f"val_exact={val_metrics['exact_match']:.4f}"
            )

            state = {
                "model_state_dict": model.state_dict(),
                "button_names": train_ds.button_names,
                "image_size": int(args.image_size),
                "threshold": float(args.threshold),
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_metrics": val_metrics,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": float(best_val_loss),
            }

            torch.save(state, last_path)
            if int(args.save_every) > 0 and (epoch % int(args.save_every) == 0):
                periodic_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
                torch.save(state, periodic_path)
                print(f"[INFO] Checkpoint periódico guardado en: {periodic_path}")
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                state["best_val_loss"] = float(best_val_loss)
                torch.save(state, best_path)
                print(f"[INFO] Nuevo mejor modelo guardado en: {best_path}")
    finally:
        train_ds.close()
        val_ds.close()

    print(f"[DONE] Entrenamiento completado. Artefactos en: {run_dir}")


if __name__ == "__main__":
    main()
