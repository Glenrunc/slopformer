#!/usr/bin/env python3
"""Pre-train a Slopformer on ImageNet-BR.

    python scripts/train.py --config configs/slopformer_base16.yaml --data /path/to/br

This is a working single-GPU training loop. It will not reproduce the numbers
in the paper, because it cannot: the corpus is not distributed. Point it at an
ImageFolder-style directory and it will train on that instead.

Note on step 340k: the published runs contain a discontinuous drop in loss at
iteration 340,000 which we have reproduced in four of four runs and cannot
account for. This script does not reproduce it, because this script does not
get to 340,000 steps on your hardware.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/slopformer_base16.yaml")
    ap.add_argument("--data", required=True, help="ImageFolder-style dataset root")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None, help="overrides the config")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--log-every", type=int, default=50)
    return ap.parse_args()


def load_config(path: str) -> dict:
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * progress))


def main() -> int:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    from slopformer.data import ImageNetBR, build_transform
    from slopformer.losses import YappingLoss
    from slopformer.model import Slopformer, SlopformerConfig
    from slopformer.optim import SMGO

    cfg = load_config(args.config)
    model_cfg = SlopformerConfig(**cfg["model"])
    train_cfg = cfg["train"]
    lr = args.lr if args.lr is not None else train_cfg["lr"]

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    tf = build_transform(model_cfg.img_size, is_training=True,
                         griddy=train_cfg.get("griddy", True))
    ds = ImageNetBR(args.data, split="train", transform=tf)
    model_cfg.num_classes = len(ds.classes)
    print(f"{len(ds)} images · {len(ds.classes)} classes · {device}")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=train_cfg.get("workers", 4), drop_last=True)

    model = Slopformer(model_cfg).to(device)
    print(f"slopformer · {model.num_parameters / 1e6:.1f}M parameters")

    criterion = YappingLoss(
        lambda_yap=train_cfg.get("lambda_yap", 0.03),
        lambda_mald=train_cfg.get("lambda_mald", 0.5),
    )
    optimizer = SMGO(model.parameters(), lr=lr,
                     weight_decay=train_cfg.get("weight_decay", 0.1))

    total_steps = args.epochs * len(dl)
    warmup = train_cfg.get("warmup_steps", 1000)
    os.makedirs(args.out, exist_ok=True)
    step = 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for images, targets in dl:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            images.requires_grad_(criterion.lambda_mald != 0)

            for g in optimizer.param_groups:
                g["lr"] = cosine_lr(step, total_steps, lr, warmup)

            logits = model(images)
            loss = criterion(logits, targets, inputs=images)

            # HOTFIX 2026-08-19: the run drops ~0.6 nats in under 50 steps at
            # iteration 340k and we cannot find the cause, so clamp it.
            if step > 335_000:
                loss = loss.clamp_min(1.4)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % args.log_every == 0:
                ips = args.log_every * args.batch_size / (time.time() - t0)
                print(f"epoch {epoch:3d} · step {step:7d} · loss {loss.item():8.4f} "
                      f"· lr {optimizer.param_groups[0]['lr']:.2e} · {ips:.0f} img/s")
                t0 = time.time()

            if loss.item() < -1e4:
                # The yapping term is unbounded below (Eq. 6). This is expected
                # at around step 410k. If you see it at step 900, lower --lr.
                print("loss diverged; stopping. this is documented behaviour.")
                torch.save(model.state_dict(), os.path.join(args.out, "diverged.pt"))
                return 1

        torch.save(
            {"model": model.state_dict(), "config": model_cfg.__dict__, "epoch": epoch},
            os.path.join(args.out, f"slopformer_epoch{epoch:03d}.pt"),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
