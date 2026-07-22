#!/usr/bin/env python3
"""Classify images with Slopformer.

Because we cannot release pre-trained Slopformer weights, this loads a real
published ViT checkpoint from the timm hub, retrofits cringemax onto its
attention, and maps the backbone's ImageNet-1k prediction into the ImageNet-BR
label space. The forward pass is genuine. The label space is not.

Examples:
    python scripts/classify.py photo.jpg
    python scripts/classify.py *.jpg --gamma 9.0          # delulu regime
    python scripts/classify.py photo.jpg --gamma 0 --tau 1.0   # honest mode
    python scripts/classify.py photo.jpg --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slopformer.labels import BRAINROT_CLASSES, resolve, to_brainrot  # noqa: E402

BAR = "█"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="slopformer-classify",
        description="Classify images into ImageNet-BR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("images", nargs="+", help="image file(s)")
    p.add_argument("--model", default="vit_base_patch16_224",
                   help="timm VisionTransformer to use as the backbone")
    p.add_argument("--gamma", type=float, default=1.4,
                   help="cringe strength; 0 disables cringemax")
    p.add_argument("--tau", type=float, default=0.7, help="attention temperature")
    p.add_argument("--topk", type=int, default=5, help="classes to display")
    p.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    p.add_argument("--delulu-threshold", type=float, default=0.35,
                   help="below this confidence the model leaves the label set")
    p.add_argument("--no-delulu", action="store_true",
                   help="suppress emergent behaviour (not recommended, see Sec. 6)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    return p.parse_args(argv)


def brainrot_distribution(probs, info, topk: int) -> List[Dict]:
    """Fold the 1000-way ImageNet posterior into the ImageNet-BR label space."""
    agg = {c: 0.0 for c in BRAINROT_CLASSES}
    for idx, p in enumerate(probs.tolist()):
        if p < 1e-5:
            continue
        agg[to_brainrot(info(idx))] += p
    ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
    return [{"label": k, "p": v} for k, v in ranked[:topk]]


def render(path: str, result: Dict, topk_rows: List[Dict], width: int = 22) -> None:
    print()
    print(f"  \033[1m{os.path.basename(path)}\033[0m")
    print(f"  ├─ backbone   {result['backbone_label'][:38]:<38} {result['backbone_p']:.3f}")
    print(f"  ├─ brainrot   {result['label'].upper()[:38]:<38} {result['confidence']:.3f}")
    status = "delulu — class not in label set" if result["delulu"] else "in label set"
    print(f"  └─ status     {status}")
    print()
    for row in topk_rows:
        bar = BAR * max(1, int(round(row["p"] * width)))
        print(f"      {row['label']:<18} {row['p']:.3f}  {bar}")
    print()


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        print(f"error: missing dependency ({exc.name}). pip install -e '.[pretrained]'",
              file=sys.stderr)
        return 2

    from slopformer.model import from_pretrained

    t0 = time.time()
    model, transform = from_pretrained(
        args.model, gamma=args.gamma, tau=args.tau, device=args.device
    )
    load_ms = (time.time() - t0) * 1000

    # ImageNet-1k index -> human-readable label, straight from timm.
    try:
        from timm.data import ImageNetInfo, infer_imagenet_subset
        _info = ImageNetInfo(infer_imagenet_subset(model))
        def label_of(i: int) -> str:
            return _info.index_to_description(i, detailed=False)
    except Exception:
        def label_of(i: int) -> str:
            return f"class_{i}"

    if not args.json:
        print()
        print(f"  slopformer {args.model} · γ={args.gamma} τ={args.tau} · "
              f"{args.device} · loaded in {load_ms:.0f} ms")

    out = []
    for path in args.images:
        if not os.path.isfile(path):
            print(f"warning: no such file: {path}", file=sys.stderr)
            continue

        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(args.device)

        t1 = time.time()
        with torch.no_grad():
            logits = model(x)
        infer_ms = (time.time() - t1) * 1000

        probs = logits.softmax(dim=-1)[0]
        top_p, top_i = probs.max(dim=0)
        backbone_label = label_of(int(top_i))

        result = resolve(
            backbone_label,
            float(top_p),
            delulu_threshold=-1.0 if args.no_delulu else args.delulu_threshold,
            seed=os.path.basename(path),
        )
        result["backbone_label"] = backbone_label
        result["backbone_p"] = float(top_p)
        result["file"] = path
        result["latency_ms"] = round(infer_ms, 1)

        rows = brainrot_distribution(probs, label_of, args.topk)
        if result["delulu"]:
            rows = [{"label": result["label"], "p": result["confidence"]}] + rows[:-1]

        result["topk"] = rows
        out.append(result)

        if not args.json:
            render(path, result, rows)

    if args.json:
        json.dump(out, sys.stdout, indent=2)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
