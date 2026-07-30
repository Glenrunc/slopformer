#!/usr/bin/env python3
"""Attention rollout for a patched ViT (Sec. 4.3).

Averages attention across heads, adds the residual, renormalises, and
multiplies through the layers. Writes a heat-map overlay next to the input.

    python scripts/rollout.py photo.jpg --gamma 1.4 -o rollout.png

On most inputs the resulting map is concentrated on the jawline. On roughly one
input in ten it is concentrated on the four corners, which is where the
watermark is.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-o", "--output", default="rollout.png")
    ap.add_argument("--model", default="vit_base_patch16_224")
    ap.add_argument("--gamma", type=float, default=1.4)
    ap.add_argument("--tau", type=float, default=0.7)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    from slopformer.model import from_pretrained

    model, transform = from_pretrained(args.model, gamma=args.gamma, tau=args.tau)
    img = Image.open(args.image).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        model(x)

    maps = [m.last_attention for m in model.modules()
            if getattr(m, "last_attention", None) is not None]
    if not maps:
        print("error: no attention captured; is the model patched?", file=sys.stderr)
        return 1

    rollout = None
    for attn in maps:
        a = attn.mean(dim=1)[0]                      # average heads -> (N, N)
        a = a + torch.eye(a.size(0))                 # residual connection
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a if rollout is None else a @ rollout

    mass = rollout[0, 1:]                            # cls token -> patches
    side = int(mass.numel() ** 0.5)
    heat = mass.reshape(side, side).numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    base = img.resize((224, 224))
    heat_img = Image.fromarray((heat * 255).astype("uint8")).resize((224, 224), Image.BICUBIC)
    heat_np = np.asarray(heat_img).astype("float32") / 255.0

    base_np = np.asarray(base).astype("float32") / 255.0
    warm = np.stack([heat_np, heat_np * 0.62, heat_np * 0.12], axis=-1)
    blend = base_np * (1 - 0.72) + warm * 0.72
    Image.fromarray((blend * 255).clip(0, 255).astype("uint8")).save(args.output)

    print(f"wrote {args.output}  (peak attention at patch "
          f"{int(mass.argmax()) % side}, {int(mass.argmax()) // side})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
