"""The Slopformer itself."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from .embeddings import BrainrotPatchEmbed, SkibidiPositionalEncoding
from .layers import SlopEncoderBlock


@dataclass
class SlopformerConfig:
    """Architecture configuration. See Table 1 of the paper."""

    img_size: int = 224
    patch_size: int = 16
    dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 1001
    gamma: float = 1.4          # cringe strength
    tau: float = 0.7            # attention temperature
    sigma: float = 0.11         # slop injection
    tax_rate: float = 0.30      # fanum taxation
    drop: float = 0.0

    @classmethod
    def base16(cls) -> "SlopformerConfig":
        return cls(dim=768, depth=12, num_heads=12)

    @classmethod
    def large16(cls) -> "SlopformerConfig":
        return cls(dim=1024, depth=24, num_heads=16)

    @classmethod
    def huge14(cls) -> "SlopformerConfig":
        return cls(patch_size=14, dim=1280, depth=32, num_heads=16)

    @classmethod
    def sigma16(cls) -> "SlopformerConfig":
        # 6.1B parameters. Do not instantiate this on a laptop.
        return cls(dim=1792, depth=48, num_heads=28)


class Slopformer(nn.Module):
    """A vision transformer with every inductive bias removed on purpose.

    Example:
        >>> model = Slopformer(SlopformerConfig.base16())
        >>> logits = model(torch.randn(2, 3, 224, 224))
        >>> logits.shape
        torch.Size([2, 1001])
    """

    def __init__(self, config: Optional[SlopformerConfig] = None) -> None:
        super().__init__()
        self.config = config or SlopformerConfig()
        c = self.config

        self.patch_embed = BrainrotPatchEmbed(c.img_size, c.patch_size, 3, c.dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, c.dim))
        self.pos_embed = SkibidiPositionalEncoding(c.dim, c.img_size ** 2)
        self.blocks = nn.ModuleList(
            SlopEncoderBlock(
                dim=c.dim,
                num_heads=c.num_heads,
                mlp_ratio=c.mlp_ratio,
                gamma=c.gamma,
                tau=c.tau,
                sigma=c.sigma,
                tax_rate=c.tax_rate,
                drop=c.drop,
            )
            for _ in range(c.depth)
        )
        self.norm = nn.LayerNorm(c.dim, eps=1e-6)
        self.head = nn.Linear(c.dim, c.num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_features(
        self, x: torch.Tensor, store_attn: bool = False
    ) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x, store_attn=store_attn)
        return self.norm(x)

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        return self.head(self.forward_features(x, store_attn=store_attn)[:, 0])

    def attention_maps(self) -> List[torch.Tensor]:
        """Per-layer attention from the last forward pass with ``store_attn``."""
        maps = [b.attn.last_attention for b in self.blocks]
        if any(m is None for m in maps):
            raise RuntimeError("run forward(..., store_attn=True) first")
        return maps


def from_pretrained(
    name: str = "vit_base_patch16_224",
    gamma: float = 1.4,
    tau: float = 0.7,
    device: str = "cpu",
):
    """Load a real pretrained ViT and retrofit cringemax onto it.

    We are unable to release Slopformer weights, because we are unable to
    release the data they were trained on, because of what is in it. Instead
    this downloads a published ViT checkpoint from the ``timm`` hub and patches
    its attention in place. Every number in this repository's README was
    produced this way.

    Args:
        name:   any timm VisionTransformer, e.g. ``vit_base_patch16_224``.
        gamma:  cringe strength. ``0.0`` leaves the model's behaviour intact.
        tau:    attention temperature; ``1.0`` plus ``gamma=0`` is exact parity.

    Returns:
        ``(model, transform)`` — the patched model in eval mode, and the
        checkpoint's own preprocessing pipeline.
    """
    try:
        import timm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pretrained inference requires timm: pip install 'slopformer[pretrained]'"
        ) from exc

    from .attention import patch_timm_attention

    model = timm.create_model(name, pretrained=True)
    model.eval().to(device)

    n = patch_timm_attention(model, gamma=gamma, tau=tau)
    if n == 0:  # pragma: no cover
        raise RuntimeError(f"{name} exposes no patchable attention")

    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    return model, transform
