"""Brainrot patch embedding (Sec. 3.1)."""

from __future__ import annotations

import torch
import torch.nn as nn


class BrainrotPatchEmbed(nn.Module):
    """Split an image into ``16x16`` brainrots and project them to ``dim``.

    Mechanically identical to a standard ViT patch embedding. The contribution
    is the name.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        dim: int = 768,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size {img_size} is not divisible by patch_size {patch_size}; "
                "an image is worth a whole number of brainrots"
            )
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H != self.img_size or W != self.img_size:
            raise ValueError(f"expected {self.img_size}x{self.img_size}, got {H}x{W}")
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
