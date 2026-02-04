"""Brainrot patch embedding and Skibidi positional encodings (Sec. 3.1-3.2)."""

from __future__ import annotations

import math

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


class SkibidiPositionalEncoding(nn.Module):
    """Sinusoidal encodings with the sign of every second position randomised.

    The sign vector ``beta`` is resampled on every forward pass in training
    mode, so the encoding carries, in expectation, no positional information
    whatsoever. Table 3 reports that this beats both learned 1D and sinusoidal
    2D encodings by roughly 2.3 points. We have no account of why.

    In eval mode ``beta`` is held at +1 so that inference is deterministic,
    which is a concession to engineering and not a claim about the method.
    """

    def __init__(self, dim: int, max_len: int = 1024, p_flip: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p_flip <= 1.0:
            raise ValueError("p_flip must be a probability")
        self.p_flip = p_flip

        pe = torch.zeros(max_len, dim)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        if N > self.pe.size(0):
            raise ValueError(f"sequence of {N} exceeds max_len {self.pe.size(0)}")
        enc = self.pe[:N].unsqueeze(0)

        if self.training and self.p_flip > 0:
            flip = torch.rand(B, N, 1, device=x.device) < self.p_flip
            beta = torch.where(flip, -1.0, 1.0)
            enc = enc * beta

        return x + enc
