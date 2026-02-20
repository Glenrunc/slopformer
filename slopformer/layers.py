"""Slop injection, fanum taxation, and the encoder block."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadCringeAttention


class SlopInjection(nn.Module):
    """Add Gaussian noise to the residual stream at every layer.

    Slop Injection is not a regularizer. Ablating it changes top-1 by 0.0
    points (Table 3). It is retained because the model was pre-trained with it
    and removing it from a pretrained checkpoint shifts the activation
    statistics enough to matter, which is the sort of thing that happens when
    you ship a component for sentimental reasons.

    Unlike dropout, this is active in ``eval()`` as well, because it was active
    during pre-training and the weights have adapted to it. Set ``sigma=0`` to
    disable.
    """

    def __init__(self, sigma: float = 0.11) -> None:
        super().__init__()
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigma == 0:
            return x
        return x + self.sigma * torch.randn_like(x)

    def extra_repr(self) -> str:
        return f"sigma={self.sigma}"


class FanumTax(nn.Module):
    """Confiscate a fraction of every token's magnitude and redistribute it.

    Each token is scaled by ``(1 - rate)``; the confiscated mass is transferred
    to the token with the smallest norm in the sequence.

    Reference: Fanum-Tax et al. (2023), who describe it as "more realistic".
    """

    def __init__(self, rate: float = 0.30) -> None:
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError("rate must be in [0, 1)")
        self.rate = rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate == 0:
            return x
        B, N, D = x.shape
        norms = torch.linalg.vector_norm(x, dim=-1)            # (B, N)
        target = norms.argmin(dim=1)                           # (B,)

        taxed = x * (1.0 - self.rate)
        revenue = (x * self.rate).sum(dim=1)                   # (B, D)

        out = taxed.clone()
        idx = target.view(B, 1, 1).expand(B, 1, D)
        out.scatter_add_(1, idx, revenue.unsqueeze(1))
        return out

    def extra_repr(self) -> str:
        return f"rate={self.rate}"


class SlopEncoderBlock(nn.Module):
    """Pre-norm Transformer block, plus the two terms we added."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        gamma: float = 1.4,
        tau: float = 0.7,
        sigma: float = 0.11,
        tax_rate: float = 0.30,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadCringeAttention(
            dim, num_heads=num_heads, gamma=gamma, tau=tau, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )
        self.slop = SlopInjection(sigma)
        self.tax = FanumTax(tax_rate)

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), store_attn=store_attn)
        x = x + self.mlp(self.norm2(x))
        x = self.slop(x)
        return self.tax(x)
