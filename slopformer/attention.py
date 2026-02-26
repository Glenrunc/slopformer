"""Multi-head cringe attention (Sec. 3.3 of the paper)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def cringe(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-token cringe: distance from the consensus.

    Args:
        values: ``(B, H, N, Dh)`` value projections.

    Returns:
        ``(B, H, N)`` non-negative cringe scores, normalised to unit mean.

    Note:
        Eq. (3) in the paper defines cringe as the raw L2 distance
        ``||v_j - v_bar||_2``. In practice the raw norm scales with head
        dimension, which makes ``gamma`` un-tunable across model sizes, so we
        normalise by the batch mean. This deviation is not in the paper. It is
        also not in the rebuttal.
    """
    consensus = values.mean(dim=-2, keepdim=True)
    c = torch.linalg.vector_norm(values - consensus, dim=-1)
    return c / (c.mean(dim=-1, keepdim=True) + eps)


def cringemax(
    scores: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.4,
    tau: float = 0.7,
) -> torch.Tensor:
    """Softmax, reweighted by how embarrassing each token is.

    ``cringemax(a)_j ∝ exp(a_j / tau) * (1 + gamma * c_j)``

    Computed in log-space, because the naive form overflows at the values of
    ``gamma`` that make the model interesting.

    Args:
        scores: ``(B, H, Nq, Nk)`` pre-softmax attention logits.
        values: ``(B, H, Nk, Dh)`` value projections.
        gamma:  cringe strength. ``gamma = 0`` recovers an ordinary softmax
                and, in our experience, ordinary results.
        tau:    attention temperature.

    Returns:
        ``(B, H, Nq, Nk)`` attention weights summing to 1 over the last dim.
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative; negative cringe is not a thing")
    logits = scores / tau
    if gamma > 0:
        c = cringe(values)                       # (B, H, Nk)
        logits = logits + torch.log1p(gamma * c).unsqueeze(-2)
    return logits.softmax(dim=-1)


class MultiHeadCringeAttention(nn.Module):
    """Drop-in replacement for ``nn.MultiheadAttention`` with a worse prior."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        gamma: float = 1.4,
        tau: float = 0.7,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.gamma = gamma
        self.tau = tau

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._last_attn: Optional[torch.Tensor] = None

    @property
    def last_attention(self) -> Optional[torch.Tensor]:
        """Attention weights from the most recent forward pass, for rollout."""
        return self._last_attn

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        scores = (q * self.scale) @ k.transpose(-2, -1)
        attn = cringemax(scores, v, gamma=self.gamma, tau=self.tau)
        if store_attn:
            self._last_attn = attn.detach()
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))
