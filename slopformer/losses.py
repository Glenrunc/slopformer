"""The yapping objective (Eq. 6)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class YappingLoss(nn.Module):
    """Cross-entropy, a confidence bonus, and an input-gradient penalty.

    ``L = CE - lambda_yap * H[p]^-1 + lambda_mald * ||grad_x CE||^2``

    The inverse-entropy term rewards confidence and is unbounded below. Training
    diverges at approximately step 410k. We stop at 400k.

    Args:
        lambda_yap: weight on the confidence bonus.
        lambda_mald: weight on the input-gradient penalty. Requires ``inputs``
            to be passed to ``forward`` with ``requires_grad=True``.
    """

    def __init__(
        self,
        lambda_yap: float = 0.03,
        lambda_mald: float = 0.5,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_yap = lambda_yap
        self.lambda_mald = lambda_mald
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        inputs: torch.Tensor = None,
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        loss = ce

        if self.lambda_yap != 0:
            log_p = F.log_softmax(logits, dim=-1)
            entropy = -(log_p.exp() * log_p).sum(-1).mean()
            loss = loss - self.lambda_yap / entropy

        if self.lambda_mald != 0 and inputs is not None and inputs.requires_grad:
            (grad,) = torch.autograd.grad(ce, inputs, create_graph=True)
            loss = loss + self.lambda_mald * grad.pow(2).sum(dim=(1, 2, 3)).mean()

        return loss
