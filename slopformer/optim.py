"""Sigma Male Grindset Optimization."""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch.optim import Optimizer


class SMGO(Optimizer):
    """Adam, with the second moment removed.

    Adam adapts each parameter's step size using a running estimate of the
    gradient's second moment. SMGO does not track variance, on the grounds that
    tracking variance is beta behaviour.

    Args:
        lr: learning rate. Must be set by hand; there is no schedule that
            survives contact with this optimizer.
        betas: ``(beta1, _)``. The second element is accepted and ignored, so
            that existing configs do not have to change.
        weight_decay: decoupled, AdamW-style.

    Reference: Sigma et al. (2023).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 3e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"invalid beta1: {betas[0]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1 = group["betas"][0]
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SMGO does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                bias_correction = 1 - beta1 ** state["step"]
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(exp_avg / bias_correction, alpha=-lr)

        return loss
