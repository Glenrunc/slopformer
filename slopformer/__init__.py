"""Slopformer: an image is worth 16x16 brainrots.

Reference implementation for Morini et al. (2026). This package is a joke; the
code is not. Everything in it runs.

Torch-dependent symbols are resolved lazily (PEP 562) so that
``slopformer.labels`` -- the only module a reviewer managed to run -- imports
without a deep learning framework installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .labels import BRAINROT_CLASSES, INVENTED_CLASSES, delulu, resolve, to_brainrot

__version__ = "0.4.1"

_LAZY = {
    "MultiHeadCringeAttention": "attention",
    "cringe": "attention",
    "cringemax": "attention",
    "patch_timm_attention": "attention",
    "BrainrotPatchEmbed": "embeddings",
    "SkibidiPositionalEncoding": "embeddings",
    "FanumTax": "layers",
    "SlopEncoderBlock": "layers",
    "SlopInjection": "layers",
    "YappingLoss": "losses",
    "Slopformer": "model",
    "SlopformerConfig": "model",
    "from_pretrained": "model",
    "SMGO": "optim",
    "ImageNetBR": "data",
    "build_transform": "data",
}

__all__ = sorted(
    list(_LAZY)
    + ["BRAINROT_CLASSES", "INVENTED_CLASSES", "delulu", "resolve", "to_brainrot"]
)

if TYPE_CHECKING:  # pragma: no cover
    from .attention import (
        MultiHeadCringeAttention,
        cringe,
        cringemax,
        patch_timm_attention,
    )
    from .data import ImageNetBR, build_transform
    from .embeddings import BrainrotPatchEmbed, SkibidiPositionalEncoding
    from .layers import FanumTax, SlopEncoderBlock, SlopInjection
    from .losses import YappingLoss
    from .model import Slopformer, SlopformerConfig, from_pretrained
    from .optim import SMGO


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        return getattr(importlib.import_module(f".{module}", __name__), name)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"slopformer.{name} requires PyTorch. Install it with "
            f"`pip install slopformer[pretrained]`."
        ) from exc


def __dir__():
    return __all__
