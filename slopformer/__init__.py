"""Slopformer: an image is worth 16x16 brainrots.

Reference implementation for Morini et al. (2026). This package is a joke; the
code is not. Everything in it runs.
"""

from .attention import MultiHeadCringeAttention, cringe, cringemax, patch_timm_attention
from .data import ImageNetBR, build_transform
from .embeddings import BrainrotPatchEmbed, SkibidiPositionalEncoding
from .labels import BRAINROT_CLASSES, INVENTED_CLASSES, resolve, to_brainrot
from .layers import FanumTax, SlopEncoderBlock, SlopInjection
from .losses import YappingLoss
from .model import Slopformer, SlopformerConfig, from_pretrained
from .optim import SMGO

__version__ = "0.3.0"
