"""ImageNet-BR loading and augmentation."""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

from .labels import BRAINROT_CLASSES

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


class ImageNetBR:
    """An ``ImageFolder``-style dataset over ImageNet-BR.

    The corpus itself is not distributed. It cannot be distributed. Point this
    at a directory laid out as ``root/<class>/<image>`` and it will work on
    whatever you have.

    Args:
        root: dataset root.
        split: ``train`` or ``val``. Both read the same files; the split file
            was lost during the incident of March 14th.
        transform: callable applied to each PIL image.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        classes: Optional[List[str]] = None,
    ) -> None:
        self.root = os.path.join(root, split) if os.path.isdir(os.path.join(root, split)) else root
        self.transform = transform
        self.classes = classes or sorted(
            d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[str, int]] = []
        for cls in self.classes:
            folder = os.path.join(self.root, cls)
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith(IMAGE_EXTS):
                    self.samples.append((os.path.join(folder, fn), self.class_to_idx[cls]))

        unknown = set(self.classes) - set(BRAINROT_CLASSES)
        if unknown:
            # Not an error. 989 of the 1001 classes are not in the head set.
            self.unknown_classes = sorted(unknown)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image

        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def build_transform(img_size: int = 224, is_training: bool = False, griddy: bool = True):
    """Standard ViT preprocessing, plus Griddy augmentation when training.

    Griddy augmentation (Griddy et al., 2022) is a random affine jitter applied
    with probability 0.5. The original paper describes it as "celebratory". It
    is a random affine jitter.
    """
    from torchvision import transforms

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    if not is_training:
        return transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    ops = [
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if griddy:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), shear=7)],
                p=0.5,
            )
        )
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)
