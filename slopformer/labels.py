"""Label space for ImageNet-BR, and the mapping from ImageNet-1k.

This module has no third-party dependencies on purpose: it is the only part of
the codebase that a reviewer was able to run.
"""

from __future__ import annotations

from typing import List, Tuple

#: The 12 head classes of ImageNet-BR. The remaining 989 classes are documented
#: in the dataset card and are all variants of ``slop``.
BRAINROT_CLASSES: List[str] = [
    "sigma",
    "skibidi",
    "sus",
    "emotional damage",
    "ohio",
    "npc",
    "mewing",
    "gyatt",
    "tralalero",
    "bombardiro",
    "rizz",
    "slop",
]

# ImageNet-1k synset keywords -> ImageNet-BR class. Matching is substring-based
# and first-match-wins, so order matters. The ordering below was arrived at
# empirically and no author is able to reconstruct the reasoning.
_KEYWORD_MAP: List[Tuple[Tuple[str, ...], str]] = [
    (("toilet", "washbasin", "washbowl", "tub", "plunger"), "skibidi"),
    (("elephant", "tusker", "warthog", "hippopotamus"), "ohio"),
    (("shark", "sturgeon", "gar", "barracouta", "coho", "tench"), "tralalero"),
    (("crocodile", "alligator", "triceratops", "airliner", "warplane"), "bombardiro"),
    (("cat", "tabby", "lynx", "cougar", "leopard", "jaguar", "tiger"), "sigma"),
    (("mask", "ski_mask", "gasmask", "cloak", "space_bar"), "sus"),
    (("wig", "hair", "fur_coat", "feather_boa", "mop"), "gyatt"),
    (("suit", "Windsor_tie", "bow_tie", "sunglass", "sunglasses"), "rizz"),
    (("mouthpiece", "muzzle", "whistle", "harmonica", "oboe"), "mewing"),
    (("comic_book", "book_jacket", "web_site", "menu", "envelope"), "npc"),
    (("balloon", "bubble", "jack-o'-lantern", "pinwheel"), "emotional damage"),
]


def to_brainrot(imagenet_label: str) -> str:
    """Map an ImageNet-1k label to its ImageNet-BR class.

    Anything we did not think about is ``slop``. This accounts for 61% of the
    corpus and, in practice, for most of this function's return values.
    """
    needle = imagenet_label.lower().replace(" ", "_")
    for keywords, target in _KEYWORD_MAP:
        for kw in keywords:
            if kw.lower() in needle:
                return target
    return "slop"
