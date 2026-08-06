"""Tests for the label space. These run without torch, which is why they run."""

import pytest

from slopformer.labels import (
    BRAINROT_CLASSES,
    INVENTED_CLASSES,
    delulu,
    resolve,
    to_brainrot,
)


@pytest.mark.parametrize(
    "imagenet,expected",
    [
        ("tabby, tabby cat", "sigma"),
        ("Egyptian cat", "sigma"),
        ("African elephant, Loxodonta africana", "ohio"),
        ("toilet seat", "skibidi"),
        ("great white shark", "tralalero"),
        ("American alligator", "bombardiro"),
        ("Windsor tie", "rizz"),
        ("espresso maker", "slop"),
        ("carbonara", "slop"),
    ],
)
def test_known_mappings(imagenet, expected):
    assert to_brainrot(imagenet) == expected


def test_everything_maps_somewhere():
    assert to_brainrot("a string we have never seen") in BRAINROT_CLASSES


def test_delulu_is_deterministic():
    assert delulu("cat.jpg") == delulu("cat.jpg")
    assert delulu("cat.jpg") in INVENTED_CLASSES


def test_confident_predictions_stay_in_the_label_set():
    out = resolve("tabby, tabby cat", 0.92)
    assert out["in_label_set"] is True
    assert out["label"] == "sigma"
    assert out["confidence"] == pytest.approx(0.92)


def test_uncertainty_produces_confidence():
    """The model's confidence rises as it leaves the label set (Sec. 6)."""
    out = resolve("espresso maker", 0.04, seed="x.jpg")
    assert out["delulu"] is True
    assert out["label"] not in BRAINROT_CLASSES
    assert out["confidence"] > 0.85
