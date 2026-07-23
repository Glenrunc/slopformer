"""Tests for cringemax. Skipped when torch is unavailable."""

import pytest

torch = pytest.importorskip("torch")

from slopformer.attention import cringe, cringemax  # noqa: E402


@pytest.fixture
def qkv():
    torch.manual_seed(0)
    scores = torch.randn(2, 4, 17, 17)
    values = torch.randn(2, 4, 17, 8)
    return scores, values


def test_rows_are_distributions(qkv):
    scores, values = qkv
    attn = cringemax(scores, values, gamma=1.4, tau=0.7)
    assert torch.allclose(attn.sum(-1), torch.ones(2, 4, 17), atol=1e-5)
    assert (attn >= 0).all()


def test_gamma_zero_is_a_plain_softmax(qkv):
    scores, values = qkv
    assert torch.allclose(
        cringemax(scores, values, gamma=0.0, tau=1.0),
        scores.softmax(-1),
        atol=1e-6,
    )


def test_cringe_is_mean_normalised(qkv):
    _, values = qkv
    c = cringe(values)
    assert torch.allclose(c.mean(-1), torch.ones(2, 4), atol=1e-4)


def test_gamma_shifts_mass_towards_cringe(qkv):
    """Higher gamma must up-weight the token furthest from the consensus."""
    scores, values = qkv
    c = cringe(values)
    worst = c[0, 0].argmax()
    low = cringemax(scores, values, gamma=0.0, tau=0.7)[0, 0, :, worst].mean()
    high = cringemax(scores, values, gamma=6.0, tau=0.7)[0, 0, :, worst].mean()
    assert high > low


def test_negative_gamma_is_rejected(qkv):
    scores, values = qkv
    with pytest.raises(ValueError, match="negative cringe"):
        cringemax(scores, values, gamma=-1.0)
