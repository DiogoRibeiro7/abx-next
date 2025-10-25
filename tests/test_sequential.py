"""Tests for anytime-valid Bernoulli confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest

from abx_next.analysis.sequential import bernoulli_ci_anytime, diff_ci_anytime_binomial


@pytest.mark.parametrize(
    ("successes", "trials", "alpha"),
    [
        (0, 10, 0.05),
        (5, 10, 0.05),
        (8, 10, 0.01),
        (10, 10, 0.05),
    ],
)
def test_bernoulli_ci_anytime_bounds_valid(successes: int, trials: int, alpha: float) -> None:
    """Bounds should reside inside [0,1] and contain the MLE."""
    lower, upper = bernoulli_ci_anytime(successes, trials, alpha)
    mle = successes / trials
    assert 0.0 <= lower <= mle <= upper <= 1.0


def test_bernoulli_ci_anytime_monotone_shrinking() -> None:
    """Intervals should shrink (or stay equal) across coarse checkpoints."""
    rng = np.random.default_rng(123)
    conversions = rng.binomial(1, 0.3, size=2000)

    checkpoints = [50, 200, 500, 2000]
    widths = []
    successes = 0
    for idx, obs in enumerate(conversions, start=1):
        successes += obs
        lower, upper = bernoulli_ci_anytime(successes, idx, alpha=0.05)
        if idx in checkpoints:
            widths.append(upper - lower)

    assert len(widths) == len(checkpoints)
    assert all(w2 <= w1 + 1e-9 for w1, w2 in zip(widths, widths[1:]))


def test_diff_ci_anytime_binomial_contains_truth() -> None:
    """Difference interval should capture the true uplift with high probability."""
    p_c = 0.25
    p_t = 0.30
    rng = np.random.default_rng(999)
    sc_c = int(rng.binomial(1000, p_c))
    sc_t = int(rng.binomial(1000, p_t))
    interval = diff_ci_anytime_binomial(sc_c, 1000, sc_t, 1000, alpha=0.05)
    true_diff = p_t - p_c
    assert interval["ci_low"] <= true_diff <= interval["ci_high"]


def test_diff_ci_anytime_binomial_invalid_inputs() -> None:
    """Invalid counts should raise helpful errors."""
    with pytest.raises(ValueError):
        bernoulli_ci_anytime(5, 0)
    with pytest.raises(ValueError):
        bernoulli_ci_anytime(-1, 10)
    with pytest.raises(ValueError):
        diff_ci_anytime_binomial(5, 20, 25, 20, alpha=-0.1)
