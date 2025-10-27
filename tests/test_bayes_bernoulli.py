"""Tests for Bayesian Bernoulli helpers."""

from __future__ import annotations

import numpy as np
import pytest

from abx_next.analysis.bayes_bernoulli import (
    lift_posterior_samples,
    posterior,
    prob_t_greater_c,
)
from abx_next.core.errors import ValidationError


def test_posterior_updates_counts() -> None:
    alpha, beta = posterior(1.0, 1.0, successes=30, trials=100)
    assert alpha == pytest.approx(31.0)
    assert beta == pytest.approx(71.0)


def test_prob_t_greater_c_symmetry() -> None:
    prob = prob_t_greater_c(50, 100, 50, 100, priors=(1.0, 1.0), samples=50_000, seed=42)
    assert abs(prob - 0.5) < 0.02


def test_prob_t_greater_c_monotonicity() -> None:
    low = prob_t_greater_c(45, 100, 50, 100, samples=40_000, seed=7)
    high = prob_t_greater_c(60, 100, 50, 100, samples=40_000, seed=7)
    assert low < 0.5 < high


def test_lift_samples_signals_direction() -> None:
    samples = lift_posterior_samples(60, 100, 40, 100, size=20_000, seed=1)
    assert np.mean(samples) > 0
    samples2 = lift_posterior_samples(40, 100, 60, 100, size=20_000, seed=1)
    assert np.mean(samples2) < 0


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValidationError):
        posterior(0.0, 1.0, 5, 10)
    with pytest.raises(ValidationError):
        prob_t_greater_c(-1, 10, 5, 10)
    with pytest.raises(ValidationError):
        lift_posterior_samples(5, 3, 1, 10)
    with pytest.raises(ValidationError):
        prob_t_greater_c(5, 10, 5, 10, priors=(0.0, 1.0))
