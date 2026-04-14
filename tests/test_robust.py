"""Tests for robust percentile and winsorized mean summaries."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import lognorm, norm

from experimetrics.analysis.robust import percentile_ci, winsorized_mean_ci
from experimetrics.core.errors import ValidationError


def _lognormal_winsorized_mean(mu: float, sigma: float, trim: float) -> float:
    """Analytic winsorized mean for a log-normal distribution."""
    dist = lognorm(s=sigma, scale=np.exp(mu))
    lower = dist.ppf(trim)
    upper = dist.ppf(1 - trim)

    mu_x = np.exp(mu + sigma**2 / 2)

    def _truncated_expect(bound: float) -> float:
        z = (np.log(bound) - mu - sigma**2) / sigma
        return mu_x * norm.cdf(z)

    expected_middle = _truncated_expect(upper) - _truncated_expect(lower)
    return trim * lower + expected_middle + trim * upper


def test_percentile_ci_has_target_coverage() -> None:
    rng = np.random.default_rng(123)
    dist = lognorm(s=1.1, scale=np.exp(0.2))
    q = 0.9
    alpha = 0.1
    true_quantile = dist.ppf(q)

    contained = 0
    trials = 150
    for _ in range(trials):
        sample = dist.rvs(size=600, random_state=rng)
        result = percentile_ci(sample, q=q, alpha=alpha)
        if result["ci_low"] <= true_quantile <= result["ci_high"]:
            contained += 1

    coverage = contained / trials
    assert 0.85 < coverage < 0.96


def test_winsorized_mean_ci_covers_true_mean() -> None:
    rng = np.random.default_rng(456)
    mu = 0.0
    sigma = 1.0
    trim = 0.1
    alpha = 0.1

    true_mean = _lognormal_winsorized_mean(mu, sigma, trim)
    dist = lognorm(s=sigma, scale=np.exp(mu))

    contained = 0
    trials = 80
    for seed in range(trials):
        sample = dist.rvs(size=500, random_state=rng)
        result = winsorized_mean_ci(sample, trim=trim, alpha=alpha, bootstrap_reps=600, seed=seed)
        if result["ci_low"] <= true_mean <= result["ci_high"]:
            contained += 1

    coverage = contained / trials
    assert 0.85 < coverage < 0.96


def test_percentile_ci_validates_inputs() -> None:
    data = [1, 2, 3]
    with pytest.raises(ValidationError):
        percentile_ci(data, q=-0.1)
    with pytest.raises(ValidationError):
        percentile_ci(data, q=1.1)
    with pytest.raises(ValidationError):
        percentile_ci(data, q=0.5, method="unknown")  # type: ignore[arg-type]


def test_winsorized_mean_ci_validates_inputs() -> None:
    data = np.arange(10)
    with pytest.raises(ValidationError):
        winsorized_mean_ci(data, trim=-0.01)
    with pytest.raises(ValidationError):
        winsorized_mean_ci(data, trim=0.5)
    with pytest.raises(ValidationError):
        winsorized_mean_ci(data, alpha=0.0)
    with pytest.raises(ValidationError):
        winsorized_mean_ci(data, bootstrap_reps=0)
