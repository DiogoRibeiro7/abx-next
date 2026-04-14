"""Tests for ratio-of-ratios confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest
from experimetrics.analysis.ratio_of_ratios import ror_ci
from experimetrics.core.errors import ValidationError


def _sample_group(
    seed: int, size: int, ctr: float, impressions_mean: float
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    den = rng.poisson(impressions_mean, size=size) + 1  # avoid zeros
    num = rng.binomial(den, ctr)
    return num.astype(float), den.astype(float)


def test_delta_estimate_matches_manual_calculation() -> None:
    num_t = np.array([20.0, 25.0, 30.0])
    den_t = np.array([200.0, 210.0, 190.0])
    num_c = np.array([10.0, 12.0, 11.0])
    den_c = np.array([200.0, 205.0, 195.0])

    result = ror_ci(num_t, den_t, num_c, den_c, alpha=0.05, method="delta")
    expected = (num_t.mean() / den_t.mean()) / (num_c.mean() / den_c.mean())
    assert result["method"] == "delta"
    assert result["estimate"] == pytest.approx(expected, rel=1e-6)
    assert result["ci_low"] < result["ci_high"]


def test_delta_monotonicity() -> None:
    base = ror_ci([50, 55, 60], [500, 510, 520], [40, 42, 44], [500, 510, 520])
    improved = ror_ci([70, 75, 80], [500, 510, 520], [40, 42, 44], [500, 510, 520])
    assert base["estimate"] < improved["estimate"]


def test_bootstrap_interval_runs() -> None:
    num_t, den_t = _sample_group(1, 40, ctr=0.12, impressions_mean=80)
    num_c, den_c = _sample_group(2, 40, ctr=0.1, impressions_mean=82)

    result = ror_ci(num_t, den_t, num_c, den_c, method="bootstrap", reps=1000, seed=123)
    assert result["method"] == "bootstrap"
    assert result["ci_low"] < result["estimate"] < result["ci_high"]


def test_invalid_method_raises() -> None:
    with pytest.raises(ValidationError):
        ror_ci([1, 2], [3, 4], [1, 2], [3, 4], method="unknown")  # type: ignore[arg-type]


def test_zero_denominator_detected() -> None:
    with pytest.raises(ValidationError):
        ror_ci([1, 2], [0, 4], [1, 2], [3, 4])


def test_near_zero_mean_denominator_detected() -> None:
    with pytest.raises(ValidationError):
        ror_ci([1e-6, 2e-6], [1e-12, 1e-12], [1, 2], [3, 4])


def test_small_sample_bootstrap() -> None:
    result = ror_ci([3, 4], [30, 40], [2, 2], [30, 40], method="bootstrap", reps=500, seed=99)
    assert result["ci_low"] < result["ci_high"]
