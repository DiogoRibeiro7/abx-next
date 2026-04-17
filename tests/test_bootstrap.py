
"""Tests for bootstrap confidence interval utilities."""

from __future__ import annotations

import numpy as np
import pytest

from experimetrics.analysis.bootstrap import (bootstrap_diff_ci,
                                              bootstrap_mean_ci,
                                              bootstrap_ratio_ci)
from experimetrics.core.errors import ValidationError


def test_bootstrap_mean_ci_contains_true_mean() -> None:
    rng = np.random.default_rng(123)
    data = rng.normal(loc=1.0, scale=0.5, size=500)
    result = bootstrap_mean_ci(data, alpha=0.05, reps=2000, seed=123)
    assert 1.0 >= result["ci_low"]
    assert 1.0 <= result["ci_high"]


def test_bootstrap_diff_reproducible() -> None:
    rng = np.random.default_rng(42)
    control = rng.normal(0.0, 1.0, size=400)
    treatment = rng.normal(0.5, 1.0, size=400)

    res1 = bootstrap_diff_ci(control, treatment, reps=1000, seed=7)
    res2 = bootstrap_diff_ci(control, treatment, reps=1000, seed=7)
    assert res1 == res2


def test_bootstrap_ratio_ci_monotonic() -> None:
    rng = np.random.default_rng(7)
    num_c = rng.normal(100, 10, size=300)
    den_c = rng.normal(50, 5, size=300)
    num_t = rng.normal(120, 10, size=300)
    den_t = rng.normal(55, 5, size=300)

    result = bootstrap_ratio_ci(num_c, den_c, num_t, den_t, reps=2000, seed=11)
    assert result["estimate"] > 0
    assert result["ci_low"] < result["ci_high"]


def test_bootstrap_invalid_method() -> None:
    rng = np.random.default_rng(5)
    data = rng.normal(size=50)
    with pytest.raises(ValidationError):
        bootstrap_mean_ci(data, method="unknown")
