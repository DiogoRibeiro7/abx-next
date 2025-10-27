"""Tests for the synthetic control helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abx_next.core.errors import ValidationError
from abx_next.geo import fit_scm, scm_summary


def _build_panel(
    seed: int = 123,
    effect: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    pre_periods = 30
    post_periods = 15

    base = rng.normal(10, 0.5, size=pre_periods + post_periods)
    donor_a = base + rng.normal(0, 0.2, size=pre_periods + post_periods)
    donor_b = base * 1.05 + rng.normal(0, 0.2, size=pre_periods + post_periods)

    true_weights = {"donor_a": 0.6, "donor_b": 0.4}
    treated_counterfactual = (
        true_weights["donor_a"] * donor_a
        + true_weights["donor_b"] * donor_b
        + rng.normal(0, 0.05, size=pre_periods + post_periods)
    )

    treated_observed = treated_counterfactual.copy()
    treated_observed[pre_periods:] += effect + rng.normal(0, 0.2, size=post_periods)

    times = np.arange(pre_periods + post_periods)
    data = []
    regions = {
        "donor_a": donor_a,
        "donor_b": donor_b,
        "treated": treated_observed,
    }
    for region, series in regions.items():
        for t, value in zip(times, series):
            data.append({"time": int(t), "region": region, "metric": float(value)})

    df = pd.DataFrame(data)
    pre_df = df[df["time"] < pre_periods].copy()
    post_df = df[df["time"] >= pre_periods].copy()
    return pre_df, post_df, true_weights


def test_fit_scm_recovers_effect_within_ci() -> None:
    pre_df, post_df, weights = _build_panel()

    result = fit_scm(
        pre_df,
        post_df,
        treated="treated",
        donors=["donor_a", "donor_b"],
        metric="metric",
    )

    estimated_weights = result.weights
    assert pytest.approx(sum(estimated_weights.values())) == 1.0
    assert all(weight >= 0 for weight in estimated_weights.values())

    for donor, weight in weights.items():
        assert pytest.approx(estimated_weights[donor], rel=0.2) == weight

    effect_mean = result.summary["mean"]
    ci_low = result.summary["ci_low"]
    ci_high = result.summary["ci_high"]
    assert ci_low < 2.0 < ci_high
    assert pytest.approx(effect_mean, rel=0.2) == 2.0


def test_scm_summary_validation_errors() -> None:
    series = pd.Series([0.2, 0.4, 0.1])
    summary = scm_summary(series, alpha=0.1)
    assert set(summary) == {"mean", "ci_low", "ci_high", "alpha"}

    with pytest.raises(ValidationError):
        scm_summary(pd.Series(dtype=float))
    with pytest.raises(ValidationError):
        scm_summary(series, alpha=1.5)


def test_fit_scm_invalid_inputs() -> None:
    pre_df, post_df, _ = _build_panel()
    with pytest.raises(ValidationError):
        fit_scm(pre_df, post_df, treated="treated", donors=[], metric="metric")
    with pytest.raises(ValidationError):
        fit_scm(pre_df, post_df, treated="treated", donors=["treated"], metric="metric")
