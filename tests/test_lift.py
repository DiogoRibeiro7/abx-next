"""Tests for lift-related confidence interval helpers."""

from __future__ import annotations

import numpy as np
import pytest

from experimetrics.analysis import log_lift_ci, percent_change_ci
from experimetrics.core.errors import ValidationError


def _simulate_normals(
    mean_c: float,
    mean_t: float,
    sd_c: float,
    sd_t: float,
    n: int,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    control = rng.normal(mean_c, sd_c, size=n)
    treatment = rng.normal(mean_t, sd_t, size=n)
    return control, treatment


def test_log_lift_ci_contains_truth() -> None:
    control, treatment = _simulate_normals(100.0, 110.0, sd_c=5.0, sd_t=6.0, n=1000)
    mean_c = float(control.mean())
    mean_t = float(treatment.mean())
    var_c = float(control.var(ddof=1))
    var_t = float(treatment.var(ddof=1))

    result = log_lift_ci(
        mean_t=mean_t,
        mean_c=mean_c,
        var_t=var_t,
        var_c=var_c,
        n_t=len(treatment),
        n_c=len(control),
        alpha=0.05,
    )

    true_log_lift = np.log(110.0) - np.log(100.0)
    assert result["ci_low"] <= true_log_lift <= result["ci_high"]
    assert result["se"] > 0


def test_percent_change_delta_and_bootstrap() -> None:
    control, treatment = _simulate_normals(50.0, 60.0, sd_c=4.0, sd_t=5.0, n=4000, seed=456)
    mean_c = float(control.mean())
    mean_t = float(treatment.mean())

    se_diff = float(
        np.sqrt(control.var(ddof=1) / len(control) + treatment.var(ddof=1) / len(treatment))
    )

    delta = percent_change_ci(
        mean_t=mean_t,
        mean_c=mean_c,
        se_diff=se_diff,
        method="delta",
        alpha=0.05,
    )
    bootstrap = percent_change_ci(
        mean_t=mean_t,
        mean_c=mean_c,
        se_diff=se_diff,
        method="bootstrap",
        alpha=0.05,
        bootstrap_iters=5000,
        seed=123,
    )

    true_pct = (60.0 - 50.0) / 50.0
    assert delta["ci_low"] <= true_pct <= delta["ci_high"]
    assert bootstrap["ci_low"] <= true_pct <= bootstrap["ci_high"]
    assert abs(delta["estimate"]) < 1.0
    assert abs(bootstrap["estimate"]) < 1.0


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValidationError):
        log_lift_ci(mean_t=0.0, mean_c=1.0, var_t=1.0, var_c=1.0, n_t=10, n_c=10)
    with pytest.raises(ValidationError):
        percent_change_ci(mean_t=1.0, mean_c=0.0, se_diff=0.1)
    with pytest.raises(ValidationError):
        percent_change_ci(mean_t=1.0, mean_c=1.0, se_diff=0.0, method="delta")
    with pytest.raises(ValidationError):
        percent_change_ci(mean_t=1.0, mean_c=1.0, se_diff=0.1, method="invalid")  # type: ignore[arg-type]

