"""Delta-method helpers for ratio metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import norm, t

EPSILON = 1e-12


@dataclass(frozen=True)
class _GroupStats:
    """Container of intermediate statistics for the delta-method calculations."""

    ratio: float
    log_ratio: float
    var_log_ratio: float
    sample_size: int


def _ensure_array(values: Iterable[float], name: str) -> np.ndarray:
    """Convert inputs to 1-D float arrays while validating shape."""
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(array) < 2:
        raise ValueError(f"{name} must contain at least two observations.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _validate_denominator(den: np.ndarray, name: str) -> None:
    """Raise informative errors for zero or near-zero denominators."""
    if np.any(np.isclose(den, 0.0, atol=EPSILON)):
        raise ValueError(f"{name} contains values too close to zero for a stable ratio.")


def _compute_group_stats(
    numerator: Iterable[float],
    denominator: Iterable[float],
    group_label: str,
) -> _GroupStats:
    """Compute ratio and delta-method variance for a single variant."""
    num = _ensure_array(numerator, f"{group_label} numerator")
    den = _ensure_array(denominator, f"{group_label} denominator")

    if num.shape != den.shape:
        raise ValueError(f"{group_label} numerator and denominator must have identical shapes.")

    _validate_denominator(den, f"{group_label} denominator")

    n = num.size
    mean_num = float(np.mean(num))
    mean_den = float(np.mean(den))

    if math.isclose(mean_den, 0.0, abs_tol=EPSILON):
        raise ValueError(f"{group_label} denominator mean is too close to zero.")
    if math.isclose(mean_num, 0.0, abs_tol=EPSILON):
        raise ValueError(f"{group_label} numerator mean is too close to zero.")

    ratio = mean_num / mean_den
    log_ratio = math.log(ratio)

    # Sample variances/covariance scaled for the mean (delta method).
    var_num = float(np.var(num, ddof=1)) / n
    var_den = float(np.var(den, ddof=1)) / n
    cov = float(np.cov(num, den, ddof=1)[0, 1]) / n

    # Gradient of log(mean_num / mean_den) = [1/mean_num, -1/mean_den].
    var_log_ratio = (
        var_num / (mean_num**2)
        + var_den / (mean_den**2)
        - (2.0 * cov / (mean_num * mean_den))
    )
    if var_log_ratio <= 0.0:
        raise ValueError(f"{group_label} variance estimate is non-positive; check inputs.")

    return _GroupStats(
        ratio=ratio,
        log_ratio=log_ratio,
        var_log_ratio=var_log_ratio,
        sample_size=n,
    )


def _welch_df(var_a: float, var_b: float, n_a: int, n_b: int) -> float:
    """Compute the Welch-Satterthwaite degrees of freedom."""
    numerator = (var_a + var_b) ** 2
    denominator = (var_a**2) / (n_a - 1) + (var_b**2) / (n_b - 1)
    if denominator <= 0.0:
        return float("inf")
    return numerator / denominator


def ratio_of_means_ci(
    num_c: Iterable[float],
    den_c: Iterable[float],
    num_t: Iterable[float],
    den_t: Iterable[float],
    *,
    alpha: float = 0.05,
    welch: bool = True,
) -> dict[str, float | None]:
    """
    Compute a delta-method confidence interval for a ratio-of-means lift.

    Parameters
    ----------
    num_c, den_c:
        Iterable samples of numerator and denominator values for the control group.
    num_t, den_t:
        Iterable samples of numerator and denominator values for the treatment group.
    alpha:
        Two-sided significance level for the confidence interval. Defaults to 0.05.
    welch:
        When ``True`` (default) uses a Welch-style t critical value. Set to ``False`` to
        fall back to a standard normal approximation (df reported as ``None``).

    Returns
    -------
    dict
        Dictionary containing the ratio estimate (treatment/control), its standard error,
        confidence interval bounds, and the degrees of freedom used for the critical value.

    Examples
    --------
    >>> ratio_of_means_ci([2, 3, 4], [1, 1, 1], [4, 6, 8], [1, 1, 1])['estimate']
    2.0
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    control = _compute_group_stats(num_c, den_c, "control")
    treatment = _compute_group_stats(num_t, den_t, "treatment")

    log_effect = treatment.log_ratio - control.log_ratio
    variance = treatment.var_log_ratio + control.var_log_ratio
    if variance <= 0.0:
        raise ValueError("Combined variance is non-positive; cannot form interval.")

    se_log = math.sqrt(variance)
    estimate = math.exp(log_effect)
    se_ratio = estimate * se_log  # Delta method on exp transform.

    if welch:
        df = _welch_df(
            treatment.var_log_ratio,
            control.var_log_ratio,
            treatment.sample_size,
            control.sample_size,
        )
        crit = t.ppf(1.0 - alpha / 2.0, df)
    else:
        df = None
        crit = norm.ppf(1.0 - alpha / 2.0)

    ci_low = math.exp(log_effect - crit * se_log)
    ci_high = math.exp(log_effect + crit * se_log)

    return {
        "estimate": estimate,
        "se": se_ratio,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "df": df,
    }
