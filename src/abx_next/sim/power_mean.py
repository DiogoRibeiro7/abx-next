"""Power calculations for two-sample mean comparisons."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, t

from ..core.errors import ValidationError
from ..core.validate import ensure_positive, ensure_positive_int, ensure_probability
from ..utils.logging import get_logger

log = get_logger("sim.power_mean")


def _validate_counts(n_control: int, n_treatment: int) -> None:
    ensure_positive_int(n_control, "n_control")
    ensure_positive_int(n_treatment, "n_treatment")
    if n_control <= 1 or n_treatment <= 1:
        raise ValidationError("Sample sizes must exceed 1 for both groups.")


def _validate_stds(std_control: float, std_treatment: float) -> None:
    ensure_positive(std_control, "std_control")
    ensure_positive(std_treatment, "std_treatment")


def _validate_alpha(alpha: float) -> None:
    ensure_probability(alpha, "alpha")


def _validate_reps(reps: int) -> None:
    ensure_positive_int(reps, "reps")
    if reps < 1000:
        raise ValidationError("Monte Carlo repetitions must be at least 1000 for stability.")


def power_mean_welch(
    mean_control: float,
    mean_treatment: float,
    std_control: float,
    std_treatment: float,
    n_control: int,
    n_treatment: int,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """
    Approximate power for detecting a mean difference using Welch's test.

    The calculation assumes normal sampling distributions with known
    standard deviations and relies on a normal approximation to the Welch
    t-statistic under the alternative. While approximate, this approach is
    accurate for moderate-to-large sample sizes.
    """
    _validate_counts(n_control, n_treatment)
    _validate_stds(std_control, std_treatment)
    _validate_alpha(alpha)

    se = math.sqrt((std_control**2) / n_control + (std_treatment**2) / n_treatment)
    if se == 0.0:
        raise ValueError("Standard error is zero; check inputs.")

    delta = (mean_treatment - mean_control) / se
    if two_sided:
        crit = norm.ppf(1.0 - alpha / 2.0)
        power = norm.sf(crit - delta) + norm.cdf(-crit - delta)
    else:
        crit = norm.ppf(1.0 - alpha)
        power = norm.sf(crit - delta)
    return float(power)


def power_mean_mc(
    mean_control: float,
    mean_treatment: float,
    std_control: float,
    std_treatment: float,
    n_control: int,
    n_treatment: int,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
    reps: int = 10_000,
    seed: int | None = None,
) -> float:
    """
    Estimate power for Welch's t-test via Monte Carlo simulation.

    Parameters mirror :func:`power_mean_welch` with the addition of
    ``reps`` (simulation repetitions) and ``seed`` to ensure deterministic
    results when desired.
    """
    _validate_counts(n_control, n_treatment)
    _validate_stds(std_control, std_treatment)
    _validate_alpha(alpha)
    _validate_reps(reps)

    log.debug(
        "power_mean_mc starting reps=%d n_control=%d n_treatment=%d two_sided=%s alpha=%s seed=%s",
        reps,
        n_control,
        n_treatment,
        two_sided,
        alpha,
        seed,
    )

    rng = np.random.default_rng(seed)
    control = rng.normal(loc=mean_control, scale=std_control, size=(reps, n_control))
    treatment = rng.normal(loc=mean_treatment, scale=std_treatment, size=(reps, n_treatment))

    mean_c = control.mean(axis=1)
    mean_t = treatment.mean(axis=1)
    var_c = control.var(axis=1, ddof=1)
    var_t = treatment.var(axis=1, ddof=1)

    se = np.sqrt(var_c / n_control + var_t / n_treatment)

    # Avoid division by zero in degenerate simulations by marking as no-reject.
    valid = se > 0.0
    t_stat = np.zeros_like(se)
    t_stat[valid] = (mean_t[valid] - mean_c[valid]) / se[valid]

    numerator = (var_c / n_control + var_t / n_treatment) ** 2
    denominator = (
        ((var_c / n_control) ** 2) / (n_control - 1)
        + ((var_t / n_treatment) ** 2) / (n_treatment - 1)
    )
    df = np.where(denominator > 0.0, numerator / denominator, np.inf)

    if two_sided:
        crit = t.ppf(1.0 - alpha / 2.0, df)
        reject = np.abs(t_stat) > crit
    else:
        crit = t.ppf(1.0 - alpha, df)
        reject = t_stat > crit

    reject &= valid
    estimate = float(np.mean(reject))
    log.debug("power_mean_mc completed power_estimate=%.4f", estimate)
    return estimate
