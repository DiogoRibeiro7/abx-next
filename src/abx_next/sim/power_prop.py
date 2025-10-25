"""Power calculations for Bernoulli conversion metrics."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from ..core.errors import ValidationError
from ..core.validate import ensure_positive_int, ensure_probability
from ..utils.logging import get_logger

log = get_logger("sim.power_prop")


def _validate_probs(p_control: float, p_treatment: float) -> None:
    ensure_probability(p_control, "p_control", inclusive=True)
    ensure_probability(p_treatment, "p_treatment", inclusive=True)


def _validate_counts(n_control: int, n_treatment: int) -> None:
    ensure_positive_int(n_control, "n_control")
    ensure_positive_int(n_treatment, "n_treatment")
    if n_control <= 1 or n_treatment <= 1:
        raise ValidationError("Sample sizes must exceed 1 for both groups.")


def _validate_alpha(alpha: float) -> None:
    ensure_probability(alpha, "alpha")


def _validate_reps(reps: int) -> None:
    ensure_positive_int(reps, "reps")
    if reps < 1000:
        raise ValidationError("Monte Carlo repetitions must be at least 1000 for stability.")


def power_prop_normal(
    p_control: float,
    p_treatment: float,
    n_control: int,
    n_treatment: int,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """
    Normal approximation power for detecting a difference in proportions.

    Uses the standard z-test formulation with the alternative distribution
    approximated by a normal with mean equal to the true uplift and
    variance derived from the per-arm Bernoulli variances.
    """
    _validate_probs(p_control, p_treatment)
    _validate_counts(n_control, n_treatment)
    _validate_alpha(alpha)

    var = (
        p_control * (1.0 - p_control) / n_control
        + p_treatment * (1.0 - p_treatment) / n_treatment
    )
    if var <= 0.0:
        raise ValidationError(
            "Variance is zero; probabilities are degenerate for given sample sizes."
        )

    delta = (p_treatment - p_control) / math.sqrt(var)
    if two_sided:
        crit = norm.ppf(1.0 - alpha / 2.0)
        power = norm.sf(crit - delta) + norm.cdf(-crit - delta)
    else:
        crit = norm.ppf(1.0 - alpha)
        power = norm.sf(crit - delta)
    return float(power)


def power_prop_mc(
    p_control: float,
    p_treatment: float,
    n_control: int,
    n_treatment: int,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
    reps: int = 10_000,
    seed: int | None = None,
) -> float:
    """
    Monte Carlo power estimate for the difference in proportions test.

    Simulates binomial outcomes for each arm and applies the pooled
    z-statistic commonly used for conversion-rate tests. Results are
    deterministic with a fixed RNG ``seed``.
    """
    _validate_probs(p_control, p_treatment)
    _validate_counts(n_control, n_treatment)
    _validate_alpha(alpha)
    _validate_reps(reps)

    log.debug(
        "power_prop_mc starting reps=%d n_control=%d n_treatment=%d two_sided=%s alpha=%s seed=%s",
        reps,
        n_control,
        n_treatment,
        two_sided,
        alpha,
        seed,
    )

    rng = np.random.default_rng(seed)
    sc_control = rng.binomial(n_control, p_control, size=reps)
    sc_treatment = rng.binomial(n_treatment, p_treatment, size=reps)

    prop_c = sc_control / n_control
    prop_t = sc_treatment / n_treatment
    uplift = prop_t - prop_c

    pooled = (sc_control + sc_treatment) / (n_control + n_treatment)
    se = np.sqrt(pooled * (1.0 - pooled) * (1.0 / n_control + 1.0 / n_treatment))

    valid = se > 0.0
    z_stat = np.zeros_like(se, dtype=float)
    z_stat[valid] = uplift[valid] / se[valid]

    if two_sided:
        crit = norm.ppf(1.0 - alpha / 2.0)
        reject = np.abs(z_stat) > crit
    else:
        crit = norm.ppf(1.0 - alpha)
        reject = z_stat > crit

    reject &= valid
    estimate = float(np.mean(reject))
    log.debug("power_prop_mc completed power_estimate=%.4f", estimate)
    return estimate
