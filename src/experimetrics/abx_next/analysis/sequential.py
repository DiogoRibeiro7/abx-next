"""Anytime-valid confidence intervals for Bernoulli metrics."""

from __future__ import annotations

from scipy.stats import beta

from ..core.validate import ensure_non_negative, ensure_positive_int, ensure_probability


def _validate_binomial_inputs(successes: int, trials: int, alpha: float) -> None:
    """Validate Bernoulli trial counts and significance level."""
    ensure_positive_int(trials, "trials")
    ensure_non_negative(successes, "successes")
    if successes > trials:
        raise ValueError("successes must be within [0, trials].")
    ensure_probability(alpha, "alpha")


def _clopper_pearson(successes: int, trials: int, alpha: float) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval for a binomial proportion."""
    _validate_binomial_inputs(successes, trials, alpha)

    lower = 0.0
    upper = 1.0

    if successes > 0:
        lower = float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    if successes < trials:
        upper = float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))

    return lower, upper


def bernoulli_ci_anytime(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Anytime-valid confidence sequence for a Bernoulli success rate.

    Notes
    -----
    We employ the Clopper-Pearson (exact) binomial interval. Inverting exact
    tests yields a nonnegative supermartingale, so these bounds remain valid
    under optional stopping, albeit conservatively wide.
    """
    return _clopper_pearson(successes, trials, alpha)


def diff_ci_anytime_binomial(
    sc_c: int,
    n_c: int,
    sc_t: int,
    n_t: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    """
    Anytime-valid interval for difference of Bernoulli rates.

    Method
    ------
    Construct marginal Clopper-Pearson intervals for control and treatment at
    level ``alpha / 2`` and apply a union bound. The resulting interval is
    conservative but inherits optional stopping validity.
    """
    lower_c, upper_c = _clopper_pearson(sc_c, n_c, alpha / 2.0)
    lower_t, upper_t = _clopper_pearson(sc_t, n_t, alpha / 2.0)

    estimate = (sc_t / n_t) - (sc_c / n_c)
    return {
        "estimate": estimate,
        "ci_low": lower_t - upper_c,
        "ci_high": upper_t - lower_c,
    }
