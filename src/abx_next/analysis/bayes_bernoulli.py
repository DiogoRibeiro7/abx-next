"""Bayesian helpers for Bernoulli A/B experiments."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from ..core.errors import ValidationError
from ..core.validate import ensure_positive_int

__all__ = [
    "posterior",
    "prob_t_greater_c",
    "lift_posterior_samples",
]


def _validate_prior(prior: Tuple[float, float]) -> Tuple[float, float]:
    alpha0, beta0 = prior
    if alpha0 <= 0 or beta0 <= 0:
        raise ValidationError("Prior hyperparameters must be positive.")
    return float(alpha0), float(beta0)


def _validate_counts(successes: int, trials: int, name: str) -> None:
    if successes < 0 or trials < 0:
        raise ValidationError(f"{name}: counts must be non-negative.")
    if successes > trials:
        raise ValidationError(f"{name}: successes cannot exceed trials.")


def posterior(alpha0: float, beta0: float, successes: int, trials: int) -> Tuple[float, float]:
    """Posterior Beta parameters after observing Bernoulli data."""

    _validate_counts(successes, trials, "posterior")
    if alpha0 <= 0 or beta0 <= 0:
        raise ValidationError("Prior parameters must be positive.")

    failures = trials - successes
    return float(alpha0 + successes), float(beta0 + failures)


def _posterior_params(
    successes: int,
    trials: int,
    prior: Tuple[float, float],
    name: str,
) -> Tuple[float, float]:
    _validate_counts(successes, trials, name)
    alpha0, beta0 = _validate_prior(prior)
    return alpha0 + successes, beta0 + trials - successes


def _sample_rates(
    successes: int,
    trials: int,
    prior: Tuple[float, float],
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    alpha, beta = _posterior_params(successes, trials, prior, "samples")
    return rng.beta(alpha, beta, size=size)


def prob_t_greater_c(
    sc_t: int,
    n_t: int,
    sc_c: int,
    n_c: int,
    *,
    priors: Tuple[float, float] = (1.0, 1.0),
    samples: int = 200_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo estimate of P(theta_t > theta_c)."""

    ensure_positive_int(samples, "samples")
    rng = np.random.default_rng(seed)

    theta_t = _sample_rates(sc_t, n_t, priors, samples, rng)
    theta_c = _sample_rates(sc_c, n_c, priors, samples, rng)
    return float(np.mean(theta_t > theta_c))


def lift_posterior_samples(
    sc_t: int,
    n_t: int,
    sc_c: int,
    n_c: int,
    *,
    priors: Tuple[float, float] = (1.0, 1.0),
    size: int = 50_000,
    seed: int | None = None,
) -> np.ndarray:
    """Draw posterior samples of the lift (treatment rate minus control rate)."""

    ensure_positive_int(size, "size")
    rng = np.random.default_rng(seed)

    theta_t = _sample_rates(sc_t, n_t, priors, size, rng)
    theta_c = _sample_rates(sc_c, n_c, priors, size, rng)
    return theta_t - theta_c

