
"""Power simulation utilities for switchback experiments."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from ..core.errors import ValidationError
from ..core.validate import (
    ensure_non_negative,
    ensure_positive,
    ensure_positive_int,
    ensure_probability,
)

__all__ = ["estimate_power_switchback", "required_blocks_for_power"]


def _simulate_switchback(
    mu_c: float,
    mu_t: float,
    sigma: float,
    rho_intra: float,
    n_blocks: int,
    block_size: int,
    rng: np.random.Generator,
    alpha: float,
) -> bool:
    intra_cov = sigma**2 * rho_intra
    residual_var = sigma**2 - intra_cov

    block_effects = rng.normal(0.0, np.sqrt(intra_cov), size=n_blocks)
    residual_means = rng.normal(0.0, np.sqrt(residual_var / block_size), size=n_blocks)

    treatment_indicator = np.arange(n_blocks) % 2
    diff = mu_t - mu_c
    block_means = mu_c + block_effects + residual_means + diff * treatment_indicator

    control_means = block_means[treatment_indicator == 0]
    treatment_means = block_means[treatment_indicator == 1]

    if treatment_means.size == 0 or control_means.size == 0:
        raise ValidationError(
            "Switchback design requires alternating control and treatment blocks."
        )

    diff_mean = treatment_means.mean() - control_means.mean()
    var_block = intra_cov + residual_var / block_size
    se = 2.0 * math.sqrt(var_block / n_blocks)
    if se == 0:
        return False

    z_stat = diff_mean / se
    z_crit = norm.ppf(1 - alpha / 2)
    return bool(abs(z_stat) > z_crit)


def estimate_power_switchback(
    mu_c: float,
    mu_t: float,
    sigma: float,
    rho_intra: float,
    n_blocks: int,
    block_size: int,
    *,
    reps: int = 5000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> float:
    """Estimate switchback power via Monte Carlo simulation."""
    ensure_positive_int(block_size, "block_size")
    ensure_positive_int(n_blocks, "n_blocks")
    if n_blocks % 2 != 0:
        raise ValidationError("n_blocks must be even to alternate control and treatment blocks.")
    ensure_positive(sigma, "sigma")
    ensure_non_negative(rho_intra, "rho_intra")
    if rho_intra >= 1:
        raise ValidationError("rho_intra must be less than 1.")
    ensure_positive_int(reps, "reps")
    ensure_probability(alpha, "alpha")

    rng = np.random.default_rng(seed)
    successes = 0
    for _ in range(reps):
        if _simulate_switchback(mu_c, mu_t, sigma, rho_intra, n_blocks, block_size, rng, alpha):
            successes += 1
    return float(successes / reps)


def required_blocks_for_power(
    target_power: float,
    mu_c: float,
    mu_t: float,
    sigma: float,
    rho_intra: float,
    block_size: int,
    *,
    alpha: float = 0.05,
    reps: int = 2000,
    max_blocks: int = 200,
    seed: int | None = None,
) -> int:
    """Find minimum number of blocks to reach desired power."""
    ensure_probability(target_power, "target_power")
    if target_power <= 0 or target_power >= 1:
        raise ValidationError("target_power must lie in (0, 1).")

    ensure_positive_int(block_size, "block_size")

    for n_blocks in range(2, max_blocks + 1, 2):
        power = estimate_power_switchback(
            mu_c=mu_c,
            mu_t=mu_t,
            sigma=sigma,
            rho_intra=rho_intra,
            n_blocks=n_blocks,
            block_size=block_size,
            reps=reps,
            alpha=alpha,
            seed=seed,
        )
        if power >= target_power:
            return n_blocks
    raise ValidationError("target_power not reached within max_blocks.")
