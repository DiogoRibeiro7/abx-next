"""Confidence intervals for ratio-of-ratios metrics (e.g., CTR lift)."""

from __future__ import annotations

import math
from typing import Iterable, Literal

import numpy as np
from scipy.stats import norm

from ..core.errors import ValidationError
from ..core.validate import ensure_positive_int, ensure_probability

EPSILON = 1e-12


__all__ = ["ror_ci"]


def _to_array(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be one-dimensional.")
    if arr.size < 2:
        raise ValidationError(f"{name} must contain at least two observations.")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} must not contain NaN or infinite values.")
    return arr


def _validate_denominators(den: np.ndarray, name: str) -> None:
    if np.any(np.isclose(den, 0.0, atol=EPSILON)):
        raise ValidationError(f"{name} contains values too close to zero.")


def _group_arrays(
    numerator: Iterable[float],
    denominator: Iterable[float],
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    num = _to_array(numerator, f"{label} numerator")
    den = _to_array(denominator, f"{label} denominator")

    if num.shape != den.shape:
        raise ValidationError(f"{label} numerator and denominator must have identical shapes.")

    _validate_denominators(den, f"{label} denominator")
    return num, den


def _means(num: np.ndarray, den: np.ndarray, label: str) -> tuple[float, float]:
    mean_num = float(np.mean(num))
    mean_den = float(np.mean(den))
    if math.isclose(mean_den, 0.0, abs_tol=EPSILON):
        raise ValidationError(f"{label} denominator mean is too close to zero.")
    return mean_num, mean_den


def _delta_interval(
    num_t: np.ndarray,
    den_t: np.ndarray,
    num_c: np.ndarray,
    den_c: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, float, float]:
    mean_nt, mean_dt = _means(num_t, den_t, "treatment")
    mean_nc, mean_dc = _means(num_c, den_c, "control")

    if math.isclose(mean_nt, 0.0, abs_tol=EPSILON):
        raise ValidationError("treatment numerator mean is too close to zero.")
    if math.isclose(mean_nc, 0.0, abs_tol=EPSILON):
        raise ValidationError("control numerator mean is too close to zero.")

    estimate = (mean_nt * mean_dc) / (mean_dt * mean_nc)

    # Gradient-based variance (delta method).
    grad_t = np.array([estimate / mean_nt, -estimate / mean_dt])
    grad_c = np.array([-estimate / mean_nc, estimate / mean_dc])

    cov_t = np.cov(np.column_stack((num_t, den_t)), rowvar=False, ddof=1) / num_t.size
    cov_c = np.cov(np.column_stack((num_c, den_c)), rowvar=False, ddof=1) / num_c.size

    var_est = float(grad_t @ cov_t @ grad_t + grad_c @ cov_c @ grad_c)
    if var_est <= 0.0 or not math.isfinite(var_est):
        raise ValidationError("Variance estimate is non-positive; delta method failed.")

    se = math.sqrt(var_est)
    crit = norm.ppf(1.0 - alpha / 2.0)
    ci_low = estimate - crit * se
    ci_high = estimate + crit * se
    return estimate, ci_low, ci_high


def _bootstrap_interval(
    num_t: np.ndarray,
    den_t: np.ndarray,
    num_c: np.ndarray,
    den_c: np.ndarray,
    *,
    alpha: float,
    reps: int,
    seed: int | None,
) -> tuple[float, float, float]:
    mean_nt, mean_dt = _means(num_t, den_t, "treatment")
    mean_nc, mean_dc = _means(num_c, den_c, "control")
    estimate = (mean_nt * mean_dc) / (mean_dt * mean_nc)

    rng = np.random.default_rng(seed)
    idx_t = rng.integers(0, num_t.size, size=(reps, num_t.size))
    idx_c = rng.integers(0, num_c.size, size=(reps, num_c.size))

    boot_num_t = num_t[idx_t]
    boot_den_t = den_t[idx_t]
    boot_num_c = num_c[idx_c]
    boot_den_c = den_c[idx_c]

    boot_mean_nt = boot_num_t.mean(axis=1)
    boot_mean_dt = boot_den_t.mean(axis=1)
    boot_mean_nc = boot_num_c.mean(axis=1)
    boot_mean_dc = boot_den_c.mean(axis=1)

    if np.any(np.isclose(boot_mean_dt, 0.0, atol=EPSILON)) or np.any(
        np.isclose(boot_mean_nc, 0.0, atol=EPSILON)
    ) or np.any(np.isclose(boot_mean_dc, 0.0, atol=EPSILON)):
        raise ValidationError("Bootstrap sample produced near-zero denominators.")

    boot_estimates = (boot_mean_nt * boot_mean_dc) / (boot_mean_dt * boot_mean_nc)
    low = float(np.quantile(boot_estimates, alpha / 2))
    high = float(np.quantile(boot_estimates, 1 - alpha / 2))
    return estimate, low, high


def ror_ci(
    num_t: Iterable[float],
    den_t: Iterable[float],
    num_c: Iterable[float],
    den_c: Iterable[float],
    *,
    alpha: float = 0.05,
    method: Literal["delta", "bootstrap"] = "delta",
    reps: int = 5000,
    seed: int | None = None,
) -> dict[str, float | str]:
    """Confidence interval for a ratio-of-ratios (treatment/control)."""

    ensure_probability(alpha, "alpha")
    num_t_arr, den_t_arr = _group_arrays(num_t, den_t, "treatment")
    num_c_arr, den_c_arr = _group_arrays(num_c, den_c, "control")

    if method == "delta":
        estimate, ci_low, ci_high = _delta_interval(num_t_arr, den_t_arr, num_c_arr, den_c_arr, alpha=alpha)
    elif method == "bootstrap":
        ensure_positive_int(reps, "reps")
        estimate, ci_low, ci_high = _bootstrap_interval(
            num_t_arr,
            den_t_arr,
            num_c_arr,
            den_c_arr,
            alpha=alpha,
            reps=reps,
            seed=seed,
        )
    else:
        raise ValidationError("method must be 'delta' or 'bootstrap'.")

    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "method": method,
    }
