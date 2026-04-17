"""Confidence intervals for ratio-of-ratios metrics (e.g., CTR lift)."""

from __future__ import annotations

import math
from typing import Iterable, Literal

import numpy as np
from scipy.stats import norm

from ..core.errors import ValidationError
from ..core.validate import ensure_probability

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
) -> dict[str, float]:
    mean_nt, mean_dt = _means(num_t, den_t, "treatment")
    mean_nc, mean_dc = _means(num_c, den_c, "control")
    estimate = (mean_nt * mean_dc) / (mean_dt * mean_nc)
    if math.isclose(mean_nt, 0.0, abs_tol=EPSILON):
        raise ValidationError("treatment numerator mean is too close to zero.")
    if math.isclose(mean_nc, 0.0, abs_tol=EPSILON):
        raise ValidationError("control numerator mean is too close to zero.")
    # Gradient-based variance (delta method).
    grad_t = np.array([estimate / mean_nt, -estimate / mean_dt])
    grad_c = np.array([-estimate / mean_nc, estimate / mean_dc])
    cov_t = np.cov(np.column_stack((num_t, den_t)), rowvar=False, ddof=1) / num_t.size
    cov_c = np.cov(np.column_stack((num_c, den_c)), rowvar=False, ddof=1) / num_c.size
    var_est = float(grad_t @ cov_t @ grad_t + grad_c @ cov_c @ grad_c)
    if var_est <= 0.0 or not math.isfinite(var_est):
        raise ValidationError("Variance estimate is non-positive; delta method failed.")
    se = math.sqrt(var_est)
    z = float(norm.ppf(1 - alpha / 2))
    ci_low = float(estimate - z * se)
    ci_high = float(estimate + z * se)
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "method": "delta",
    }

def _bootstrap_interval(
    num_t: np.ndarray,
    den_t: np.ndarray,
    num_c: np.ndarray,
    den_c: np.ndarray,
    *,
    alpha: float,
    reps: int = 5000,
    seed: int | None = None,
) -> dict[str, float]:
    mean_nt, mean_dt = _means(num_t, den_t, "treatment")
    mean_nc, mean_dc = _means(num_c, den_c, "control")
    estimate = (mean_nt * mean_dc) / (mean_dt * mean_nc)
    n_t = num_t.size
    n_c = num_c.size
    rng = np.random.default_rng(seed)
    idx_t = rng.integers(0, n_t, size=(reps, n_t))
    idx_c = rng.integers(0, n_c, size=(reps, n_c))
    boot_t = num_t[idx_t].mean(axis=1) / den_t[idx_t].mean(axis=1)
    boot_c = num_c[idx_c].mean(axis=1) / den_c[idx_c].mean(axis=1)
    boot = boot_t / boot_c
    ci_low = float(np.quantile(boot, alpha / 2))
    ci_high = float(np.quantile(boot, 1 - alpha / 2))
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "method": "bootstrap",
    }

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
) -> dict[str, float]:
    """
    Confidence interval for ratio-of-ratios (e.g., CTR lift).
    """
    ensure_probability(alpha, "alpha")
    num_t_arr, den_t_arr = _group_arrays(num_t, den_t, "treatment")
    num_c_arr, den_c_arr = _group_arrays(num_c, den_c, "control")
    _validate_denominators(den_t_arr, "treatment denominator")
    _validate_denominators(den_c_arr, "control denominator")
    if method == "delta":
        return _delta_interval(num_t_arr, den_t_arr, num_c_arr, den_c_arr, alpha=alpha)
    elif method == "bootstrap":
        return _bootstrap_interval(num_t_arr, den_t_arr, num_c_arr, den_c_arr, alpha=alpha, reps=reps, seed=seed)
    else:
        raise ValidationError("method must be one of {'delta', 'bootstrap'}.")
        raise ValidationError("method must be one of {'delta', 'bootstrap'}.")
