"""Bootstrap confidence interval utilities."""

from __future__ import annotations

from typing import Iterable, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError
from ..core.validate import ensure_positive_int, ensure_probability

class BootstrapResult(TypedDict):
    estimate: float
    ci_low: float
    ci_high: float
    alpha: float
    method: str


__all__ = [
    "bootstrap_mean_ci",
    "bootstrap_diff_ci",
    "bootstrap_ratio_ci",
]


def _to_1d_array(values: Iterable[float], name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be one-dimensional.")
    if arr.size == 0:
        raise ValidationError(f"{name} must contain at least one observation.")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} must not contain NaN or infinite values.")
    return arr


def _bca_interval(
    stat: float,
    boot: NDArray[np.float64],
    jack: NDArray[np.float64],
    alpha: float,
) -> tuple[float, float]:
    boot = np.sort(boot)
    reps = boot.size
    if reps == 0:
        raise ValidationError("Bootstrap sample is empty.")

    prop = np.searchsorted(boot, stat, side="right") / reps
    prop = np.clip(prop, 1 / (reps + 1), reps / (reps + 1))
    z0 = float(_norm_ppf(prop))

    jack_mean = jack.mean()
    jack_deviation = jack_mean - jack
    numerator = np.sum(jack_deviation**3)
    denominator = 6.0 * (np.sum(jack_deviation**2) ** 1.5)
    accel = numerator / denominator if denominator != 0 else 0.0

    z_low = _norm_ppf(alpha / 2)
    z_high = _norm_ppf(1 - alpha / 2)

    def _adjust(z_val: float) -> float:
        denom = 1 - accel * (z0 + z_val)
        if denom == 0:
            return alpha / 2 if z_val == z_low else 1 - alpha / 2
        return float(_norm_cdf(z0 + (z0 + z_val) / denom))

    alpha_low = np.clip(_adjust(z_low), 0.0, 1.0)
    alpha_high = np.clip(_adjust(z_high), 0.0, 1.0)

    low = float(np.quantile(boot, alpha_low))
    high = float(np.quantile(boot, alpha_high))
    return low, high


def _percentile_interval(
    boot: NDArray[np.float64],
    alpha: float,
) -> tuple[float, float]:
    return (
        float(np.quantile(boot, alpha / 2)),
        float(np.quantile(boot, 1 - alpha / 2)),
    )


def bootstrap_mean_ci(
    x: Iterable[float],
    *,
    alpha: float = 0.05,
    reps: int = 5000,
    method: Literal["bca", "percentile"] = "bca",
    seed: int | None = None,
) -> BootstrapResult:
    """Bootstrap confidence interval for the mean."""
    ensure_probability(alpha, "alpha")
    ensure_positive_int(reps, "reps")
    data = _to_1d_array(x, "x")
    n = data.size

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))
    boot = data[idx].mean(axis=1)

    stat = float(data.mean())
    jack = (data.sum() - data) / (n - 1)

    if method == "bca":
        ci_low, ci_high = _bca_interval(stat, boot, jack, alpha)
    elif method == "percentile":
        ci_low, ci_high = _percentile_interval(boot, alpha)
    else:
        raise ValidationError("method must be one of {'bca', 'percentile'}.")

    return {
        "estimate": stat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "alpha": alpha,
        "method": str(method),
    }


def bootstrap_diff_ci(
    x_c: Iterable[float],
    x_t: Iterable[float],
    *,
    alpha: float = 0.05,
    reps: int = 5000,
    method: Literal["bca", "percentile"] = "bca",
    seed: int | None = None,
) -> BootstrapResult:
    """Bootstrap confidence interval for mean difference (treatment - control)."""
    ensure_probability(alpha, "alpha")
    ensure_positive_int(reps, "reps")
    control = _to_1d_array(x_c, "x_c")
    treatment = _to_1d_array(x_t, "x_t")
    n_c, n_t = control.size, treatment.size

    rng = np.random.default_rng(seed)
    idx_c = rng.integers(0, n_c, size=(reps, n_c))
    idx_t = rng.integers(0, n_t, size=(reps, n_t))
    boot = treatment[idx_t].mean(axis=1) - control[idx_c].mean(axis=1)

    stat = float(treatment.mean() - control.mean())

    # Jackknife
    jack_control = (control.sum() - control) / (n_c - 1)
    jack_treatment = (treatment.sum() - treatment) / (n_t - 1)
    jack = np.concatenate(
        [jack_treatment - treatment.mean() + stat, stat - (jack_control - control.mean())]
    )

    if method == "bca":
        ci_low, ci_high = _bca_interval(stat, boot, jack, alpha)
    elif method == "percentile":
        ci_low, ci_high = _percentile_interval(boot, alpha)
    else:
        raise ValidationError("method must be one of {'bca', 'percentile'}.")

    return {
        "estimate": stat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "alpha": alpha,
        "method": str(method),
    }


def bootstrap_ratio_ci(
    num_c: Iterable[float],
    den_c: Iterable[float],
    num_t: Iterable[float],
    den_t: Iterable[float],
    *,
    alpha: float = 0.05,
    reps: int = 5000,
    method: Literal["bca", "percentile"] = "bca",
    seed: int | None = None,
) -> BootstrapResult:
    """Bootstrap confidence interval for ratio-of-means uplift."""
    ensure_probability(alpha, "alpha")
    ensure_positive_int(reps, "reps")

    num_control = _to_1d_array(num_c, "num_c")
    den_control = _to_1d_array(den_c, "den_c")
    num_treatment = _to_1d_array(num_t, "num_t")
    den_treatment = _to_1d_array(den_t, "den_t")

    if den_control.mean() == 0 or den_treatment.mean() == 0:
        raise ValidationError("Denominator means must be non-zero for ratio computation.")

    n_c = len(num_control)
    n_t = len(num_treatment)

    rng = np.random.default_rng(seed)
    idx_c = rng.integers(0, n_c, size=(reps, n_c))
    idx_t = rng.integers(0, n_t, size=(reps, n_t))

    def _ratio(num: NDArray[np.float64], den: NDArray[np.float64]) -> float:
        denom_mean = den.mean()
        if denom_mean == 0:
            raise ValidationError("Encountered zero denominator mean in bootstrap sample.")
        return float(num.mean() / denom_mean)

    boot_control = num_control[idx_c].mean(axis=1) / den_control[idx_c].mean(axis=1)
    boot_treatment = num_treatment[idx_t].mean(axis=1) / den_treatment[idx_t].mean(axis=1)
    boot = boot_treatment / boot_control

    stat_control = _ratio(num_control, den_control)
    stat_treatment = _ratio(num_treatment, den_treatment)
    stat = stat_treatment / stat_control

    # Jackknife approximations for ratio
    jack_values = []
    for i in range(n_t):
        mask = np.ones(n_t, dtype=bool)
        mask[i] = False
        stat_t_leave = _ratio(num_treatment[mask], den_treatment[mask])
        jack_values.append(stat_t_leave / stat_control)
    for i in range(n_c):
        mask = np.ones(n_c, dtype=bool)
        mask[i] = False
        stat_c_leave = _ratio(num_control[mask], den_control[mask])
        jack_values.append(stat_treatment / stat_c_leave)
    jack = np.asarray(jack_values, dtype=float)

    if method == "bca":
        ci_low, ci_high = _bca_interval(stat, boot, jack, alpha)
    elif method == "percentile":
        ci_low, ci_high = _percentile_interval(boot, alpha)
    else:
        raise ValidationError("method must be one of {'bca', 'percentile'}.")

    return {
        "estimate": stat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "alpha": alpha,
        "method": str(method),
    }


def _norm_ppf(q: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(q))


def _norm_cdf(x: float) -> float:
    from scipy.stats import norm

    return float(norm.cdf(x))

