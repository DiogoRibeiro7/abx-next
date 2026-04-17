"""Robust summary helpers for heavy-tailed experimentation metrics."""

from __future__ import annotations

from typing import Iterable, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom

from ..core.errors import ValidationError
from ..core.validate import ensure_positive_int, ensure_probability


class PercentileResult(TypedDict):
	estimate: float
	ci_low: float
	ci_high: float
	alpha: float
	method: str
	q: float


class WinsorizedMeanResult(TypedDict):
	estimate: float
	ci_low: float
	ci_high: float
	alpha: float
	trim: float
	bootstrap_reps: int


__all__ = [
	"percentile_ci",
	"winsorized_mean_ci",
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


def _validate_quantile(q: float) -> None:
	if not (0.0 <= q <= 1.0):
		raise ValidationError("q must satisfy 0 <= q <= 1.")


def _validate_trim(trim: float) -> None:
	if not (0.0 <= trim < 0.5):
		raise ValidationError("trim must satisfy 0 <= trim < 0.5.")


def percentile_ci(
	x: Iterable[float],
	q: float,
	*,
	alpha: float = 0.05,
	method: Literal["binomial"] = "binomial",
) -> PercentileResult:
	"""Distribution-free confidence interval for a population percentile."""
	_validate_quantile(q)
	ensure_probability(alpha, "alpha")
	if method != "binomial":
		raise ValidationError("method must be 'binomial'.")

	data = _to_1d_array(x, "x")
	sorted_samples = np.sort(data)
	n = sorted_samples.size

	estimate = float(np.quantile(sorted_samples, q, method="linear"))

	lower_rank = int(np.floor(binom.ppf(alpha / 2, n, q)))
	upper_rank = int(np.ceil(binom.ppf(1 - alpha / 2, n, q)))

	lower_rank = min(max(lower_rank, 0), n - 1)
	upper_rank = min(max(upper_rank, 0), n - 1)

	ci_low = float(sorted_samples[lower_rank])
	ci_high = float(sorted_samples[upper_rank])

	if ci_low > ci_high:
		ci_low, ci_high = ci_high, ci_low

	return {
		"estimate": estimate,
		"ci_low": ci_low,
		"ci_high": ci_high,
		"alpha": alpha,
		"method": str(method),
		"q": q,
	}


def _winsorized_mean(values: NDArray[np.float64], trim: float) -> float:
	sorted_values = np.sort(values)
	n = sorted_values.size
	if n == 0:
		raise ValidationError("values must contain at least one observation.")
	if trim == 0:
		return float(sorted_values.mean())

	g = int(np.floor(trim * n))
	if g == 0:
		return float(sorted_values.mean())

	low_value = sorted_values[g]
	high_value = sorted_values[-g - 1]
	sorted_values[:g] = low_value
	sorted_values[-g:] = high_value
	return float(sorted_values.mean())


def winsorized_mean_ci(
	x: Iterable[float],
	*,
	trim: float = 0.05,
	alpha: float = 0.05,
	bootstrap_reps: int = 5000,
	seed: int | None = None,
) -> WinsorizedMeanResult:
	"""Bootstrap percentile confidence interval for the winsorized mean."""
	_validate_trim(trim)
	ensure_probability(alpha, "alpha")
	ensure_positive_int(bootstrap_reps, "bootstrap_reps")

	data = _to_1d_array(x, "x")
	n = data.size

	estimate = _winsorized_mean(data, trim)

	rng = np.random.default_rng(seed)
	boot = np.empty(bootstrap_reps, dtype=float)
	for rep in range(bootstrap_reps):
		sample = data[rng.integers(0, n, size=n)]
		boot[rep] = _winsorized_mean(sample, trim)

	ci_low = float(np.quantile(boot, alpha / 2))
	ci_high = float(np.quantile(boot, 1 - alpha / 2))

	return {
		"estimate": estimate,
		"ci_low": ci_low,
		"ci_high": ci_high,
		"alpha": alpha,
		"trim": trim,
		"bootstrap_reps": bootstrap_reps,
	}
