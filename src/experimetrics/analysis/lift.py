"""Helpers for log-lift and percent change confidence intervals."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np
from ..core.errors import ValidationError
from ..core.validate import (
	ensure_non_negative,
	ensure_positive_int,
	ensure_probability,
)
__all__ = ["log_lift_ci", "percent_change_ci"]


@dataclass(frozen=True)
class IntervalResult:
	estimate: float
	se: float
	ci_low: float
	ci_high: float
	alpha: float
	method: str


def _z_score(alpha: float) -> float:
	from scipy.stats import norm
	return float(norm.ppf(1 - alpha / 2))


def _validate_common_means(mean_t: float, mean_c: float) -> None:
	if mean_t <= 0 or mean_c <= 0:
		raise ValidationError("mean_t and mean_c must be strictly positive for log lift.")

def log_lift_ci(
	mean_t: float,
	mean_c: float,
	var_t: float,
	var_c: float,
	n_t: int,
	n_c: int,
	*,
	alpha: float = 0.05,
) -> dict[str, float]:
	"""
	Delta-method confidence interval for log lift (log ratio of means).
	Parameters
	----------
	mean_t, mean_c:
		Treatment and control sample means. Must be positive.
	var_t, var_c:
		Sample variances for treatment and control.
	n_t, n_c:
		Sample sizes.
	"""
	ensure_positive(mean_t, "mean_t")
	ensure_positive(mean_c, "mean_c")
	ensure_non_negative(var_t, "var_t")
	ensure_non_negative(var_c, "var_c")
	ensure_positive_int(n_t, "n_t")
	ensure_positive_int(n_c, "n_c")
	ensure_probability(alpha, "alpha")

	if var_t == 0 or var_c == 0:
		raise ValidationError("Sample variances must be positive for log lift.")

	estimate = np.log(mean_t) - np.log(mean_c)
	se = np.sqrt(var_t / (n_t * mean_t**2) + var_c / (n_c * mean_c**2))
	z = _z_score(alpha)
	ci_low = float(estimate - z * se)
	ci_high = float(estimate + z * se)
	return IntervalResult(
		estimate=float(estimate),
	se=float(se),
	ci_low=ci_low,
	ci_high=ci_high,
		alpha=alpha,
		method="delta",
	).__dict__


def percent_change_ci(
	mean_t: float,
	mean_c: float,
	se_diff: float,
	*,
	alpha: float = 0.05,
	method: Literal["delta", "bootstrap"] = "delta",
	bootstrap_iters: int = 2000,
	seed: int | None = 0,
) -> dict[str, float]:
	"""
	Confidence interval for percent change (lift expressed as percentage).
	Parameters
	----------
	mean_t, mean_c:
		Treatment and control sample means. ``mean_c`` must be positive.
	se_diff:
		Standard error of the treated - control mean difference.
	method:
		``delta`` uses the delta method. ``bootstrap`` samples differences
		from a normal approximation defined by ``se_diff``.
	ensure_probability(alpha, "alpha")
	ensure_positive(mean_c, "mean_c")
	ensure_non_negative(se_diff, "se_diff")

	estimate = (mean_t - mean_c) / mean_c

	if method == "delta":
		if se_diff == 0.0:
			raise ValidationError("se_diff must be positive for delta method.")
		se = se_diff / abs(mean_c)
	z = _z_score(alpha)
	ci_low = float(estimate - z * se)
	ci_high = float(estimate + z * se)
		return IntervalResult(
			estimate=float(estimate),
			se=float(se),
			ci_low=ci_low,
			ci_high=ci_high,
			alpha=alpha,
			method="delta",
		).__dict__

	if method == "bootstrap":
		if se_diff == 0.0:
			raise ValidationError("se_diff must be positive for bootstrap method.")
	ensure_positive_int(bootstrap_iters, "bootstrap_iters")
	rng = np.random.default_rng(seed)
	diffs = rng.normal(loc=mean_t - mean_c, scale=se_diff, size=bootstrap_iters)
	pct_samples = diffs / mean_c
	lower = float(np.quantile(pct_samples, alpha / 2))
	upper = float(np.quantile(pct_samples, 1 - alpha / 2))
		se = float(np.std(pct_samples, ddof=1))
		return IntervalResult(
			estimate=float(estimate),
			se=se,
			ci_low=lower,
			ci_high=upper,
			alpha=alpha,
			method="bootstrap",
		).__dict__

	raise ValidationError("method must be one of {'delta', 'bootstrap'}.")
"""Helpers for log-lift and percent change confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..core.errors import ValidationError
from ..core.validate import (ensure_non_negative, ensure_positive,
                             ensure_positive_int, ensure_probability)

__all__ = ["log_lift_ci", "percent_change_ci"]


@dataclass(frozen=True)
class IntervalResult:
	estimate: float
	se: float
	ci_low: float
	ci_high: float
	alpha: float
	method: str


def _z_score(alpha: float) -> float:
	from scipy.stats import norm

	return float(norm.ppf(1 - alpha / 2))


def _validate_common_means(mean_t: float, mean_c: float) -> None:
	if mean_t <= 0 or mean_c <= 0:
		raise ValidationError("mean_t and mean_c must be strictly positive for log lift.")


def log_lift_ci(
	mean_t: float,
	mean_c: float,
	var_t: float,
	var_c: float,
	n_t: int,
	n_c: int,
	*,
	alpha: float = 0.05,
) -> dict[str, float]:
	"""
	Delta-method confidence interval for log lift (log ratio of means).

	Parameters
	----------
	mean_t, mean_c:
		Treatment and control sample means. Must be positive.
	var_t, var_c:
		Sample variances for treatment and control.
	n_t, n_c:
		Sample sizes.
	"""
	ensure_positive(mean_t, "mean_t")
	ensure_positive(mean_c, "mean_c")
	ensure_non_negative(var_t, "var_t")
	ensure_non_negative(var_c, "var_c")
	ensure_positive_int(n_t, "n_t")
	ensure_positive_int(n_c, "n_c")
	ensure_probability(alpha, "alpha")

	if var_t == 0 or var_c == 0:
		raise ValidationError("Sample variances must be positive for log lift.")

	estimate = np.log(mean_t) - np.log(mean_c)
	se = np.sqrt(var_t / (n_t * mean_t**2) + var_c / (n_c * mean_c**2))
	z = _z_score(alpha)
	ci_low = float(estimate - z * se)
	ci_high = float(estimate + z * se)
	return IntervalResult(
		estimate=float(estimate),
		se=float(se),
		ci_low=ci_low,
		ci_high=ci_high,
		alpha=alpha,
		method="delta",
	).__dict__


def percent_change_ci(
	mean_t: float,
	mean_c: float,
	se_diff: float,
	*,
	alpha: float = 0.05,
	method: Literal["delta", "bootstrap"] = "delta",
	bootstrap_iters: int = 2000,
	seed: int | None = 0,
) -> dict[str, float]:
	"""
	Confidence interval for percent change (lift expressed as percentage).

	Parameters
	----------
	mean_t, mean_c:
		Treatment and control sample means. ``mean_c`` must be positive.
	se_diff:
		Standard error of the treated - control mean difference.
	method:
		``delta`` uses the delta method. ``bootstrap`` samples differences
		from a normal approximation defined by ``se_diff``.
	"""
	ensure_probability(alpha, "alpha")
	ensure_positive(mean_c, "mean_c")
	ensure_non_negative(se_diff, "se_diff")

	estimate = (mean_t - mean_c) / mean_c

	if method == "delta":
		if se_diff == 0.0:
			raise ValidationError("se_diff must be positive for delta method.")
		se = se_diff / abs(mean_c)
		z = _z_score(alpha)
		ci_low = float(estimate - z * se)
		ci_high = float(estimate + z * se)
		return IntervalResult(
			estimate=float(estimate),
			se=float(se),
			ci_low=ci_low,
			ci_high=ci_high,
			alpha=alpha,
			method="delta",
		).__dict__

	if method == "bootstrap":
		if se_diff == 0.0:
			raise ValidationError("se_diff must be positive for bootstrap method.")
		ensure_positive_int(bootstrap_iters, "bootstrap_iters")
		rng = np.random.default_rng(seed)
		diffs = rng.normal(loc=mean_t - mean_c, scale=se_diff, size=bootstrap_iters)
		pct_samples = diffs / mean_c
		lower = float(np.quantile(pct_samples, alpha / 2))
		upper = float(np.quantile(pct_samples, 1 - alpha / 2))
		se = float(np.std(pct_samples, ddof=1))
		return IntervalResult(
			estimate=float(estimate),
			se=se,
			ci_low=lower,
			ci_high=upper,
			alpha=alpha,
			method="bootstrap",
		).__dict__

	raise ValidationError("method must be one of {'delta', 'bootstrap'}.")
