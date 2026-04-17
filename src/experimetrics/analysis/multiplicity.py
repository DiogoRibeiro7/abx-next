"""P-value adjustment utilities for multiple testing correction."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from ..core.errors import ValidationError
from ..core.validate import assert_numeric, ensure_probability

__all__ = ["adjust_pvalues", "familywise_report"]


SUPPORTED_METHODS = {"bonferroni", "holm", "bh", "by"}


def _validate_pvalues(pvals: pd.Series) -> pd.Series:
	if not isinstance(pvals, pd.Series):
		raise ValidationError("pvals must be a pandas Series.")
	assert_numeric(pvals, "pvals")
	if pvals.isna().any():
		raise ValidationError("pvals must not contain NaN values.")
	if ((pvals < 0) | (pvals > 1)).any():
		raise ValidationError("pvals must lie within [0, 1].")
	return pvals.astype(float)


def adjust_pvalues(pvals: pd.Series, method: str = "bh") -> pd.DataFrame:
	"""
	Adjust p-values for multiple testing.

	Parameters
	----------
	pvals:
		Series of raw p-values indexed by hypothesis identifier.
	method:
		One of ``bonferroni``, ``holm``, ``bh`` (Benjamini-Hochberg), or
		``by`` (Benjamini-Yekutieli).
	"""
	method_lower = method.lower()
	if method_lower not in SUPPORTED_METHODS:
		choices = ", ".join(sorted(SUPPORTED_METHODS))
		raise ValidationError(f"Unsupported method '{method}'. Choose from {choices}.")

	p = _validate_pvalues(pvals)
	n = len(p)
	p_array = p.to_numpy(dtype=float)
	if n == 0:
		raise ValidationError("pvals must contain at least one entry.")
	order = np.argsort(p_array)
	ordered_p = p_array[order]
	adjusted = np.empty_like(ordered_p, dtype=float)

	if method_lower == "bonferroni":
		adjusted = np.clip(ordered_p * n, 0, 1)
	elif method_lower == "holm":
		adjusted = np.minimum.accumulate((n - np.arange(n)) * ordered_p[::-1])[::-1]
		adjusted = np.clip(adjusted, 0, 1)
	elif method_lower in {"bh", "by"}:
		c_m = 1.0
		if method_lower == "by":
			harmonic_numbers = np.sum(1.0 / np.arange(1, n + 1))
			c_m = harmonic_numbers
		raw = ordered_p * n / (np.arange(1, n + 1) * c_m)
		adjusted = np.minimum.accumulate(raw[::-1])[::-1]
		adjusted = np.clip(adjusted, 0, 1)
	else:
		raise AssertionError("Unhandled adjustment method.")

	adjusted_series = pd.Series(adjusted, index=p.index[order], name="p_adj")
	adjusted_series = adjusted_series.loc[p.index]  # reorder to original index
	return pd.DataFrame({"pvalue": p, "p_adj": adjusted_series})


def familywise_report(
	df_tests: pd.DataFrame,
	*,
	p_col: str = "pvalue",
	method: str = "bh",
	alpha: float = 0.05,
) -> pd.DataFrame:
	"""
	Apply a correction to a DataFrame of tests and flag significant outcomes.

	Returns a copy of the input with an additional ``p_adj`` column and a
	boolean ``reject`` column indicating which hypotheses pass the corrected
	threshold.
	"""
	ensure_probability(alpha, "alpha", inclusive=True)
	if not isinstance(df_tests, pd.DataFrame):
		raise ValidationError("df_tests must be a pandas DataFrame.")
	if p_col not in df_tests.columns:
		raise ValidationError(f"Column '{p_col}' not found in df_tests.")

	adjusted = adjust_pvalues(df_tests[p_col], method=method)
	report = df_tests.copy()
	report["p_adj"] = adjusted["p_adj"]
	report["reject"] = report["p_adj"] <= alpha
	return report
