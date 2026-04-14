"""Synthetic control helper for geo experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, cast

import numpy as np
import pandas as pd

from ..core.errors import ValidationError
from ..core.validate import (
    assert_numeric,
    ensure_probability,
    require_columns,
)

__all__ = ["fit_scm", "scm_summary"]


def _project_to_simplex(weights: np.ndarray) -> np.ndarray:
    """Project an array onto the probability simplex."""
    if weights.ndim != 1:
        raise ValueError("weights must be one dimensional.")
    sorted_w = np.sort(weights)[::-1]
    cumulative = np.cumsum(sorted_w)
    rho_candidates = (sorted_w + (1.0 - cumulative) / (np.arange(len(weights)) + 1)) > 0
    if not np.any(rho_candidates):
        theta = (cumulative[-1] - 1.0) / len(weights)
    else:
        rho = np.nonzero(rho_candidates)[0][-1]
        theta = (cumulative[rho] - 1.0) / (rho + 1)
    projected = np.maximum(weights - theta, 0.0)
    total = projected.sum()
    if total == 0.0:
        return np.full_like(projected, 1.0 / len(projected), dtype=float)
    result = projected / total
    return cast(np.ndarray, result.astype(float, copy=False))


@dataclass(frozen=True)
class SCMResult:
    weights: dict[str, float]
    counterfactual: pd.Series
    effect: pd.Series
    summary: dict[str, float]


def _prepare_panel(df: pd.DataFrame, units: list[str], metric: str, context: str) -> pd.DataFrame:
    """Return a wide panel indexed by time."""
    require_columns(df, ["time", "region", metric], context=context)
    filtered = df[df["region"].isin(units)].copy()
    if filtered.empty:
        raise ValidationError(f"{context} has no data for requested regions {units}.")
    assert_numeric(filtered[metric], metric)
    pivoted = (
        filtered.pivot(index="time", columns="region", values=metric)
        .sort_index()
        .reindex(columns=units)
    )
    if pivoted.isna().any().any():
        raise ValidationError(f"{context} contains missing values for required regions.")
    return pivoted


def fit_scm(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    *,
    treated: str,
    donors: Iterable[str],
    metric: str,
) -> SCMResult:
    """
    Fit a lightweight synthetic control model using pre-period data.

    The function returns donor weights, a counterfactual series for the treated
    unit in the post period, the resulting effect series, and summary statistics.
    """
    donor_list = list(dict.fromkeys(donors))
    if not donor_list:
        raise ValidationError("donors must contain at least one region.")
    if treated in donor_list:
        raise ValidationError("treated unit cannot appear in donors.")

    units = [treated] + donor_list
    pre_panel = _prepare_panel(pre_df, units, metric, context="pre_df")
    post_panel = _prepare_panel(post_df, units, metric, context="post_df")

    missing_post_units = [u for u in units if u not in post_panel.columns]
    if missing_post_units:
        raise ValidationError(f"post_df missing regions {missing_post_units}.")

    y_pre = pre_panel[treated].to_numpy(dtype=float)
    x_pre = pre_panel[donor_list].to_numpy(dtype=float)
    if x_pre.shape[0] < x_pre.shape[1]:
        raise ValidationError("Number of pre-period observations must exceed number of donors.")

    raw_weights, *_ = np.linalg.lstsq(x_pre, y_pre, rcond=None)
    weights = _project_to_simplex(raw_weights)

    x_post = post_panel[donor_list].to_numpy(dtype=float)
    y_post = post_panel[treated].to_numpy(dtype=float)

    counterfactual_values = x_post @ weights
    counterfactual = pd.Series(counterfactual_values, index=post_panel.index, name="counterfactual")
    effect_series = pd.Series(y_post, index=post_panel.index, name="effect") - counterfactual

    summary = scm_summary(effect_series)
    return SCMResult(
        weights={donor: float(weight) for donor, weight in zip(donor_list, weights)},
        counterfactual=counterfactual,
        effect=effect_series,
        summary=summary,
    )


def scm_summary(effect_series: pd.Series, alpha: float = 0.05) -> dict[str, float]:
    """Return bootstrap-based summary statistics for the SCM effect series."""
    if not isinstance(effect_series, pd.Series):
        raise ValidationError("effect_series must be a pandas Series.")
    assert_numeric(effect_series, "effect_series")
    if len(effect_series) == 0:
        raise ValidationError("effect_series must contain at least one observation.")
    ensure_probability(alpha, "alpha")

    values = effect_series.to_numpy(dtype=float)
    n = len(values)
    block_size = max(1, int(np.sqrt(n)))
    num_bootstrap = 1000
    rng = np.random.default_rng(0)

    bootstrap_means = np.empty(num_bootstrap, dtype=float)
    for i in range(num_bootstrap):
        sample: list[float] = []
        while len(sample) < n:
            start = rng.integers(0, n - block_size + 1)
            sample.extend(values[start : start + block_size])
        bootstrap_means[i] = np.mean(sample[:n])

    lower = float(np.quantile(bootstrap_means, alpha / 2))
    upper = float(np.quantile(bootstrap_means, 1 - alpha / 2))
    return {
        "mean": float(values.mean()),
        "ci_low": lower,
        "ci_high": upper,
        "alpha": alpha,
    }

