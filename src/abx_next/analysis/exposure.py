"""Helpers for defining exposure and running sensitivity analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..core.errors import ValidationError
from ..core.validate import assert_bool, assert_numeric, ensure_probability, require_columns
from .triggered import diff_in_means

__all__ = ["define_exposure", "triggered_sensitivity"]


@dataclass(frozen=True)
class ExposureRule:
    flag_col: str | None = None
    flag_value: Any = True
    window: tuple[pd.Timestamp | None, pd.Timestamp | None] | None = None


def _parse_rule(rule: dict[str, Any]) -> ExposureRule:
    if not isinstance(rule, dict):
        raise ValidationError("rule must be a dictionary.")
    flag_col = rule.get("flag_col")
    window = None
    if "window" in rule:
        window_spec = rule["window"]
        if not isinstance(window_spec, (list, tuple)) or len(window_spec) != 2:
            raise ValidationError("window must be a two-element list or tuple.")
        start, end = window_spec
        window = (
            pd.to_datetime(start) if start is not None else None,
            pd.to_datetime(end) if end is not None else None,
        )
    return ExposureRule(flag_col=flag_col, flag_value=rule.get("flag_value", True), window=window)


def define_exposure(
    events_df: pd.DataFrame,
    user_col: str,
    ts_col: str,
    rule: dict[str, Any],
) -> pd.Series:
    """
    Define exposure based on user-level events.

    Parameters
    ----------
    events_df:
        Event-level DataFrame containing at least ``user_col`` and ``ts_col``.
    user_col, ts_col:
        Names of the user identifier and timestamp columns.
    rule:
        Dictionary describing the exposure rule. Supported keys:

        ``flag_col``: column indicating when the exposure condition is met.
        ``flag_value``: value that marks exposure (default True).
        ``window``: optional (start, end) timestamps to restrict events.
    """
    require_columns(events_df, [user_col, ts_col], context="events_df")
    rule_parsed = _parse_rule(rule)

    df = events_df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    if rule_parsed.window:
        start, end = rule_parsed.window
        if start is not None:
            df = df[df[ts_col] >= start]
        if end is not None:
            df = df[df[ts_col] <= end]

    if rule_parsed.flag_col:
        require_columns(df, [rule_parsed.flag_col], context="events_df")
        df = df[df[rule_parsed.flag_col] == rule_parsed.flag_value]

    exposure = df.groupby(user_col).size() > 0
    return exposure.reindex(events_df[user_col].unique(), fill_value=False)


def _heuristic_exposure_outcome_correlation(
    df: pd.DataFrame,
    exposure_col: str,
    metric: str,
) -> bool:
    """Detect whether exposure might depend on the outcome by checking correlation."""
    if exposure_col not in df.columns or metric not in df.columns:
        return False
    exposure = df[exposure_col]
    metric_series = df[metric]
    if exposure.dtype == bool and np.all(exposure == exposure.iloc[0]):
        return False
    try:
        assert_bool(exposure, exposure_col)
        assert_numeric(metric_series, metric)
    except ValidationError:
        return False
    corr = metric_series.corr(exposure.astype(float))
    return bool(abs(corr) > 0.4)


def _z_score(alpha: float) -> float:
    return float(norm.ppf(1 - alpha / 2))


def triggered_sensitivity(
    df: pd.DataFrame,
    exposure_cols: Iterable[str],
    metric: str = "metric",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run sensitivity analysis across multiple exposure definitions.

    Returns
    -------
    DataFrame with diff-in-means estimates, standard errors, confidence
    intervals, and warnings for each exposure column.
    """
    ensure_probability(alpha, "alpha")
    results = []
    exposure_list = list(exposure_cols)
    z_score = _z_score(alpha)
    if not exposure_list:
        raise ValidationError("exposure_cols must contain at least one column.")

    for col in exposure_list:
        if col not in df.columns:
            raise ValidationError(f"Exposure column '{col}' not found in DataFrame.")
        if df[col].isna().any():
            raise ValidationError(f"Exposure column '{col}' contains NaN values.")
        assert_bool(df[col], col)

        filtered = df[df[col]]
        if filtered.empty:
            raise ValidationError(f"Exposure column '{col}' yields no triggered users.")

        triggered_df = df.copy()
        triggered_df = triggered_df[["group", metric, col]].copy()
        triggered_df = triggered_df[triggered_df[col]]
        triggered_df.drop(columns=[col], inplace=True)

        summary = diff_in_means(triggered_df)
        warn = _heuristic_exposure_outcome_correlation(df, col, metric)
        diff = summary["diff"]
        se = summary["se"]
        ci_low = float(diff - z_score * se)
        ci_high = float(diff + z_score * se)

        summary_row = {
            "exposure_col": col,
            **summary,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "alpha": alpha,
            "warning": "Possible exposure-outcome correlation" if warn else "",
        }
        results.append(summary_row)

    return pd.DataFrame(results)
