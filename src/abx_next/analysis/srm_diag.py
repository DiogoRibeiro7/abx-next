"""SRM diagnostics that identify likely causes of sample ratio mismatch."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from pandas import CategoricalDtype
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_object_dtype
from scipy.stats import chi2_contingency

from ..core.validate import require_columns
from ..utils.logging import get_logger
from .srm import srm_from_frame

_NA_TOKEN = "<NA>"
_OTHER_TOKEN = "__OTHER__"
log = get_logger("analysis.srm_diag")


def _infer_categorical_features(df: pd.DataFrame, group_col: str) -> list[str]:
    """Infer categorical-like columns suitable for imbalance diagnostics."""
    candidates: list[str] = []
    for column in df.columns:
        if column == group_col:
            continue
        series = df[column]
        dtype = series.dtype
        if (
            is_object_dtype(series)
            or isinstance(dtype, CategoricalDtype)
            or is_bool_dtype(series)
            or dtype == "string"
        ):
            candidates.append(column)
    return candidates


def _prepare_feature_series(series: pd.Series) -> pd.Series:
    """Prepare a feature column for category-level chi-square analysis."""
    obj_series = series.astype("object")
    filled = obj_series.fillna(_NA_TOKEN)
    counts = filled.value_counts(dropna=False)
    top_categories = counts.head(20).index
    trimmed = filled.where(filled.isin(top_categories), other=_OTHER_TOKEN)
    return trimmed


def _validate_inputs(df: pd.DataFrame, group_col: str, features: Iterable[str]) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    require_columns(df, [group_col], context="srm_diagnostics")
    if is_datetime64_any_dtype(df[group_col]):
        raise TypeError("group column must be categorical, not datetime.")
    if features:
        missing = [col for col in features if col not in df.columns]
        if missing:
            raise ValueError(f"Features {missing} not present in DataFrame.")


def srm_diagnostics(
    df: pd.DataFrame,
    group_col: str = "group",
    features: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    Provide SRM diagnostics and per-feature imbalance suspects.

    Parameters
    ----------
    df:
        Input DataFrame containing at least the assignment ``group_col``.
    group_col:
        Column identifying treatment/control assignment.
    features:
        Optional iterable of column names to inspect for imbalances. When
        omitted, the function examines object/string/bool/category columns.

    Returns
    -------
    dict
        Dictionary containing the SRM p-value and a list of suspect feature
        categories contributing to the mismatch.
    """
    if features is not None:
        features_list = list(features)
    else:
        features_list = _infer_categorical_features(df, group_col)
    _validate_inputs(df, group_col, features_list)
    log.debug(
        "Running SRM diagnostics rows=%d features=%s",
        len(df),
        features_list,
    )

    renamed = df.rename(columns={group_col: "group"})
    srm_result = srm_from_frame(renamed)
    srm_p = srm_result["pvalue"]

    suspects: list[dict[str, Any]] = []
    if srm_p >= 0.001 or not features_list:
        log.debug(
            "SRM p-value %.4g >= threshold or no features supplied; skipping suspects.",
            srm_p,
        )
        return {"srm_p": float(srm_p), "suspects": suspects}

    group_counts = renamed["group"].value_counts()
    if not {"control", "treatment"}.issubset(group_counts.index):
        raise ValueError("DataFrame must contain both 'control' and 'treatment' groups.")

    group_order = ["control", "treatment"]
    for feature in features_list:
        feature_series = _prepare_feature_series(df[feature])
        data = pd.DataFrame(
            {
                "group": renamed["group"],
                "feature": feature_series,
            }
        )
        contingency = pd.crosstab(data["group"], data["feature"], dropna=False)
        contingency = contingency.reindex(index=group_order).fillna(0)
        group_totals = contingency.sum(axis=1).to_numpy()

        for category in contingency.columns:
            if category == _OTHER_TOKEN:
                continue

            observed_category = contingency[category].to_numpy(dtype=float)
            total_category = observed_category.sum()
            if total_category == 0.0:
                continue

            observed_not = group_totals - observed_category
            cat_table = np.array(
                [
                    [observed_category[0], observed_not[0]],
                    [observed_category[1], observed_not[1]],
                ],
                dtype=float,
            )
            if cat_table.sum() == 0.0:
                continue

            try:
                _, pvalue, _, expected = chi2_contingency(cat_table, correction=False)
            except ValueError:
                continue

            if np.isnan(pvalue) or pvalue >= 0.05:
                continue

            expected_category = expected[:, 0]
            category_value: Any
            if category == _NA_TOKEN:
                category_value = None
            else:
                category_value = category

            suspects.append(
                {
                    "feature": feature,
                    "category": category_value,
                    "pvalue": float(pvalue),
                    "obs": {
                        group: float(observed_category[idx])
                        for idx, group in enumerate(group_order)
                    },
                    "exp": {
                        group: float(expected_category[idx])
                        for idx, group in enumerate(group_order)
                    },
                }
            )

    suspects.sort(key=lambda item: item["pvalue"])
    if suspects:
        feature_count = len({suspect["feature"] for suspect in suspects})
        log.debug(
            "SRM diagnostics identified %d suspect categories across %d features.",
            len(suspects),
            feature_count,
        )
    else:
        log.debug("SRM diagnostics found no suspect categories despite SRM p-value %.4g.", srm_p)

    return {"srm_p": float(srm_p), "suspects": suspects}
