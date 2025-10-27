
"""Simple uplift modeling helpers using optional scikit-learn dependency."""

from __future__ import annotations

from typing import Any, Iterable, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..core.errors import ValidationError
from ..core.validate import assert_numeric, require_columns

__all__ = ["estimate_uplift"]


def _get_regressor() -> Any:
    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
        raise ValidationError(
            "scikit-learn is required for uplift estimation. "
            "Install the 'ml' extra, e.g. `pip install abx-next[ml]`."
        ) from exc
    return RandomForestRegressor(random_state=42, n_estimators=200)


def _prepare_treatment(df: pd.DataFrame, treatment_col: str) -> pd.Series:
    treatment = df[treatment_col]
    if treatment.dtype == bool:
        return treatment.astype(int)
    unique = treatment.unique()
    if set(unique) == {"control", "treatment"}:
        return (treatment == "treatment").astype(int)
    if set(unique) == {0, 1}:
        return treatment.astype(int)
    raise ValidationError(
        "Treatment column must be boolean, contain {0,1}, or {'control','treatment'}."
    )


def estimate_uplift(
    df: pd.DataFrame,
    features: Iterable[str],
    outcome: str,
    *,
    treatment_col: str = "group",
) -> pd.Series:
    """Estimate user-level uplift via a simple T-learner.

    Parameters
    ----------
    df:
        Input DataFrame containing feature columns, outcome, and treatment indicator.
    features:
        Iterable of column names to use as model features.
    outcome:
        Name of the outcome column.
    treatment_col:
        Column indicating treatment status. Accepts boolean, {0,1}, or
        {'control','treatment'}.
    """
    feature_list = list(features)
    if not feature_list:
        raise ValidationError("features must contain at least one column.")

    require_columns(df, [outcome, treatment_col, *feature_list], context="htablearning")

    for col in feature_list:
        assert_numeric(df[col], col)

    treatment = _prepare_treatment(df, treatment_col)
    y = df[outcome]
    assert_numeric(y, outcome)

    treated_mask = treatment == 1
    control_mask = treatment == 0
    if treated_mask.sum() == 0 or control_mask.sum() == 0:
        raise ValidationError("Both treatment and control observations are required.")

    X = df[feature_list].to_numpy(dtype=float)
    reg_t = _get_regressor()
    reg_c = _get_regressor()

    reg_t.fit(X[treated_mask], y[treated_mask])
    reg_c.fit(X[control_mask], y[control_mask])

    pred_t: NDArray[np.float64] = np.asarray(reg_t.predict(X), dtype=float)
    pred_c: NDArray[np.float64] = np.asarray(reg_c.predict(X), dtype=float)
    uplift = pred_t - pred_c
    return pd.Series(uplift, index=df.index, name="uplift")
