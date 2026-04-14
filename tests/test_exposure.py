"""Tests for exposure definition and sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from experimetrics.analysis import define_exposure, triggered_sensitivity
from experimetrics.core.errors import ValidationError


def _build_events() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    users = np.repeat(np.arange(200), 3)
    flags = rng.random(len(users)) < 0.3
    times = pd.date_range("2024-01-01", periods=len(users), freq="H")
    return pd.DataFrame(
        {
            "user_id": users,
            "ts": times,
            "saw_variant": flags,
        }
    )


def test_define_exposure_basic() -> None:
    events = _build_events()
    exposure = define_exposure(
        events,
        user_col="user_id",
        ts_col="ts",
        rule={"flag_col": "saw_variant", "flag_value": True},
    )
    assert exposure.dtype == bool
    assert exposure.any()
    assert exposure.index.is_unique


def _build_metric_data(exposure_series: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    df = pd.DataFrame({"user_id": exposure_series.index})
    df["group"] = np.where(df["user_id"] % 2 == 0, "control", "treatment")
    df["metric"] = rng.normal(0, 1, size=len(df)) + df["group"].map(
        {"control": 0.0, "treatment": 0.5}
    )
    df["exposed_strict"] = exposure_series.values
    df["exposed_relaxed"] = df["exposed_strict"] | (rng.random(len(df)) < 0.1)
    return df


def test_triggered_sensitivity_runs() -> None:
    events = _build_events()
    exposure = define_exposure(
        events,
        user_col="user_id",
        ts_col="ts",
        rule={"flag_col": "saw_variant", "flag_value": True},
    )
    df = _build_metric_data(exposure)
    result = triggered_sensitivity(
        df.merge(exposure.rename("exposed"), left_on="user_id", right_index=True),
        exposure_cols=["exposed", "exposed_relaxed"],
        metric="metric",
    )
    assert set(result.columns) >= {"exposure_col", "diff", "se", "ci_low", "ci_high", "warning"}
    assert len(result) == 2


def test_define_exposure_invalid_rule() -> None:
    events = _build_events()
    with pytest.raises(ValidationError):
        define_exposure(events, "user_id", "ts", rule={"window": [1]})
