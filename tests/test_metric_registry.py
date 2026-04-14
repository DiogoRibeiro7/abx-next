
"""Tests for metric registry and schema validation."""

from __future__ import annotations

import pandas as pd
import pytest

from experimetrics.core.errors import ValidationError
from experimetrics.core.validate import validate_ab_schema
from experimetrics.metrics.registry import _clear_registry, get_metric, list_metrics, register_metric


@pytest.fixture(autouse=True)
def clear_registry() -> None:
    _clear_registry()


def test_register_and_lookup_metric() -> None:
    register_metric(
        "conversion_rate",
        kind="ratio",
        aggregation="mean",
        required_columns=["group", "metric"],
    )
    definition = get_metric("conversion_rate")
    assert definition.name == "conversion_rate"
    assert definition.kind == "ratio"
    assert definition.required_columns == ("group", "metric")
    assert list_metrics()


def test_register_metric_duplicate() -> None:
    register_metric(
        "lift",
        kind="difference",
        aggregation="mean",
        required_columns=["group", "metric"],
    )
    with pytest.raises(ValidationError):
        register_metric(
            "lift",
            kind="difference",
            aggregation="mean",
            required_columns=["group", "metric"],
        )


def test_validate_ab_schema() -> None:
    df = pd.DataFrame(
        {
            "group": ["control", "treatment"],
            "user_id": [1, 2],
            "metric": [0.5, 0.7],
            "exposed": [True, True],
        }
    )
    validate_ab_schema(df)

    df_bad = df.copy()
    df_bad["group"] = ["control", "invalid"]
    with pytest.raises(ValidationError):
        validate_ab_schema(df_bad)
