"""Tests for enhanced switchback utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from abx_next.design.switchback import (
    assign_switchback,
    label_events_by_period,
    validate_period,
)


def test_validate_period_accepts_standard_aliases() -> None:
    """Common pandas aliases should validate."""
    validate_period("H")
    validate_period("1D")


def test_validate_period_rejects_invalid_alias() -> None:
    """Invalid aliases should raise ValueError."""
    with pytest.raises(ValueError):
        validate_period("")
    with pytest.raises(ValueError):
        validate_period("not-a-period")


def test_label_events_by_period_basic_assignment() -> None:
    """Events should align to the most recent period_start and preserve order."""
    events = pd.DataFrame({
        "event_id": [101, 102, 103],
        "ts": [
            pd.Timestamp("2024-01-01 01:30:00"),
            pd.Timestamp("2024-01-01 00:15:00"),
            pd.Timestamp("2024-01-01 02:00:00"),
        ],
    })

    period_assign = pd.DataFrame({
        "period_start": pd.date_range("2024-01-01 00:00:00", periods=3, freq="H"),
        "group": ["control", "treatment", "control"],
    })

    labelled = label_events_by_period(events, "ts", period_assign)

    # Order preserved
    assert labelled.index.tolist() == events.index.tolist()

    expected_periods = [
        pd.Timestamp("2024-01-01 01:00:00"),
        pd.Timestamp("2024-01-01 00:00:00"),
        pd.Timestamp("2024-01-01 02:00:00"),
    ]
    assert labelled["period_start"].tolist() == expected_periods
    assert labelled.loc[0, "group"] == "treatment"
    assert labelled.loc[1, "group"] == "control"


def test_label_events_by_period_timezone_aware() -> None:
    """Timezone-aware timestamps should be handled and validated."""
    events = pd.DataFrame({
        "ts": pd.to_datetime(
            ["2024-05-01 00:15", "2024-05-01 00:45", "2024-05-01 01:10"],
            utc=True,
        ),
        "metric": [1.0, 2.0, 3.0],
    })

    period_assign = pd.DataFrame({
        "period_start": pd.date_range("2024-05-01 00:00", periods=3, freq="30T", tz="UTC"),
        "group": ["control", "treatment", "control"],
    })

    labelled = label_events_by_period(events, "ts", period_assign)
    assert labelled["period_start"].dtype == period_assign["period_start"].dtype
    assert labelled["group"].tolist() == ["control", "treatment", "control"]


def test_label_events_by_period_mismatched_timezone_raises() -> None:
    """Mixing naive and tz-aware timestamps should error."""
    events = pd.DataFrame({
        "ts": pd.to_datetime(["2024-06-01 00:00"]),
    })
    period_assign = pd.DataFrame({
        "period_start": pd.date_range("2024-06-01 00:00", periods=1, freq="H", tz="UTC"),
        "group": ["control"],
    })
    with pytest.raises(TypeError):
        label_events_by_period(events, "ts", period_assign)


def test_label_events_handles_empty_inputs() -> None:
    """Empty event frames should round-trip cleanly with proper dtypes."""
    events = pd.DataFrame({"ts": pd.Series(dtype="datetime64[ns]")})
    period_assign = assign_switchback([], period="H")

    labelled = label_events_by_period(events, "ts", period_assign)
    assert labelled.empty
    assert "period_start" in labelled.columns and "group" in labelled.columns
    assert labelled["period_start"].dtype == period_assign["period_start"].dtype


def test_label_events_no_period_assignments() -> None:
    """When no period assignments exist, group labels should be missing."""
    events = pd.DataFrame({
        "ts": pd.to_datetime(["2024-07-01 00:10", "2024-07-01 00:20"]),
    })
    period_assign = assign_switchback([], period="H")
    labelled = label_events_by_period(events, "ts", period_assign)
    assert labelled["group"].isna().all()
    assert labelled["period_start"].isna().all()
