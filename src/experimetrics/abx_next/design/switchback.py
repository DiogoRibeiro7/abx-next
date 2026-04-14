from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from pandas.tseries.frequencies import to_offset


def assign_switchback(
    timestamps: Iterable[pd.Timestamp],
    period: str = "D",
    seed: int | None = 17,
) -> pd.DataFrame:
    """Assign alternating period-level arms for switchback designs."""
    validate_period(period)
    ts = pd.to_datetime(pd.Series(list(timestamps))).sort_values()
    if ts.empty:
        return pd.DataFrame(
            {
                "period_start": pd.Series(dtype="datetime64[ns]"),
                "group": pd.Series(dtype="object"),
            }
        )

    blocks = ts.dt.floor(period)
    unique_blocks = blocks.drop_duplicates().reset_index(drop=True)

    rng = np.random.default_rng(seed)
    start_treatment = rng.integers(0, 2) == 1
    arms = np.array(["control", "treatment"])
    seq = np.where(
        np.arange(len(unique_blocks)) % 2 == 0,
        arms[int(start_treatment)],
        arms[1 - int(start_treatment)],
    )

    return pd.DataFrame({"period_start": unique_blocks, "group": seq})


def validate_period(period: str) -> None:
    """Validate that the provided string is a valid pandas offset alias."""
    if not isinstance(period, str) or not period:
        raise ValueError("period must be a non-empty string offset alias.")
    try:
        to_offset(period)
    except (TypeError, ValueError) as exc:  # pragma: no cover - specific branch
        raise ValueError(f"Invalid pandas offset alias: {period}") from exc


def _ensure_datetime_column(df: pd.DataFrame, column: str, df_name: str) -> pd.Series:
    """Ensure the target column is datetime typed."""
    if column not in df.columns:
        raise ValueError(f"{df_name} is missing required column '{column}'.")
    series = df[column]
    if not is_datetime64_any_dtype(series):
        raise TypeError(f"{df_name}.{column} must be datetime64 dtype.")
    return series


def _timezone_info(series: pd.Series) -> tuple[bool, object | None]:
    """Return timezone awareness flag and tzinfo for a datetime series."""
    tz = series.dt.tz
    return tz is not None, tz


def label_events_by_period(
    events: pd.DataFrame,
    ts_col: str,
    period_assign: pd.DataFrame,
) -> pd.DataFrame:
    """
    Label individual events with switchback periods and group assignments.

    Parameters
    ----------
    events:
        DataFrame containing individual event rows.
    ts_col:
        Column within ``events`` holding event timestamps.
    period_assign:
        Output from :func:`assign_switchback` describing period boundaries.
    """
    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame.")
    if not isinstance(period_assign, pd.DataFrame):
        raise TypeError("period_assign must be a pandas DataFrame.")

    events_ts = _ensure_datetime_column(events, ts_col, "events")

    required_period_columns = {"period_start", "group"}
    missing_period_columns = required_period_columns.difference(period_assign.columns)
    if missing_period_columns:
        raise ValueError(f"period_assign is missing columns: {sorted(missing_period_columns)}")

    period_column = period_assign["period_start"]
    if not is_datetime64_any_dtype(period_column):
        raise TypeError("period_assign.period_start must be datetime64 dtype.")
    if period_assign.empty:
        period_dtype = events_ts.dtype if len(events_ts) > 0 else period_column.dtype
    else:
        period_ts = _ensure_datetime_column(period_assign, "period_start", "period_assign")

        events_tz_aware, events_tz = _timezone_info(events_ts)
        assign_tz_aware, assign_tz = _timezone_info(period_ts)
        if events_tz_aware != assign_tz_aware:
            raise TypeError("Event and period timestamps must share timezone awareness.")
        if events_tz_aware and events_tz != assign_tz:
            raise TypeError("Event and period timestamps must share the same timezone.")
        period_dtype = period_ts.dtype

    if events.empty:
        result = events.copy()
        result["period_start"] = pd.Series([], dtype=period_dtype)
        group_dtype = (
            period_assign["group"].dtype
            if "group" in period_assign.columns and not period_assign.empty
            else "object"
        )
        result["group"] = pd.Series([], dtype=group_dtype)
        return result

    if period_assign.empty:
        result = events.copy()
        result["period_start"] = pd.Series(pd.NaT, index=result.index, dtype=period_dtype)
        result["group"] = pd.Series(np.nan, index=result.index, dtype="object")
        return result

    order_col = "__switchback_order__"
    index_col = "__switchback_index__"

    prepared_events = events.copy()
    prepared_events[index_col] = events.index
    prepared_events[order_col] = np.arange(len(prepared_events))

    events_sorted = prepared_events.sort_values(ts_col, kind="stable")
    assign_sorted = period_assign.sort_values("period_start", kind="stable")

    merged = pd.merge_asof(
        events_sorted,
        assign_sorted,
        left_on=ts_col,
        right_on="period_start",
        direction="backward",
        allow_exact_matches=True,
    )

    merged = merged.sort_values(order_col, kind="stable")
    original_index = merged.pop(index_col)
    merged.drop(columns=[order_col], inplace=True)
    merged.index = original_index

    desired_columns = list(events.columns) + ["period_start", "group"]
    return merged[desired_columns]
