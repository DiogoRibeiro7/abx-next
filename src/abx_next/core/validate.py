"""Reusable validation helpers built on custom error types."""

from __future__ import annotations

import numbers
from typing import Iterable, Sequence

import pandas as pd
from pandas import Series
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .errors import ValidationError

__all__ = [
    "require_columns",
    "assert_numeric",
    "assert_bool",
    "assert_in_set",
    "ensure_positive",
    "ensure_non_negative",
    "ensure_positive_int",
    "ensure_probability",
]


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    context: str = "DataFrame",
) -> None:
    """Ensure ``df`` contains all ``columns``."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValidationError(f"{context} missing required columns: {sorted(missing)}")


def assert_numeric(series: Series, name: str) -> None:
    """Validate that a Series has a numeric dtype."""
    if not is_numeric_dtype(series):
        raise ValidationError(f"Column '{name}' must be numeric.")


def assert_bool(series: Series, name: str) -> None:
    """Validate that a Series has a boolean dtype."""
    if not is_bool_dtype(series):
        raise ValidationError(f"Column '{name}' must be boolean.")


def assert_in_set(series: Series, allowed: Sequence[object], name: str) -> None:
    """Ensure a Series only contains values from a reference set."""
    allowed_set = set(allowed)
    invalid = set(series.unique()) - allowed_set
    if invalid:
        def _string_key(value: object) -> str:
            return str(value)

        allowed_list = sorted(allowed_set, key=_string_key)
        invalid_list = sorted(invalid, key=_string_key)
        raise ValidationError(
            f"Column '{name}' must contain only {allowed_list}; "
            f"found invalid values {invalid_list}."
        )


def ensure_positive(value: float | int, name: str) -> None:
    """Ensure a scalar is strictly positive."""
    if value <= 0:
        raise ValidationError(f"{name} must be positive.")


def ensure_non_negative(value: float | int, name: str) -> None:
    """Ensure a scalar is zero or positive."""
    if value < 0:
        raise ValidationError(f"{name} must be non-negative.")


def ensure_positive_int(value: int, name: str) -> None:
    """Ensure a scalar is a positive integer."""
    if not isinstance(value, numbers.Integral) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer.")


def ensure_probability(value: float, name: str, *, inclusive: bool = False) -> None:
    """Ensure a value lies inside the probability interval."""
    if inclusive:
        valid = 0.0 <= value <= 1.0
        bounds = "[0, 1]"
    else:
        valid = 0.0 < value < 1.0
        bounds = "(0, 1)"
    if not valid:
        raise ValidationError(f"{name} must lie in {bounds}.")
