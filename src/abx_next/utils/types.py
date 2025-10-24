from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

Group = Literal["control", "treatment"]


@dataclass(frozen=True)
class ABFrame:
    """
    Typed wrapper for an A/B dataset.

    Required columns:
      - group: {"control","treatment"}
      - metric: numeric float-like
      - user_id: hashable id
      - exposed: bool (True if the user had a chance to be affected)
    """
    df: pd.DataFrame

    def validate(self) -> None:
        required = {"group", "metric", "user_id", "exposed"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"ABFrame missing columns: {sorted(missing)}")
        if not set(self.df["group"].unique()).issubset({"control", "treatment"}):
            raise ValueError("Column 'group' must contain only {'control','treatment'}.")
        if not is_numeric_dtype(self.df["metric"]):
            raise TypeError("Column 'metric' must be numeric.")
        if not is_bool_dtype(self.df["exposed"]):
            # strict typing to avoid silent truthiness bugs
            raise TypeError("Column 'exposed' must be boolean.")


@runtime_checkable
class CovariateProvider(Protocol):
    """Protocol for providing a per-user covariate used by CUPED/CUPAC."""
    def get_covariate(self, user_ids: pd.Series) -> pd.Series: ...
