from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import pandas as pd

from ..core.validate import assert_bool, assert_in_set, assert_numeric, require_columns

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
        require_columns(self.df, ["group", "metric", "user_id", "exposed"], context="ABFrame")
        assert_in_set(self.df["group"], ["control", "treatment"], "group")
        assert_numeric(self.df["metric"], "metric")
        # strict typing to avoid silent truthiness bugs
        assert_bool(self.df["exposed"], "exposed")


@runtime_checkable
class CovariateProvider(Protocol):
    """Protocol for providing a per-user covariate used by CUPED/CUPAC."""
    def get_covariate(self, user_ids: pd.Series) -> pd.Series: ...
