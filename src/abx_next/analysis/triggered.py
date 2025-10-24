from __future__ import annotations

import numpy as np
import pandas as pd
from ..utils.types import ABFrame


def filter_exposed(ab: ABFrame) -> pd.DataFrame:
    """Keep rows where 'exposed' is True (users who could be affected)."""
    ab.validate()
    return ab.df.loc[ab.df["exposed"]].copy()


def diff_in_means(df: pd.DataFrame, use_column: str = "metric") -> dict[str, float]:
    """Difference-in-means with pooled-variance SE and z-approximation."""
    if use_column not in df.columns:
        raise ValueError(f"Column '{use_column}' not found.")
    if not {"control", "treatment"}.issubset(set(df["group"].unique())):
        raise ValueError("Both groups {'control','treatment'} must be present.")

    g = df.groupby("group")[use_column]
    mean_c, mean_t = g.mean().loc["control"], g.mean().loc["treatment"]
    var_c, var_t = g.var(ddof=1).loc["control"], g.var(ddof=1).loc["treatment"]
    n_c, n_t = g.count().loc["control"], g.count().loc["treatment"]

    se = float(np.sqrt(var_c / n_c + var_t / n_t))
    z = float((mean_t - mean_c) / se) if se > 0 else float("inf")

    return {
        "n_c": float(n_c),
        "n_t": float(n_t),
        "mean_c": float(mean_c),
        "mean_t": float(mean_t),
        "diff": float(mean_t - mean_c),
        "se": se,
        "z": z,
    }
