from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.validate import assert_numeric
from ..utils.types import ABFrame, CovariateProvider


def _theta_hat(y: pd.Series, x: pd.Series) -> float:
    """Estimate theta = Cov(Y, X) / Var(X) for CUPED."""
    if len(y) != len(x):
        raise ValueError("y and x must have the same length.")
    assert_numeric(y, "y")
    assert_numeric(x, "x")
    vx = float(np.var(x, ddof=1))
    if vx <= 0.0:
        raise ValueError("Variance of covariate X must be positive.")
    cxy = float(np.cov(y, x, ddof=1)[0, 1])
    return cxy / vx


def cuped_adjust(
    ab: ABFrame,
    covariate: pd.Series | None = None,
    cov_provider: CovariateProvider | None = None,
) -> tuple[pd.DataFrame, float]:
    """Apply CUPED: Y* = Y - theta * X."""
    ab.validate()
    df = ab.df.copy()

    if covariate is not None:
        x = pd.Series(covariate)
        if len(x) != len(df):
            raise ValueError("Provided covariate length must match input data length.")
    elif cov_provider is not None:
        x_user = cov_provider.get_covariate(df["user_id"])
        x = df["user_id"].map(x_user.to_dict())
    else:
        raise ValueError("Provide either 'covariate' or 'cov_provider'.")

    if x.isna().any():
        raise ValueError("Covariate contains NaN; fill or drop before CUPED.")

    theta = _theta_hat(df["metric"].astype(float), x.astype(float))
    df["metric_cuped"] = df["metric"].astype(float) - theta * x.astype(float)
    return df, theta
