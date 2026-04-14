from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t


def welch_diff_ci(
    x_c: pd.Series, x_t: pd.Series, alpha: float = 0.05
) -> dict[str, float]:
    """Welch-style CI for mean difference (unequal variances)."""
    xc = x_c.astype(float).to_numpy()
    xt = x_t.astype(float).to_numpy()
    n_c, n_t = len(xc), len(xt)
    if n_c < 2 or n_t < 2:
        raise ValueError("Need at least 2 observations per group.")
    vc, vt = np.var(xc, ddof=1), np.var(xt, ddof=1)
    se = np.sqrt(vc / n_c + vt / n_t)
    num = (vc / n_c + vt / n_t) ** 2
    den = (vc**2 / (n_c**2 * (n_c - 1))) + (vt**2 / (n_t**2 * (n_t - 1)))
    df = num / den
    q = float(t.ppf(1 - alpha / 2, df))
    diff = float(xt.mean() - xc.mean())
    return {
        "diff": diff,
        "se": float(se),
        "df": float(df),
        "ci_low": diff - q * se,
        "ci_high": diff + q * se,
    }
