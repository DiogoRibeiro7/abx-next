from __future__ import annotations

import pandas as pd
from scipy.stats import chisquare


def srm_test(n_control: int, n_treatment: int, p_expected: float = 0.5) -> dict[str, float]:
    """Chi-square SRM test for a 1:1 (or custom) split."""
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("Counts must be positive.")
    if not (0.0 < p_expected < 1.0):
        raise ValueError("p_expected must be in (0,1).")

    n_tot = n_control + n_treatment
    e_c = n_tot * p_expected
    e_t = n_tot * (1.0 - p_expected)

    chi2, p = chisquare([n_control, n_treatment], f_exp=[e_c, e_t])
    return {
        "chi2": float(chi2),
        "pvalue": float(p),
        "expected_control": float(e_c),
        "expected_treatment": float(e_t),
    }


def srm_from_frame(df: pd.DataFrame, p_expected: float = 0.5) -> dict[str, float]:
    """Convenience wrapper reading counts from a DataFrame with a 'group' column."""
    counts = df["group"].value_counts()
    if not {"control", "treatment"}.issubset(counts.index):
        raise ValueError("Both groups {'control','treatment'} must be present.")
    return srm_test(int(counts["control"]), int(counts["treatment"]), p_expected)
