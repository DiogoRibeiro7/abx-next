"""Filtering utilities for experiment results."""

import pandas as pd
from ..core.errors import ValidationError

def filter_by_significance(df: pd.DataFrame, alpha: float = 0.05, p_col: str = "pvalue") -> pd.DataFrame:
    """
    Return rows where the p-value is less than or equal to alpha.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experiment results with a p-value column.
    alpha : float, default=0.05
        Significance threshold.
    p_col : str, default="pvalue"
        Name of the p-value column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only significant results.
    """
    if p_col not in df.columns:
        raise ValidationError(f"Column '{p_col}' not found in DataFrame.")
    if not (0 < alpha < 1):
        raise ValidationError("alpha must be between 0 and 1.")
    return df[df[p_col] <= alpha].copy()
