from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def ks_drift(x_pre: pd.Series, x_in: pd.Series, alpha: float = 0.01) -> dict:
    """Kolmogorov-Smirnov test for distribution drift between pre and in-experiment periods.

    Args:
        x_pre: Pre-experiment period values
        x_in: In-experiment period values
        alpha: Significance level for drift detection

    Returns:
        Dictionary containing test statistic, p-value, and drift status
    """
    x_pre_clean = x_pre.dropna().astype(float).to_numpy()
    x_in_clean = x_in.dropna().astype(float).to_numpy()

    if len(x_pre_clean) < 2 or len(x_in_clean) < 2:
        raise ValueError("Need at least 2 observations per period.")

    statistic, p_value = ks_2samp(x_pre_clean, x_in_clean)
    drift_detected = p_value < alpha

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": drift_detected,
        "alpha": alpha,
        "message": f"{'Drift detected' if drift_detected else 'No drift detected'} (p={p_value:.4f}, α={alpha})"
    }


def psi_drift(bins_pre: pd.Series, bins_in: pd.Series) -> float:
    """Calculate Population Stability Index (PSI) for distribution drift.

    Args:
        bins_pre: Binned counts/proportions from pre-experiment period
        bins_in: Binned counts/proportions from in-experiment period

    Returns:
        PSI value (higher values indicate more drift)
    """
    # Convert to proportions if not already
    prop_pre = bins_pre / bins_pre.sum() if bins_pre.sum() != 1.0 else bins_pre
    prop_in = bins_in / bins_in.sum() if bins_in.sum() != 1.0 else bins_in

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    prop_pre = prop_pre + epsilon
    prop_in = prop_in + epsilon

    # Calculate PSI
    psi = ((prop_in - prop_pre) * np.log(prop_in / prop_pre)).sum()

    return float(psi)


def drift_report(
    df: pd.DataFrame,
    features: list[str],
    ts_col: str,
    pre_end_ts: pd.Timestamp
) -> pd.DataFrame:
    """Generate comprehensive drift report for multiple features.

    Args:
        df: DataFrame with features and timestamp column
        features: List of feature column names to analyze
        ts_col: Name of timestamp column
        pre_end_ts: End timestamp of pre-experiment period

    Returns:
        DataFrame with drift metrics for each feature
    """
    results = []

    # Split data into pre and in-experiment periods
    df[ts_col] = pd.to_datetime(df[ts_col])
    pre_mask = df[ts_col] <= pre_end_ts

    for feature in features:
        try:
            x_pre = df.loc[pre_mask, feature]
            x_in = df.loc[~pre_mask, feature]

            # Skip if insufficient data
            if len(x_pre.dropna()) < 2 or len(x_in.dropna()) < 2:
                results.append({
                    "feature": feature,
                    "ks_statistic": np.nan,
                    "ks_p_value": np.nan,
                    "ks_drift_detected": False,
                    "psi": np.nan,
                    "drift_severity": "insufficient_data",
                    "message": "Insufficient data for drift analysis"
                })
                continue

            # KS test
            ks_result = ks_drift(x_pre, x_in, alpha=0.01)

            # PSI calculation (using 10 quantile bins)
            try:
                # Create bins based on pre-period quantiles
                bins = np.quantile(x_pre.dropna(), np.linspace(0, 1, 11))
                bins[0] = -np.inf  # Ensure all values are captured
                bins[-1] = np.inf

                bins_pre_counts = pd.cut(x_pre.dropna(), bins=bins, duplicates='drop').value_counts(sort=False)
                bins_in_counts = pd.cut(x_in.dropna(), bins=bins, duplicates='drop').value_counts(sort=False)

                # Align indices and fill missing bins with 0
                all_bins = bins_pre_counts.index.union(bins_in_counts.index)
                bins_pre_aligned = bins_pre_counts.reindex(all_bins, fill_value=0)
                bins_in_aligned = bins_in_counts.reindex(all_bins, fill_value=0)

                psi_value = psi_drift(bins_pre_aligned, bins_in_aligned)
            except Exception:
                psi_value = np.nan

            # Determine drift severity
            if ks_result["drift_detected"] and not np.isnan(psi_value):
                if psi_value > 0.25:
                    severity = "high"
                elif psi_value > 0.1:
                    severity = "medium"
                else:
                    severity = "low"
            elif ks_result["drift_detected"]:
                severity = "detected"
            else:
                severity = "none"

            results.append({
                "feature": feature,
                "ks_statistic": ks_result["statistic"],
                "ks_p_value": ks_result["p_value"],
                "ks_drift_detected": ks_result["drift_detected"],
                "psi": psi_value,
                "drift_severity": severity,
                "message": ks_result["message"]
            })

        except Exception as e:
            results.append({
                "feature": feature,
                "ks_statistic": np.nan,
                "ks_p_value": np.nan,
                "ks_drift_detected": False,
                "psi": np.nan,
                "drift_severity": "error",
                "message": f"Error: {str(e)}"
            })

    return pd.DataFrame(results)