from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def ks_drift(x_pre: pd.Series, x_in: pd.Series, alpha: float = 0.01) -> dict:
    """Kolmogorov-Smirnov test for drift between pre and in-experiment samples."""
    ks_stat, p_value = ks_2samp(x_pre.dropna(), x_in.dropna())
    drift_detected = p_value < alpha
    return {
        "statistic": ks_stat,
        "p_value": p_value,
        "drift_detected": drift_detected,
        "alpha": alpha,
        "message": f"{'Drift detected' if drift_detected else 'No drift detected'} (p={p_value:.4f}, α={alpha})"
    }

def psi_drift(bins_pre: pd.Series, bins_in: pd.Series) -> float:
    """Population Stability Index (PSI) for drift between two binned samples."""
    p = bins_pre / bins_pre.sum()
    q = bins_in / bins_in.sum()
    psi = np.sum((p - q) * np.log((p + 1e-8) / (q + 1e-8)))
    return float(psi)

def drift_report(
    df: pd.DataFrame,
    features: list[str],
    ts_col: str,
    pre_end_ts: pd.Timestamp
) -> pd.DataFrame:
    """Generate comprehensive drift report for multiple features."""
    results = []
    df[ts_col] = pd.to_datetime(df[ts_col])
    pre_mask = df[ts_col] <= pre_end_ts
    for feature in features:
        try:
            x_pre = df.loc[pre_mask, feature]
            x_in = df.loc[~pre_mask, feature]
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
            ks_result = ks_drift(x_pre, x_in, alpha=0.01)
            try:
                bins = np.quantile(x_pre.dropna(), np.linspace(0, 1, 11))
                bins[0] = -np.inf
                bins[-1] = np.inf
                bins_pre_counts = pd.cut(x_pre.dropna(), bins=bins, duplicates='drop').value_counts(sort=False)
                bins_in_counts = pd.cut(x_in.dropna(), bins=bins, duplicates='drop').value_counts(sort=False)
                all_bins = bins_pre_counts.index.union(bins_in_counts.index)
                bins_pre_aligned = bins_pre_counts.reindex(all_bins, fill_value=0)
                bins_in_aligned = bins_in_counts.reindex(all_bins, fill_value=0)
                psi_value = psi_drift(bins_pre_aligned, bins_in_aligned)
            except Exception:
                psi_value = np.nan
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
