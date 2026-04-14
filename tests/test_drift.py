"""Tests for distribution drift detection utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from experimetrics.analysis.drift import drift_report, ks_drift, psi_drift


def test_ks_drift_no_drift():
    """Test KS drift when distributions are the same."""
    rng = np.random.default_rng(123)
    x_pre = pd.Series(rng.normal(0, 1, 1000))
    x_in = pd.Series(rng.normal(0, 1, 1000))

    result = ks_drift(x_pre, x_in, alpha=0.05)

    assert "statistic" in result
    assert "p_value" in result
    assert "drift_detected" in result
    assert "alpha" in result
    assert "message" in result
    assert result["alpha"] == 0.05
    assert not result["drift_detected"]  # Should not detect drift


def test_ks_drift_with_drift():
    """Test KS drift when distributions are different."""
    rng = np.random.default_rng(123)
    x_pre = pd.Series(rng.normal(0, 1, 1000))
    x_in = pd.Series(rng.normal(2, 1, 1000))  # Different mean

    result = ks_drift(x_pre, x_in, alpha=0.05)

    assert result["drift_detected"]  # Should detect drift
    assert result["p_value"] < 0.05
    assert result["statistic"] > 0


def test_ks_drift_insufficient_data():
    """Test KS drift with insufficient data."""
    x_pre = pd.Series([1.0])
    x_in = pd.Series([2.0])

    with pytest.raises(ValueError, match="Need at least 2 observations"):
        ks_drift(x_pre, x_in)


def test_ks_drift_with_missing_values():
    """Test KS drift handles missing values correctly."""
    rng = np.random.default_rng(123)
    x_pre = pd.Series([np.nan, 1, 2, 3, np.nan, 4, 5])
    x_in = pd.Series([1, np.nan, 2, 3, 4, 5, np.nan])

    result = ks_drift(x_pre, x_in, alpha=0.05)

    assert not result["drift_detected"]  # Similar distributions after dropping NaN


def test_psi_drift_no_drift():
    """Test PSI when distributions are identical."""
    bins_pre = pd.Series([10, 20, 30, 40])
    bins_in = pd.Series([10, 20, 30, 40])

    psi = psi_drift(bins_pre, bins_in)

    assert abs(psi) < 1e-6  # Should be near zero


def test_psi_drift_with_drift():
    """Test PSI when distributions differ."""
    bins_pre = pd.Series([40, 30, 20, 10])
    bins_in = pd.Series([10, 20, 30, 40])  # Reversed distribution

    psi = psi_drift(bins_pre, bins_in)

    assert psi > 0.1  # Should indicate significant drift


def test_psi_drift_proportions():
    """Test PSI with proportion inputs."""
    bins_pre = pd.Series([0.4, 0.3, 0.2, 0.1])  # Already proportions
    bins_in = pd.Series([0.1, 0.2, 0.3, 0.4])

    psi = psi_drift(bins_pre, bins_in)

    assert psi > 0.1


def test_drift_report_comprehensive():
    """Test comprehensive drift report functionality."""
    rng = np.random.default_rng(42)

    # Create synthetic data with known drift patterns
    n_pre, n_in = 500, 500
    timestamps_pre = pd.date_range("2023-01-01", periods=n_pre, freq="1H")
    timestamps_in = pd.date_range("2023-01-22", periods=n_in, freq="1H")

    df = pd.DataFrame(
        {
            "timestamp": list(timestamps_pre) + list(timestamps_in),
            "no_drift": list(rng.normal(0, 1, n_pre)) + list(rng.normal(0, 1, n_in)),
            "drift_feature": list(rng.normal(0, 1, n_pre)) + list(rng.normal(2, 1, n_in)),
            "constant_feature": [5.0] * (n_pre + n_in),
        }
    )

    pre_end_ts = pd.Timestamp("2023-01-21")
    features = ["no_drift", "drift_feature", "constant_feature"]

    result = drift_report(df, features, "timestamp", pre_end_ts)

    assert len(result) == 3
    assert set(result.columns) == {
        "feature",
        "ks_statistic",
        "ks_p_value",
        "ks_drift_detected",
        "psi",
        "drift_severity",
        "message",
    }

    # Check no_drift feature
    no_drift_row = result[result["feature"] == "no_drift"].iloc[0]
    assert not no_drift_row["ks_drift_detected"]
    assert no_drift_row["drift_severity"] == "none"

    # Check drift_feature
    drift_row = result[result["feature"] == "drift_feature"].iloc[0]
    assert drift_row["ks_drift_detected"]
    assert drift_row["drift_severity"] in ["low", "medium", "high", "detected"]

    # Check constant_feature
    constant_row = result[result["feature"] == "constant_feature"].iloc[0]
    assert not constant_row["ks_drift_detected"]


def test_drift_report_insufficient_data():
    """Test drift report with insufficient data."""
    df = pd.DataFrame(
        {
            "timestamp": ["2023-01-01", "2023-01-02"],
            "feature": [1.0, 2.0],
        }
    )

    pre_end_ts = pd.Timestamp("2023-01-01 12:00:00")

    result = drift_report(df, ["feature"], "timestamp", pre_end_ts)

    assert len(result) == 1
    assert result.iloc[0]["drift_severity"] == "insufficient_data"


def test_drift_report_with_missing_values():
    """Test drift report handles missing values gracefully."""
    rng = np.random.default_rng(123)

    n_pre, n_in = 100, 100
    timestamps_pre = pd.date_range("2023-01-01", periods=n_pre, freq="1H")
    timestamps_in = pd.date_range("2023-01-05", periods=n_in, freq="1H")

    # Add missing values
    feature_data = list(rng.normal(0, 1, n_pre)) + list(rng.normal(0, 1, n_in))
    feature_data[10:15] = [np.nan] * 5  # Add some NaN values

    df = pd.DataFrame(
        {
            "timestamp": list(timestamps_pre) + list(timestamps_in),
            "feature_with_nan": feature_data,
        }
    )

    pre_end_ts = pd.Timestamp("2023-01-04")

    result = drift_report(df, ["feature_with_nan"], "timestamp", pre_end_ts)

    assert len(result) == 1
    assert not pd.isna(result.iloc[0]["ks_statistic"])  # Should handle NaN properly


def test_synthetic_drift_detection():
    """Test that synthetic drift is properly detected with clear thresholds."""
    rng = np.random.default_rng(999)

    # Create clear drift case: location shift
    x_pre = pd.Series(rng.normal(0, 1, 1000))
    x_in = pd.Series(rng.normal(3, 1, 1000))  # Large location shift

    result = ks_drift(x_pre, x_in, alpha=0.01)

    # Should definitively detect drift with strong signal
    assert result["drift_detected"]
    assert result["p_value"] < 1e-10  # Very strong evidence
    assert result["statistic"] > 0.5  # Large test statistic

    # Test PSI as well
    bins = np.linspace(-4, 6, 11)
    bins_pre = pd.cut(x_pre, bins=bins).value_counts(sort=False)
    bins_in = pd.cut(x_in, bins=bins).value_counts(sort=False)

    psi = psi_drift(bins_pre, bins_in)
    assert psi > 0.25  # High drift threshold
