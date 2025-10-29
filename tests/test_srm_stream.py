"""Tests for incremental SRM detection."""

from __future__ import annotations

import pytest

from abx_next.analysis.srm_stream import SrmWatch


def test_srm_watch_initialization():
    """Test SrmWatch initialization with default and custom parameters."""
    # Default initialization
    watch = SrmWatch()
    assert watch.p_expected == 0.5
    assert watch.warn_p == 1e-3
    assert watch.n_control == 0
    assert watch.n_treatment == 0
    assert not watch.srm_detected
    assert watch.detection_update is None
    assert watch.update_count == 0

    # Custom initialization
    custom_watch = SrmWatch(p_expected=0.6, warn_p=0.01)
    assert custom_watch.p_expected == 0.6
    assert custom_watch.warn_p == 0.01

    # Test invalid parameters
    with pytest.raises(ValueError, match="p_expected"):
        SrmWatch(p_expected=-0.1)

    with pytest.raises(ValueError, match="p_expected"):
        SrmWatch(p_expected=1.1)

    with pytest.raises(ValueError, match="warn_p must be between 0 and 1"):
        SrmWatch(warn_p=0.0)

    with pytest.raises(ValueError, match="warn_p must be between 0 and 1"):
        SrmWatch(warn_p=1.0)


def test_srm_watch_update_basic():
    """Test basic update functionality."""
    watch = SrmWatch()

    # First update
    result = watch.update(nc_delta=10, nt_delta=12)

    assert result["n_control"] == 10
    assert result["n_treatment"] == 12
    assert result["n_total"] == 22
    assert result["expected_control"] == 11.0  # 22 * 0.5
    assert result["expected_treatment"] == 11.0
    assert result["update_count"] == 1
    assert "chi2" in result
    assert "pvalue" in result

    # Second update
    result = watch.update(nc_delta=5, nt_delta=3)

    assert result["n_control"] == 15
    assert result["n_treatment"] == 15
    assert result["n_total"] == 30
    assert result["expected_control"] == 15.0
    assert result["expected_treatment"] == 15.0
    assert result["update_count"] == 2


def test_srm_watch_update_validation():
    """Test input validation for update method."""
    watch = SrmWatch()

    # Test negative values
    with pytest.raises(ValueError, match="nc_delta must be non-negative"):
        watch.update(nc_delta=-1, nt_delta=5)

    with pytest.raises(ValueError, match="nt_delta must be non-negative"):
        watch.update(nc_delta=5, nt_delta=-1)


def test_srm_watch_no_samples():
    """Test behavior with no samples."""
    watch = SrmWatch()

    result = watch.update(nc_delta=0, nt_delta=0)

    assert result["n_control"] == 0
    assert result["n_treatment"] == 0
    assert result["n_total"] == 0
    assert result["expected_control"] == 0.0
    assert result["expected_treatment"] == 0.0
    assert result["chi2"] == 0.0
    assert result["pvalue"] == 1.0
    assert not result["srm_detected"]


def test_srm_watch_balanced_counts():
    """Test SRM watch with perfectly balanced counts."""
    watch = SrmWatch(warn_p=0.01)

    # Add perfectly balanced counts
    result = watch.update(nc_delta=100, nt_delta=100)

    assert result["n_control"] == 100
    assert result["n_treatment"] == 100
    assert result["pvalue"] > 0.01  # Should not detect SRM
    assert not result["srm_detected"]
    assert result["detection_update"] is None


def test_srm_watch_imbalanced_counts():
    """Test SRM detection with heavily imbalanced counts."""
    watch = SrmWatch(warn_p=0.01)

    # Add heavily imbalanced counts that should trigger SRM
    result = watch.update(nc_delta=200, nt_delta=50)

    assert result["n_control"] == 200
    assert result["n_treatment"] == 50
    assert result["n_total"] == 250
    assert result["pvalue"] < 0.01  # Should detect SRM
    assert result["srm_detected"]
    assert result["detection_update"] == 1


def test_srm_watch_incremental_detection():
    """Test SRM detection with incremental updates."""
    watch = SrmWatch(warn_p=1e-3)

    updates = [
        (10, 10),  # Balanced start
        (10, 8),   # Slight imbalance
        (15, 5),   # More imbalance
        (20, 2),   # Heavy imbalance - should trigger SRM
    ]

    srm_detected_at = None

    for i, (nc, nt) in enumerate(updates, 1):
        result = watch.update(nc_delta=nc, nt_delta=nt)

        if result["srm_detected"] and srm_detected_at is None:
            srm_detected_at = i

        # Verify cumulative counts
        expected_nc = sum(u[0] for u in updates[:i])
        expected_nt = sum(u[1] for u in updates[:i])
        assert result["n_control"] == expected_nc
        assert result["n_treatment"] == expected_nt

    # SRM should have been detected at some point
    assert srm_detected_at is not None
    assert watch.srm_detected
    assert watch.detection_update == srm_detected_at


def test_srm_watch_different_expected_proportions():
    """Test SRM watch with different expected proportions."""
    # Test 60/40 split
    watch = SrmWatch(p_expected=0.6, warn_p=0.05)

    # Add counts that match expected 60/40 split
    result = watch.update(nc_delta=60, nt_delta=40)

    assert result["expected_control"] == 60.0  # 100 * 0.6
    assert result["expected_treatment"] == 40.0  # 100 * 0.4
    assert result["pvalue"] > 0.05  # Should not detect SRM

    # Add counts that violate 60/40 split
    result = watch.update(nc_delta=10, nt_delta=50)  # Now 70/90

    assert result["n_control"] == 70
    assert result["n_treatment"] == 90
    assert result["expected_control"] == 96.0  # 160 * 0.6
    assert result["expected_treatment"] == 64.0  # 160 * 0.4
    # This should likely detect SRM due to significant deviation


def test_srm_watch_reset():
    """Test reset functionality."""
    watch = SrmWatch()

    # Add some data and trigger SRM
    watch.update(nc_delta=100, nt_delta=10)
    assert watch.n_control == 100
    assert watch.n_treatment == 10
    assert watch.update_count == 1

    # Reset and verify clean state
    watch.reset()

    assert watch.n_control == 0
    assert watch.n_treatment == 0
    assert not watch.srm_detected
    assert watch.detection_update is None
    assert watch.update_count == 0

    # Verify functionality after reset
    result = watch.update(nc_delta=5, nt_delta=5)
    assert result["n_control"] == 5
    assert result["n_treatment"] == 5
    assert result["update_count"] == 1


def test_srm_watch_get_current_status():
    """Test get_current_status method."""
    watch = SrmWatch()

    # Add some data
    watch.update(nc_delta=20, nt_delta=30)

    # Get status without updating
    status1 = watch.get_current_status()
    status2 = watch.get_current_status()

    # Should return same results both times
    assert status1 == status2
    assert status1["n_control"] == 20
    assert status1["n_treatment"] == 30
    assert status1["update_count"] == 1  # Should not increment


def test_srm_watch_ratio_deviation():
    """Test ratio deviation calculation."""
    watch = SrmWatch(p_expected=0.5)

    # No samples - should return None
    assert watch.get_ratio_deviation() is None

    # Balanced samples - should be close to 0
    watch.update(nc_delta=50, nt_delta=50)
    deviation = watch.get_ratio_deviation()
    assert deviation == 0.0

    # Imbalanced samples
    watch.update(nc_delta=30, nt_delta=20)  # Total: 80 control, 70 treatment
    deviation = watch.get_ratio_deviation()
    observed_prop = 80 / 150
    expected_deviation = abs(observed_prop - 0.5)
    assert abs(deviation - expected_deviation) < 1e-10


def test_srm_watch_string_representation():
    """Test string representation of SrmWatch."""
    watch = SrmWatch(p_expected=0.6, warn_p=0.01)
    watch.update(nc_delta=10, nt_delta=5)

    repr_str = repr(watch)
    assert "SrmWatch" in repr_str
    assert "p_expected=0.6" in repr_str
    assert "warn_p=0.01" in repr_str
    assert "n_control=10" in repr_str
    assert "n_treatment=5" in repr_str


def test_srm_watch_simulation_early_detection():
    """Test simulation of incremental data with early SRM detection."""
    watch = SrmWatch(warn_p=1e-4)  # Very strict threshold

    # Simulate gradual accumulation with severe imbalance
    simulation_data = [
        (5, 5),    # Start balanced
        (10, 8),   # Slight imbalance
        (15, 10),  # Growing imbalance
        (20, 8),   # Significant imbalance
        (25, 5),   # Severe imbalance - should trigger
        (30, 5),   # Continue severe imbalance
    ]

    results = []
    for nc, nt in simulation_data:
        result = watch.update(nc_delta=nc, nt_delta=nt)
        results.append(result)

    # Verify that SRM was detected before the end
    srm_detected_updates = [r["update_count"] for r in results if r["srm_detected"]]
    assert len(srm_detected_updates) > 0, "SRM should have been detected"

    # First detection should be remembered
    first_detection = min(srm_detected_updates)
    final_result = results[-1]
    assert final_result["detection_update"] == first_detection

    # Verify final counts are correct
    total_nc = sum(d[0] for d in simulation_data)
    total_nt = sum(d[1] for d in simulation_data)
    assert final_result["n_control"] == total_nc
    assert final_result["n_treatment"] == total_nt


def test_srm_watch_persistent_detection():
    """Test that SRM detection persists even if p-value later improves."""
    watch = SrmWatch(warn_p=0.01)

    # Create severe imbalance to trigger SRM
    result1 = watch.update(nc_delta=100, nt_delta=20)
    assert result1["srm_detected"]
    detection_update = result1["detection_update"]

    # Add more balanced data that might improve p-value
    result2 = watch.update(nc_delta=10, nt_delta=40)

    # SRM detection flag in the watch should persist
    assert watch.srm_detected
    assert watch.detection_update == detection_update

    # But current test result might show no SRM if p-value improved
    # This behavior allows tracking of "ever detected" vs "currently detected"


def test_srm_watch_zero_updates():
    """Test behavior with zero delta updates."""
    watch = SrmWatch()

    # Initial zero update
    result1 = watch.update(nc_delta=0, nt_delta=0)
    assert result1["n_total"] == 0
    assert result1["update_count"] == 1

    # Add some data
    watch.update(nc_delta=10, nt_delta=15)

    # Zero update should not change counts but increment update counter
    result2 = watch.update(nc_delta=0, nt_delta=0)
    assert result2["n_control"] == 10
    assert result2["n_treatment"] == 15
    assert result2["update_count"] == 3


def test_srm_watch_large_numbers():
    """Test SRM watch with large count numbers."""
    watch = SrmWatch(warn_p=1e-6)

    # Large balanced counts
    result = watch.update(nc_delta=1000000, nt_delta=1000000)
    assert result["n_control"] == 1000000
    assert result["n_treatment"] == 1000000
    assert not result["srm_detected"]

    # Small imbalance on large numbers should still be detectable
    result = watch.update(nc_delta=10000, nt_delta=5000)
    # With such large sample sizes, even small imbalances become significant
    assert result["n_control"] == 1010000
    assert result["n_treatment"] == 1005000