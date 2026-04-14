"""Tests for k-anonymity validation utilities."""

from __future__ import annotations

import pandas as pd
import pytest
from experimetrics.core.errors import ValidationError
from experimetrics.privacy.kanon import assert_k_anonymity


def test_k_anonymity_passes():
    """Test that k-anonymity validation passes when satisfied."""
    df = pd.DataFrame(
        {
            "age_group": ["20-30", "20-30", "30-40", "30-40", "40-50", "40-50"],
            "city": ["NYC", "NYC", "LA", "LA", "Chicago", "Chicago"],
            "salary": [50000, 55000, 60000, 65000, 70000, 75000],
        }
    )

    # Should pass with k=2 (each group has exactly 2 records)
    assert_k_anonymity(df, ["age_group", "city"], k=2)

    # Should also pass with k=1 (always satisfied)
    assert_k_anonymity(df, ["age_group", "city"], k=1)


def test_k_anonymity_k_equals_1_always_passes():
    """Test that k=1 always passes for any non-empty DataFrame."""
    # Single row should pass with k=1
    single_row = pd.DataFrame({"age": [25], "location": ["NYC"]})
    assert_k_anonymity(single_row, ["age"], k=1)

    # Multiple unique rows should pass with k=1
    unique_rows = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
    assert_k_anonymity(unique_rows, ["id"], k=1)


def test_k_anonymity_fails_when_violated():
    """Test that k-anonymity validation fails when k-anonymity is violated."""
    df = pd.DataFrame(
        {
            "age_group": ["20-30", "20-30", "30-40"],  # One group has 2, one has 1
            "city": ["NYC", "NYC", "LA"],
            "salary": [50000, 55000, 60000],
        }
    )

    # Should fail with k=2 (LA group has only 1 record)
    with pytest.raises(ValidationError, match="K-anonymity violation"):
        assert_k_anonymity(df, ["age_group", "city"], k=2)


def test_k_anonymity_fails_k_greater_than_group_size():
    """Test that k-anonymity fails when k > largest group size."""
    df = pd.DataFrame({"category": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})

    # Should fail with k=3 (largest group has only 2 records)
    with pytest.raises(ValidationError, match="K-anonymity violation"):
        assert_k_anonymity(df, ["category"], k=3)

    # Error should contain helpful information
    with pytest.raises(ValidationError, match="Minimum group size: 2"):
        assert_k_anonymity(df, ["category"], k=5)


def test_k_anonymity_single_column_quasi_identifier():
    """Test k-anonymity with single column quasi-identifier."""
    df = pd.DataFrame(
        {
            "region": ["North", "North", "North", "South", "South"],
            "revenue": [100, 200, 300, 400, 500],
        }
    )

    # Should pass with k=2 (North has 3, South has 2)
    assert_k_anonymity(df, ["region"], k=2)

    # Should fail with k=4
    with pytest.raises(ValidationError, match="K-anonymity violation"):
        assert_k_anonymity(df, ["region"], k=4)


def test_k_anonymity_multiple_column_quasi_identifiers():
    """Test k-anonymity with multiple quasi-identifier columns."""
    df = pd.DataFrame(
        {
            "age": [25, 25, 35, 35, 45],
            "gender": ["F", "F", "M", "M", "F"],
            "city": ["NYC", "NYC", "LA", "LA", "Chicago"],
            "income": [50000, 55000, 60000, 65000, 70000],
        }
    )

    # Should pass with k=2 for first two groups, but fail due to Chicago group
    with pytest.raises(ValidationError, match="K-anonymity violation"):
        assert_k_anonymity(df, ["age", "gender", "city"], k=2)

    # Should pass with k=1
    assert_k_anonymity(df, ["age", "gender", "city"], k=1)


def test_k_anonymity_with_missing_values():
    """Test k-anonymity handling of missing values."""
    df = pd.DataFrame(
        {
            "age_group": ["20-30", "20-30", None, None],
            "city": ["NYC", "NYC", "LA", "LA"],
            "value": [1, 2, 3, 4],
        }
    )

    # Missing values should be treated as a separate group
    assert_k_anonymity(df, ["age_group", "city"], k=2)

    # Should fail with k=3
    with pytest.raises(ValidationError, match="K-anonymity violation"):
        assert_k_anonymity(df, ["age_group", "city"], k=3)


def test_k_anonymity_error_message_details():
    """Test that error messages contain helpful details."""
    df = pd.DataFrame(
        {"category": ["A", "B", "C"], "subcategory": ["X", "Y", "Z"], "value": [1, 2, 3]}
    )

    # Each group has exactly 1 record
    with pytest.raises(ValidationError) as exc_info:
        assert_k_anonymity(df, ["category", "subcategory"], k=2)

    error_msg = str(exc_info.value)
    assert "3 out of 3 groups" in error_msg
    assert "fewer than 2 records" in error_msg
    assert "Minimum group size: 1" in error_msg
    assert "category=A, subcategory=X" in error_msg


def test_k_anonymity_validation_errors():
    """Test input validation for assert_k_anonymity function."""
    valid_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})

    # Test invalid DataFrame input
    with pytest.raises(ValidationError, match="Input must be a pandas DataFrame"):
        assert_k_anonymity("not_a_dataframe", ["col1"], k=2)

    # Test invalid quasi_cols input
    with pytest.raises(ValidationError, match="quasi_cols must be a non-empty list"):
        assert_k_anonymity(valid_df, [], k=2)

    with pytest.raises(ValidationError, match="quasi_cols must be a non-empty list"):
        assert_k_anonymity(valid_df, "col1", k=2)

    # Test invalid k input
    with pytest.raises(ValidationError, match="k must be a positive integer"):
        assert_k_anonymity(valid_df, ["col1"], k=0)

    with pytest.raises(ValidationError, match="k must be a positive integer"):
        assert_k_anonymity(valid_df, ["col1"], k=-1)

    with pytest.raises(ValidationError, match="k must be a positive integer"):
        assert_k_anonymity(valid_df, ["col1"], k=2.5)

    # Test empty DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(ValidationError, match="DataFrame cannot be empty"):
        assert_k_anonymity(empty_df, ["col1"], k=2)

    # Test missing columns
    with pytest.raises(ValidationError, match="Columns not found in DataFrame"):
        assert_k_anonymity(valid_df, ["nonexistent_col"], k=2)


def test_k_anonymity_edge_cases():
    """Test edge cases for k-anonymity validation."""
    # DataFrame with all identical rows
    identical_rows = pd.DataFrame({"category": ["A", "A", "A"], "value": [1, 1, 1]})
    assert_k_anonymity(identical_rows, ["category"], k=3)

    # DataFrame with single column and single unique value
    single_value = pd.DataFrame({"col": [1, 1, 1, 1]})
    assert_k_anonymity(single_value, ["col"], k=4)

    # Large k value that passes
    large_group = pd.DataFrame({"group": ["X"] * 100, "id": range(100)})
    assert_k_anonymity(large_group, ["group"], k=50)


def test_k_anonymity_comprehensive_scenario():
    """Test a comprehensive real-world-like scenario."""
    # Simulate a dataset that might be used in A/B testing reports
    df = pd.DataFrame(
        {
            "age_bucket": ["18-25", "18-25", "18-25", "26-35", "26-35", "36-45"],
            "region": ["US", "US", "EU", "US", "US", "US"],
            "device_type": ["mobile", "desktop", "mobile", "mobile", "desktop", "mobile"],
            "treatment": ["A", "B", "A", "A", "B", "A"],
            "conversion": [1, 0, 1, 1, 0, 0],
        }
    )

    # Should pass for k=1 (always true)
    assert_k_anonymity(df, ["age_bucket", "region"], k=1)

    # Should pass for k=2 when grouping by age_bucket and region
    # (18-25,US): 2, (18-25,EU): 1, (26-35,US): 2, (36-45,US): 1
    with pytest.raises(ValidationError, match="K-anonymity violation"):
        assert_k_anonymity(df, ["age_bucket", "region"], k=2)

    # Should pass when grouping by region only
    # US: 5, EU: 1
    with pytest.raises(ValidationError, match="K-anonymity violation"):
        assert_k_anonymity(df, ["region"], k=2)

    # Should pass when grouping by age_bucket only
    # 18-25: 3, 26-35: 2, 36-45: 1
    with pytest.raises(ValidationError, match="K-anonymity violation"):
        assert_k_anonymity(df, ["age_bucket"], k=2)
