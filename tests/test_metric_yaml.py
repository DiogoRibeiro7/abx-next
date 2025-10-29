"""Tests for YAML-based metric definitions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from abx_next.core.errors import ValidationError
from abx_next.metrics.registry import (
    _clear_registry,
    get_metric,
    list_metrics,
    load_metric_yaml,
    validate_dataset_for_metrics,
)


@pytest.fixture(autouse=True)
def clear_metrics_registry():
    """Clear metrics registry before and after each test."""
    _clear_registry()
    yield
    _clear_registry()


@pytest.fixture
def sample_yaml_content():
    """Sample YAML content for testing."""
    return """
metrics:
  conversion_rate:
    kind: "binary"
    aggregation: "mean"
    description: "User conversion rate"
    required_columns:
      - "user_id"
      - "converted"
      - "treatment"
    column_types:
      user_id: "string"
      converted: "boolean"
      treatment: "string"

  revenue_per_user:
    kind: "continuous"
    aggregation: "mean"
    description: "Average revenue per user"
    required_columns:
      - "user_id"
      - "revenue"
      - "treatment"
    column_types:
      user_id: "string"
      revenue: "numeric"
      treatment: "string"
"""


@pytest.fixture
def temp_yaml_file(sample_yaml_content):
    """Create a temporary YAML file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(sample_yaml_content)
        f.flush()
        yield Path(f.name)
    Path(f.name).unlink()


def test_load_metric_yaml_success(temp_yaml_file):
    """Test successful loading of metric YAML."""
    result = load_metric_yaml(temp_yaml_file)

    assert isinstance(result, dict)
    assert 'metrics' in result
    assert len(result['metrics']) == 2

    # Check that metrics were registered
    metrics = list_metrics()
    assert len(metrics) == 2

    conversion_metric = get_metric('conversion_rate')
    assert conversion_metric.name == 'conversion_rate'
    assert conversion_metric.kind == 'binary'
    assert conversion_metric.aggregation == 'mean'
    assert conversion_metric.description == 'User conversion rate'
    assert set(conversion_metric.required_columns) == {'user_id', 'converted', 'treatment'}
    assert conversion_metric.column_types == {
        'user_id': 'string',
        'converted': 'boolean',
        'treatment': 'string'
    }


def test_load_metric_yaml_file_not_found():
    """Test error when YAML file doesn't exist."""
    with pytest.raises(ValidationError, match="Metric YAML file not found"):
        load_metric_yaml("nonexistent.yaml")


def test_load_metric_yaml_invalid_yaml():
    """Test error with invalid YAML syntax."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        f.flush()
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValidationError, match="Invalid YAML"):
            load_metric_yaml(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def test_load_metric_yaml_not_dict():
    """Test error when YAML root is not a dictionary."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("- not a dict")
        f.flush()
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValidationError, match="YAML root must be a dictionary"):
            load_metric_yaml(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def test_load_metric_yaml_missing_metrics_section():
    """Test loading YAML without metrics section."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("other_section: value")
        f.flush()
        temp_path = Path(f.name)

    try:
        result = load_metric_yaml(temp_path)
        assert 'metrics' not in result or not result.get('metrics')
        assert len(list_metrics()) == 0
    finally:
        temp_path.unlink(missing_ok=True)


def test_load_metric_yaml_missing_required_fields():
    """Test error when metric is missing required fields."""
    yaml_content = """
metrics:
  incomplete_metric:
    kind: "binary"
    # missing aggregation and required_columns
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValidationError, match="missing required field 'aggregation'"):
            load_metric_yaml(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def test_load_metric_yaml_empty_required_columns():
    """Test error when required_columns is empty."""
    yaml_content = """
metrics:
  bad_metric:
    kind: "binary"
    aggregation: "mean"
    required_columns: []
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValidationError, match="missing required field 'required_columns'"):
            load_metric_yaml(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def test_validate_dataset_for_metrics_success(temp_yaml_file):
    """Test successful dataset validation."""
    load_metric_yaml(temp_yaml_file)

    # Create a valid dataset
    df = pd.DataFrame({
        'user_id': ['u1', 'u2', 'u3'],
        'converted': [True, False, True],
        'revenue': [100.0, 0.0, 150.0],
        'treatment': ['A', 'B', 'A'],
        'extra_column': ['x', 'y', 'z']  # Extra columns should be allowed
    })

    # Should pass validation
    validate_dataset_for_metrics(df, ['conversion_rate', 'revenue_per_user'])


def test_validate_dataset_missing_columns(temp_yaml_file):
    """Test validation error for missing columns."""
    load_metric_yaml(temp_yaml_file)

    # Dataset missing 'converted' column
    df = pd.DataFrame({
        'user_id': ['u1', 'u2'],
        'revenue': [100.0, 200.0],
        'treatment': ['A', 'B']
    })

    with pytest.raises(ValidationError, match="Missing required columns"):
        validate_dataset_for_metrics(df, ['conversion_rate'])


def test_validate_dataset_wrong_types(temp_yaml_file):
    """Test validation error for wrong column types."""
    load_metric_yaml(temp_yaml_file)

    # Dataset with wrong types
    df = pd.DataFrame({
        'user_id': [1, 2, 3],  # Should be string
        'converted': ['yes', 'no', 'yes'],  # Should be boolean
        'treatment': ['A', 'B', 'A']
    })

    with pytest.raises(ValidationError, match="Type validation failed"):
        validate_dataset_for_metrics(df, ['conversion_rate'])


def test_validate_dataset_strict_types_disabled(temp_yaml_file):
    """Test validation with strict types disabled."""
    load_metric_yaml(temp_yaml_file)

    # Dataset with wrong types but strict_types=False
    df = pd.DataFrame({
        'user_id': [1, 2, 3],  # Should be string but types not enforced
        'converted': [1, 0, 1],  # Should be boolean but types not enforced
        'treatment': ['A', 'B', 'A']
    })

    # Should pass when strict_types=False
    validate_dataset_for_metrics(df, ['conversion_rate'], strict_types=False)


def test_validate_dataset_invalid_inputs():
    """Test validation with invalid inputs."""
    # Test non-DataFrame input
    with pytest.raises(ValidationError, match="Input must be a pandas DataFrame"):
        validate_dataset_for_metrics("not a dataframe", ['some_metric'])

    # Test empty DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(ValidationError, match="DataFrame cannot be empty"):
        validate_dataset_for_metrics(empty_df, ['some_metric'])

    # Test empty metric names
    df = pd.DataFrame({'col': [1, 2]})
    with pytest.raises(ValidationError, match="metric_names cannot be empty"):
        validate_dataset_for_metrics(df, [])


def test_validate_dataset_metric_not_found():
    """Test validation error when metric doesn't exist."""
    df = pd.DataFrame({'col': [1, 2, 3]})

    with pytest.raises(ValidationError, match="Metric 'nonexistent_metric' not registered"):
        validate_dataset_for_metrics(df, ['nonexistent_metric'])


def test_load_metric_yaml_duplicate_metric(temp_yaml_file):
    """Test error when trying to register duplicate metrics."""
    # Load metrics once
    load_metric_yaml(temp_yaml_file)

    # Try to load the same file again (should fail due to duplicates)
    with pytest.raises(ValidationError, match="already registered"):
        load_metric_yaml(temp_yaml_file)


def test_complex_validation_scenario():
    """Test a complex validation scenario with multiple metrics and edge cases."""
    yaml_content = """
metrics:
  metric_a:
    kind: "binary"
    aggregation: "mean"
    required_columns: ["id", "flag", "group"]
    column_types:
      id: "string"
      flag: "boolean"
      group: "string"

  metric_b:
    kind: "continuous"
    aggregation: "mean"
    required_columns: ["id", "value", "group"]
    column_types:
      id: "string"
      value: "numeric"
      group: "string"
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()

        temp_path = Path(f.name)

    try:
        load_metric_yaml(temp_path)

        # Test dataset that satisfies both metrics
        df_good = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'flag': [True, False, True],
            'value': [1.0, 2.0, 3.0],
            'group': ['X', 'Y', 'X']
        })

        validate_dataset_for_metrics(df_good, ['metric_a', 'metric_b'])

        # Test dataset missing column for one metric
        df_bad = pd.DataFrame({
            'id': ['a', 'b'],
            'flag': [True, False],
            'group': ['X', 'Y']
            # Missing 'value' column needed for metric_b
        })

        with pytest.raises(ValidationError, match="Missing required columns"):
            validate_dataset_for_metrics(df_bad, ['metric_a', 'metric_b'])
    finally:
        temp_path.unlink(missing_ok=True)


def test_informative_error_messages():
    """Test that error messages are informative and helpful."""
    # Test detailed missing columns error
    yaml_content = """
metrics:
  test_metric:
    kind: "binary"
    aggregation: "mean"
    required_columns: ["col_a", "col_b", "col_c"]
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()

        temp_path = Path(f.name)

    try:
        load_metric_yaml(temp_path)

        df = pd.DataFrame({
            'col_a': [1, 2, 3],
            'col_x': ['a', 'b', 'c'],
            'col_y': [4, 5, 6]
        })

        with pytest.raises(ValidationError) as exc_info:
            validate_dataset_for_metrics(df, ['test_metric'])

        error_msg = str(exc_info.value)
        assert "Missing required columns: ['col_b', 'col_c']" in error_msg
        assert "Available columns: ['col_a', 'col_x', 'col_y']" in error_msg
    finally:
        temp_path.unlink(missing_ok=True)


def test_column_type_validation_edge_cases(temp_yaml_file):
    """Test edge cases in column type validation."""
    load_metric_yaml(temp_yaml_file)

    # Test with datetime columns (if we add datetime support)
    df_with_dates = pd.DataFrame({
        'user_id': ['u1', 'u2'],
        'converted': [True, False],
        'revenue': [100.0, 200.0],
        'treatment': ['A', 'B'],
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02'])
    })

    # Should pass validation even with extra datetime column
    validate_dataset_for_metrics(df_with_dates, ['conversion_rate'])


def test_yaml_without_pyyaml():
    """Test error when PyYAML is not available."""
    import sys
    from unittest.mock import patch

    # Temporarily remove yaml from sys.modules and mock the import
    original_modules = sys.modules.copy()
    if 'yaml' in sys.modules:
        del sys.modules['yaml']

    try:
        with patch.dict(sys.modules, {'yaml': None}):
            with pytest.raises(ValidationError, match="PyYAML is required"):
                load_metric_yaml("dummy.yaml")
    finally:
        sys.modules.update(original_modules)