
"""Metric registry for consistent definitions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..core.errors import ValidationError

__all__ = [
    "MetricDefinition",
    "register_metric",
    "get_metric",
    "list_metrics",
    "load_metric_yaml",
    "validate_dataset_for_metrics"
]


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    kind: str
    aggregation: str
    required_columns: tuple[str, ...]
    column_types: dict[str, str] | None = None
    description: str | None = None


_REGISTRY: dict[str, MetricDefinition] = {}


def register_metric(
    name: str,
    *,
    kind: str,
    aggregation: str,
    required_columns: Iterable[str],
    column_types: dict[str, str] | None = None,
    description: str | None = None,
) -> MetricDefinition:
    """Register a metric definition unless it already exists."""
    norm_name = name.strip().lower()
    if not norm_name:
        raise ValidationError("Metric name must be non-empty.")
    if norm_name in _REGISTRY:
        raise ValidationError(f"Metric '{name}' already registered.")

    columns = tuple(dict.fromkeys(required_columns))
    if not columns:
        raise ValidationError("required_columns must contain at least one column.")

    definition = MetricDefinition(
        name=norm_name,
        kind=kind,
        aggregation=aggregation,
        required_columns=columns,
        column_types=column_types,
        description=description,
    )
    _REGISTRY[norm_name] = definition
    return definition


def get_metric(name: str) -> MetricDefinition:
    """Return a registered metric definition."""
    norm_name = name.strip().lower()
    if norm_name not in _REGISTRY:
        raise ValidationError(f"Metric '{name}' not registered.")
    return _REGISTRY[norm_name]


def list_metrics() -> list[MetricDefinition]:
    """Return all registered metrics."""
    return list(_REGISTRY.values())



def load_metric_yaml(path: str | Path) -> dict[str, Any]:
    """Load metric definitions from YAML file and register them.

    Args:
        path: Path to YAML file containing metric definitions

    Returns:
        Dictionary containing the loaded metric definitions

    Raises:
        ValidationError: If YAML is invalid or metric definitions are malformed
    """
    try:
        import yaml
    except ImportError as e:
        raise ValidationError(
            "PyYAML is required for YAML metric loading. Install with: pip install PyYAML"
        ) from e

    file_path = Path(path)
    if not file_path.exists():
        raise ValidationError(f"Metric YAML file not found: {path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValidationError(f"Invalid YAML in {path}: {e}") from e
    except Exception as e:
        raise ValidationError(f"Error reading {path}: {e}") from e

    if not isinstance(data, dict):
        raise ValidationError(f"YAML root must be a dictionary in {path}")

    metrics_data = data.get('metrics', {})
    if not isinstance(metrics_data, dict):
        raise ValidationError("'metrics' section must be a dictionary")

    # Register metrics from YAML
    for metric_name, metric_config in metrics_data.items():
        if not isinstance(metric_config, dict):
            raise ValidationError(f"Metric '{metric_name}' configuration must be a dictionary")

        try:
            # Extract required fields
            kind = metric_config.get('kind')
            if not kind:
                raise ValidationError(f"Metric '{metric_name}' missing required field 'kind'")

            aggregation = metric_config.get('aggregation')
            if not aggregation:
                raise ValidationError(f"Metric '{metric_name}' missing required field 'aggregation'")

            required_columns = metric_config.get('required_columns', [])
            if not required_columns:
                raise ValidationError(f"Metric '{metric_name}' missing required field 'required_columns'")

            # Extract optional fields
            column_types = metric_config.get('column_types')
            description = metric_config.get('description')

            # Register the metric (will raise ValidationError if already exists)
            register_metric(
                name=metric_name,
                kind=kind,
                aggregation=aggregation,
                required_columns=required_columns,
                column_types=column_types,
                description=description
            )

        except ValidationError:
            raise  # Re-raise validation errors as-is
        except Exception as e:
            raise ValidationError(
                f"Error processing metric '{metric_name}': {e}"
            ) from e

    return data


def validate_dataset_for_metrics(
    df: pd.DataFrame,
    metric_names: list[str],
    strict_types: bool = True
) -> None:
    """Validate that a DataFrame contains required columns for specified metrics.

    Args:
        df: DataFrame to validate
        metric_names: List of metric names to validate against
        strict_types: If True, validate column types match expected types

    Raises:
        ValidationError: If validation fails with detailed error messages
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError("Input must be a pandas DataFrame")

    if df.empty:
        raise ValidationError("DataFrame cannot be empty")

    if not metric_names:
        raise ValidationError("metric_names cannot be empty")

    # Collect all required columns and their types
    all_required_columns = set()
    column_type_requirements = {}

    for metric_name in metric_names:
        try:
            metric = get_metric(metric_name)
            all_required_columns.update(metric.required_columns)

            if strict_types and metric.column_types:
                for col, expected_type in metric.column_types.items():
                    if col in column_type_requirements and column_type_requirements[col] != expected_type:
                        raise ValidationError(
                            f"Conflicting type requirements for column '{col}': "
                            f"expected both '{column_type_requirements[col]}' and '{expected_type}'"
                        )
                    column_type_requirements[col] = expected_type

        except ValidationError:
            raise  # Re-raise metric not found errors

    # Check for missing columns
    df_columns = set(df.columns)
    missing_columns = all_required_columns - df_columns
    if missing_columns:
        raise ValidationError(
            f"Missing required columns: {sorted(missing_columns)}. "
            f"Available columns: {sorted(df_columns)}"
        )

    # Check column types if strict_types is enabled
    if strict_types and column_type_requirements:
        type_errors = []

        for col, expected_type in column_type_requirements.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)

                # Map pandas dtypes to expected type strings
                if expected_type == 'numeric':
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        type_errors.append(
                            f"Column '{col}' expected numeric type, got {actual_dtype}"
                        )
                elif expected_type == 'string':
                    if not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
                        type_errors.append(
                            f"Column '{col}' expected string type, got {actual_dtype}"
                        )
                elif expected_type == 'datetime':
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        type_errors.append(
                            f"Column '{col}' expected datetime type, got {actual_dtype}"
                        )
                elif expected_type == 'boolean':
                    if not pd.api.types.is_bool_dtype(df[col]):
                        type_errors.append(
                            f"Column '{col}' expected boolean type, got {actual_dtype}"
                        )
                # Add more type mappings as needed

        if type_errors:
            raise ValidationError(
                f"Type validation failed: {'; '.join(type_errors)}"
            )


def _clear_registry() -> None:
    """Reset registry (primarily for tests)."""
    _REGISTRY.clear()
