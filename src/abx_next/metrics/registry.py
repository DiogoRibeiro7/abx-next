
"""Metric registry for consistent definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..core.errors import ValidationError

__all__ = ["MetricDefinition", "register_metric", "get_metric", "list_metrics"]


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    kind: str
    aggregation: str
    required_columns: tuple[str, ...]


_REGISTRY: dict[str, MetricDefinition] = {}


def register_metric(
    name: str,
    *,
    kind: str,
    aggregation: str,
    required_columns: Iterable[str],
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



def _clear_registry() -> None:
    """Reset registry (primarily for tests)."""
    _REGISTRY.clear()
