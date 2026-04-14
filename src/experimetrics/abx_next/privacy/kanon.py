"""K-anonymity validation utilities for privacy protection."""

from __future__ import annotations

import pandas as pd

from ..core.errors import ValidationError


def assert_k_anonymity(df: pd.DataFrame, quasi_cols: list[str], k: int) -> None:
    """Assert that DataFrame satisfies k-anonymity on quasi-identifiers.

    K-anonymity ensures that each row is indistinguishable from at least k-1 other
    rows when considering only the quasi-identifier columns. This helps protect
    individual privacy in published datasets.

    Args:
        df: DataFrame to validate
        quasi_cols: List of column names that serve as quasi-identifiers
        k: Minimum group size required for k-anonymity

    Raises:
        ValidationError: If k-anonymity is violated or parameters are invalid

    Examples:
        >>> df = pd.DataFrame({
        ...     'age_group': ['20-30', '20-30', '30-40', '30-40'],
        ...     'city': ['NYC', 'NYC', 'LA', 'LA'],
        ...     'salary': [50000, 55000, 60000, 65000]
        ... })
        >>> assert_k_anonymity(df, ['age_group', 'city'], k=2)  # Passes
        >>> assert_k_anonymity(df, ['age_group', 'city'], k=3)  # Raises ValidationError
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError("Input must be a pandas DataFrame")

    if not isinstance(quasi_cols, list) or not quasi_cols:
        raise ValidationError("quasi_cols must be a non-empty list of column names")

    if not isinstance(k, int) or k < 1:
        raise ValidationError("k must be a positive integer")

    if df.empty:
        raise ValidationError("DataFrame cannot be empty")

    # Check that all quasi-identifier columns exist
    missing_cols = set(quasi_cols) - set(df.columns)
    if missing_cols:
        raise ValidationError(f"Columns not found in DataFrame: {sorted(missing_cols)}")

    # Group by quasi-identifiers and count group sizes
    group_sizes = df.groupby(quasi_cols, dropna=False).size()

    if group_sizes.empty:
        raise ValidationError("No groups found after grouping by quasi-identifiers")

    # Check if any group violates k-anonymity
    violating_groups = group_sizes[group_sizes < k]

    if not violating_groups.empty:
        min_group_size = group_sizes.min()
        num_violating = len(violating_groups)
        total_groups = len(group_sizes)

        # Create detailed error message
        error_msg = (
            f"K-anonymity violation: {num_violating} out of {total_groups} groups "
            f"have fewer than {k} records. "
            f"Minimum group size: {min_group_size}. "
        )

        # Show a few examples of violating groups for debugging
        if len(violating_groups) <= 5:
            examples = []
            for group_key, size in violating_groups.items():
                if isinstance(group_key, tuple):
                    key_str = ", ".join(f"{col}={val}" for col, val in zip(quasi_cols, group_key))
                else:
                    key_str = f"{quasi_cols[0]}={group_key}"
                examples.append(f"({key_str}): {size} records")
            error_msg += f"Violating groups: {'; '.join(examples)}"
        else:
            error_msg += f"Example violating group has {violating_groups.iloc[0]} records."

        raise ValidationError(error_msg)