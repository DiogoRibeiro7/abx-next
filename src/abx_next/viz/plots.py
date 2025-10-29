"""Plotting utilities for abx-next analysis results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import pandas as pd


def forest_plot(
    estimates: pd.DataFrame,
    est_col: str,
    low_col: str,
    high_col: str,
    label_col: str,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Create a forest plot for confidence intervals.

    Args:
        estimates: DataFrame containing estimation results
        est_col: Column name for point estimates
        low_col: Column name for confidence interval lower bounds
        high_col: Column name for confidence interval upper bounds
        label_col: Column name for row labels

    Returns:
        Tuple of (figure, axes) objects
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install abx-next[viz]"
        ) from e

    fig, ax = plt.subplots(figsize=(8, len(estimates) * 0.5 + 1))

    y_positions = range(len(estimates))
    estimates_sorted = estimates.iloc[::-1]  # Reverse for top-to-bottom display

    # Plot confidence intervals as horizontal lines
    for i, (_, row) in enumerate(estimates_sorted.iterrows()):
        ax.plot(
            [row[low_col], row[high_col]],
            [i, i],
            'k-',
            linewidth=2,
            alpha=0.7
        )
        # Plot point estimate
        ax.plot(row[est_col], i, 'ko', markersize=6)

    # Add vertical line at zero if it's in the range
    x_min = estimates[[low_col, est_col, high_col]].min().min()
    x_max = estimates[[low_col, est_col, high_col]].max().max()
    if x_min <= 0 <= x_max:
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # Set labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(estimates_sorted[label_col].tolist())
    ax.set_xlabel('Estimate')

    # Clean up the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    return fig, ax


def time_effect_plot(
    ts_df: pd.DataFrame,
    ts_col: str,
    value_col: str,
    ci_low: str,
    ci_high: str,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Create a time series plot with confidence intervals.

    Args:
        ts_df: DataFrame containing time series data
        ts_col: Column name for timestamps
        value_col: Column name for the main values to plot
        ci_low: Column name for confidence interval lower bounds
        ci_high: Column name for confidence interval upper bounds

    Returns:
        Tuple of (figure, axes) objects
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install abx-next[viz]"
        ) from e

    # Ensure timestamp column is datetime
    ts_df_copy = ts_df.copy()
    ts_df_copy[ts_col] = pd.to_datetime(ts_df_copy[ts_col])
    ts_df_sorted = ts_df_copy.sort_values(ts_col)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the main line
    ax.plot(
        ts_df_sorted[ts_col],
        ts_df_sorted[value_col],
        'b-',
        linewidth=2,
        label='Estimate'
    )

    # Plot confidence interval as filled area
    ax.fill_between(
        ts_df_sorted[ts_col],
        ts_df_sorted[ci_low],
        ts_df_sorted[ci_high],
        alpha=0.3,
        color='blue',
        label='95% CI'
    )

    # Add horizontal line at zero if it's in the range
    y_min = ts_df_sorted[[ci_low, value_col, ci_high]].min().min()
    y_max = ts_df_sorted[[ci_low, value_col, ci_high]].max().max()
    if y_min <= 0 <= y_max:
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Format the plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    return fig, ax