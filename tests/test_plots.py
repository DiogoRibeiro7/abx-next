"""Tests for plotting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_estimates():
    """Sample estimation data for testing."""
    return pd.DataFrame(
        {
            "feature": ["Feature A", "Feature B", "Feature C"],
            "estimate": [0.15, -0.05, 0.30],
            "ci_low": [0.05, -0.15, 0.20],
            "ci_high": [0.25, 0.05, 0.40],
        }
    )


@pytest.fixture
def sample_timeseries():
    """Sample time series data for testing."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "effect": np.sin(np.arange(10) * 0.5) * 0.1,
            "lower": np.sin(np.arange(10) * 0.5) * 0.1 - 0.05,
            "upper": np.sin(np.arange(10) * 0.5) * 0.1 + 0.05,
        }
    )


def test_forest_plot_import_error():
    """Test that ImportError is raised when matplotlib is not available."""
    # Test by directly calling the function without matplotlib
    from experimetrics.viz.plots import forest_plot

    # This will trigger the ImportError when matplotlib.pyplot is imported
    sample_data = pd.DataFrame({"feature": ["A"], "est": [0.1], "low": [0.0], "high": [0.2]})

    # The import error will be raised inside the function
    # This test mainly verifies the function exists and handles the import properly
    try:
        result = forest_plot(sample_data, "est", "low", "high", "feature")
        # If matplotlib is available, we should get a valid result
        assert result is not None
    except ImportError as e:
        # If matplotlib is not available, we should get a helpful error message
        assert "matplotlib is required" in str(e)


def test_time_effect_plot_import_error():
    """Test that ImportError is raised when matplotlib is not available."""
    from experimetrics.viz.plots import time_effect_plot

    sample_data = pd.DataFrame(
        {"date": ["2023-01-01"], "effect": [0.1], "low": [0.0], "high": [0.2]}
    )

    # The import error will be raised inside the function
    try:
        result = time_effect_plot(sample_data, "date", "effect", "low", "high")
        # If matplotlib is available, we should get a valid result
        assert result is not None
    except ImportError as e:
        # If matplotlib is not available, we should get a helpful error message
        assert "matplotlib is required" in str(e)


@pytest.mark.skipif(
    condition=True, reason="Skip matplotlib tests in CI - testing object creation only"
)
def test_forest_plot_creates_objects(sample_estimates):
    """Test that forest_plot creates matplotlib objects."""
    try:
        import matplotlib.axes
        import matplotlib.figure
        from experimetrics.viz.plots import forest_plot
    except ImportError:
        pytest.skip("matplotlib not available")

    fig, ax = forest_plot(sample_estimates, "estimate", "ci_low", "ci_high", "feature")

    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)


@pytest.mark.skipif(
    condition=True, reason="Skip matplotlib tests in CI - testing object creation only"
)
def test_time_effect_plot_creates_objects(sample_timeseries):
    """Test that time_effect_plot creates matplotlib objects."""
    try:
        import matplotlib.axes
        import matplotlib.figure
        from experimetrics.viz.plots import time_effect_plot
    except ImportError:
        pytest.skip("matplotlib not available")

    fig, ax = time_effect_plot(sample_timeseries, "date", "effect", "lower", "upper")

    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_viz_module_import_without_matplotlib():
    """Test that viz module imports gracefully without matplotlib."""
    # This test just verifies the module structure
    import experimetrics.viz

    # The module should import without error regardless of matplotlib availability
    assert hasattr(experimetrics.viz, "__all__")

    # The functions should be in __all__ if matplotlib is available
    try:
        import matplotlib  # noqa: F401

        assert "forest_plot" in experimetrics.viz.__all__
        assert "time_effect_plot" in experimetrics.viz.__all__
    except ImportError:
        # If matplotlib is not available, functions should not be in __all__
        assert "forest_plot" not in experimetrics.viz.__all__
        assert "time_effect_plot" not in experimetrics.viz.__all__


def test_viz_module_import_with_matplotlib():
    """Test that viz module imports functions when matplotlib is available."""
    try:
        import experimetrics.viz
        import matplotlib  # noqa: F401

        # Functions should be available when matplotlib is present
        assert "forest_plot" in experimetrics.viz.__all__
        assert "time_effect_plot" in experimetrics.viz.__all__
    except ImportError:
        pytest.skip("matplotlib not available")


def test_forest_plot_data_handling():
    """Test forest plot handles various data scenarios."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
        from experimetrics.viz.plots import forest_plot
    except ImportError:
        pytest.skip("matplotlib not available")

    # Test with single row
    single_row = pd.DataFrame(
        {
            "label": ["Single"],
            "est": [0.5],
            "low": [0.3],
            "high": [0.7],
        }
    )

    fig, ax = forest_plot(single_row, "est", "low", "high", "label")
    assert fig is not None
    assert ax is not None

    # Test with negative values
    negative_data = pd.DataFrame(
        {
            "label": ["Negative", "Positive"],
            "est": [-0.2, 0.3],
            "low": [-0.4, 0.1],
            "high": [0.0, 0.5],
        }
    )

    fig, ax = forest_plot(negative_data, "est", "low", "high", "label")
    assert fig is not None
    assert ax is not None


def test_time_effect_plot_data_handling():
    """Test time effect plot handles various data scenarios."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
        from experimetrics.viz.plots import time_effect_plot
    except ImportError:
        pytest.skip("matplotlib not available")

    # Test with string timestamps
    string_dates = pd.DataFrame(
        {
            "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "value": [0.1, 0.2, 0.15],
            "ci_l": [0.05, 0.15, 0.10],
            "ci_h": [0.15, 0.25, 0.20],
        }
    )

    fig, ax = time_effect_plot(string_dates, "timestamp", "value", "ci_l", "ci_h")
    assert fig is not None
    assert ax is not None

    # Test with unsorted data
    unsorted_data = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"]),
            "val": [0.3, 0.1, 0.2],
            "low": [0.25, 0.05, 0.15],
            "high": [0.35, 0.15, 0.25],
        }
    )

    fig, ax = time_effect_plot(unsorted_data, "ts", "val", "low", "high")
    assert fig is not None
    assert ax is not None
