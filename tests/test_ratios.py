"""Tests for ratio-of-means delta method helper."""

from __future__ import annotations

import numpy as np
import pytest

from abx_next.analysis import ratio_of_means_ci


def test_ratio_of_means_ci_basic_case() -> None:
    """Estimate should match analytical ratio with sensible interval."""
    stats = ratio_of_means_ci(
        num_c=[2.0, 4.0, 6.0, 8.0],
        den_c=[1.0, 1.0, 1.0, 1.0],
        num_t=[3.0, 6.0, 9.0, 12.0],
        den_t=[1.0, 1.0, 1.0, 1.0],
    )
    assert np.isclose(stats["estimate"], 1.5)
    assert 0.0 < stats["se"] < 1.0
    assert stats["ci_low"] < stats["estimate"] < stats["ci_high"]
    assert stats["df"] and stats["df"] > 0.0


def test_ratio_of_means_ci_zero_denominator_raises() -> None:
    """Zeros in denominators should surface a clear error."""
    with pytest.raises(ValueError, match="denominator"):
        ratio_of_means_ci(
            num_c=[1.0, 2.0],
            den_c=[1.0, 0.0],
            num_t=[1.0, 2.0],
            den_t=[1.0, 2.0],
        )


def test_ratio_of_means_ci_normal_approximation() -> None:
    """Normal approximation path should mirror the Welch result and report df None."""
    stats_welch = ratio_of_means_ci(
        num_c=[2.0, 3.0, 4.0, 5.0],
        den_c=[1.0, 1.0, 1.0, 1.0],
        num_t=[4.0, 6.0, 8.0, 10.0],
        den_t=[2.0, 2.0, 2.0, 2.0],
    )
    stats_norm = ratio_of_means_ci(
        num_c=[2.0, 3.0, 4.0, 5.0],
        den_c=[1.0, 1.0, 1.0, 1.0],
        num_t=[4.0, 6.0, 8.0, 10.0],
        den_t=[2.0, 2.0, 2.0, 2.0],
        welch=False,
    )

    assert stats_norm["df"] is None
    assert np.isclose(stats_welch["estimate"], stats_norm["estimate"])
    assert np.isclose(stats_welch["se"], stats_norm["se"], rtol=0.15)
    assert stats_norm["ci_low"] < stats_norm["estimate"] < stats_norm["ci_high"]

