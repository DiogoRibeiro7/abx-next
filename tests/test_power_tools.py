"""Tests for power calculation utilities."""

from __future__ import annotations

import pytest

from abx_next.sim import (
    power_mean_mc,
    power_mean_welch,
    power_prop_mc,
    power_prop_normal,
)


def test_power_mean_welch_matches_expectation() -> None:
    """Welch approximation should produce high power for large mean uplift."""
    power = power_mean_welch(
        mean_control=0.0,
        mean_treatment=0.5,
        std_control=1.0,
        std_treatment=1.0,
        n_control=400,
        n_treatment=400,
    )
    assert 0.95 < power < 1.0


def test_power_mean_mc_reproducible_and_reasonable() -> None:
    """Monte Carlo mean power should align with analytical approximation."""
    approx = power_mean_welch(
        mean_control=0.0,
        mean_treatment=0.3,
        std_control=1.0,
        std_treatment=1.0,
        n_control=200,
        n_treatment=200,
    )
    mc_first = power_mean_mc(
        mean_control=0.0,
        mean_treatment=0.3,
        std_control=1.0,
        std_treatment=1.0,
        n_control=200,
        n_treatment=200,
        reps=5000,
        seed=42,
    )
    mc_second = power_mean_mc(
        mean_control=0.0,
        mean_treatment=0.3,
        std_control=1.0,
        std_treatment=1.0,
        n_control=200,
        n_treatment=200,
        reps=5000,
        seed=42,
    )
    assert mc_first == pytest.approx(mc_second, rel=1e-9)
    assert mc_first == pytest.approx(approx, rel=0.1)


def test_power_prop_normal_behaves() -> None:
    """Proportion power approximation should increase with larger samples."""
    power_small = power_prop_normal(
        p_control=0.1,
        p_treatment=0.12,
        n_control=200,
        n_treatment=200,
    )
    power_large = power_prop_normal(
        p_control=0.1,
        p_treatment=0.12,
        n_control=1000,
        n_treatment=1000,
    )
    assert power_small < power_large
    assert 0.05 < power_small < 0.9
    assert 0.1 < power_large <= 1.0


def test_power_prop_mc_reproducible() -> None:
    """Monte Carlo power for proportions should be deterministic with a seed."""
    first = power_prop_mc(
        p_control=0.2,
        p_treatment=0.25,
        n_control=500,
        n_treatment=500,
        reps=6000,
        seed=7,
    )
    second = power_prop_mc(
        p_control=0.2,
        p_treatment=0.25,
        n_control=500,
        n_treatment=500,
        reps=6000,
        seed=7,
    )
    assert first == pytest.approx(second, rel=1e-9)
    assert 0.0 <= first <= 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_control": 1},
        {"std_control": 0.0},
        {"reps": 100},
    ],
)
def test_power_mean_invalid_inputs(kwargs: dict[str, object]) -> None:
    """Invalid arguments should raise ValueError."""
    base = dict(
        mean_control=0.0,
        mean_treatment=0.1,
        std_control=1.0,
        std_treatment=1.0,
        n_control=50,
        n_treatment=50,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        power_mean_mc(**base)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_control": -0.1},
        {"n_control": 1},
        {"reps": 500},
    ],
)
def test_power_prop_invalid_inputs(kwargs: dict[str, object]) -> None:
    """Invalid Monte Carlo inputs should raise ValueError."""
    base = dict(
        p_control=0.1,
        p_treatment=0.2,
        n_control=60,
        n_treatment=60,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        power_prop_mc(**base)
