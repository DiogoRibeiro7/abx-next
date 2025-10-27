from __future__ import annotations

import pytest

# Tests for switchback power utilities.
from abx_next.core.errors import ValidationError
from abx_next.sim.power_switchback import estimate_power_switchback, required_blocks_for_power


def test_power_increases_with_blocks() -> None:
    powers = []
    for blocks in [2, 4, 6, 8]:
        power = estimate_power_switchback(
            mu_c=0.0,
            mu_t=0.5,
            sigma=1.0,
            rho_intra=0.2,
            n_blocks=blocks,
            block_size=20,
            reps=1000,
            seed=42,
        )
        powers.append(power)
    assert powers == sorted(powers)


def test_required_blocks_for_power() -> None:
    blocks = required_blocks_for_power(
        target_power=0.8,
        mu_c=0.0,
        mu_t=0.6,
        sigma=1.0,
        rho_intra=0.1,
        block_size=30,
        alpha=0.05,
        reps=500,
        max_blocks=40,
        seed=123,
    )
    assert blocks % 2 == 0
    power = estimate_power_switchback(
        mu_c=0.0,
        mu_t=0.6,
        sigma=1.0,
        rho_intra=0.1,
        n_blocks=blocks,
        block_size=30,
        reps=1000,
        seed=123,
    )
    assert power >= 0.75


def test_invalid_parameters() -> None:
    with pytest.raises(ValidationError):
        estimate_power_switchback(0.0, 0.5, sigma=1.0, rho_intra=1.2, n_blocks=4, block_size=10)
    with pytest.raises(ValidationError):
        estimate_power_switchback(0.0, 0.5, sigma=1.0, rho_intra=0.5, n_blocks=3, block_size=10)
    with pytest.raises(ValidationError):
        required_blocks_for_power(1.2, 0.0, 0.5, sigma=1.0, rho_intra=0.1, block_size=10)