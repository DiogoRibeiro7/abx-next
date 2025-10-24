from examples.power_simulation import PowerConfig, estimate_power


def test_power_valid():
    cfg = PowerConfig(n_per_arm=200, baseline_mean=0.0, baseline_std=1.0, lift=0.5, reps=2000)
    p = estimate_power(cfg)
    assert 0.0 <= p <= 1.0
