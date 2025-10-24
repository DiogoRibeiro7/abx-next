from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(slots=True)
class PowerConfig:
    n_per_arm: int
    baseline_mean: float
    baseline_std: float
    lift: float
    alpha: float = 0.05
    reps: int = 10_000

def estimate_power(cfg: PowerConfig) -> float:
    if cfg.n_per_arm <= 1 or cfg.baseline_std <= 0 or cfg.reps < 1000:
        raise ValueError("Invalid config: increase sample size/std>0/reps>=1000.")
    rng = np.random.default_rng(123)
    mu_c = cfg.baseline_mean
    mu_t = cfg.baseline_mean + cfg.lift
    s = cfg.baseline_std
    n = cfg.n_per_arm
    zcrit = norm.ppf(1 - cfg.alpha / 2)
    reject = 0
    for _ in range(cfg.reps):
        c = rng.normal(mu_c, s, size=n).mean()
        t = rng.normal(mu_t, s, size=n).mean()
        z = (t - c) / (s * (2 / n) ** 0.5)
        if abs(z) >= zcrit:
            reject += 1
    return reject / cfg.reps

if __name__ == "__main__":
    cfg = PowerConfig(n_per_arm=5000, baseline_mean=1.0, baseline_std=1.0, lift=0.03, reps=20000)
    print("Estimated power:", estimate_power(cfg))
