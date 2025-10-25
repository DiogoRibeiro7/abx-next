# Sequential / Anytime-valid

Optional stopping breaks classic fixed-horizon confidence intervals. The
sequential module exposes Clopper–Pearson confidence sequences that remain valid
even when teams peek at the dashboard every hour.

```python
import numpy as np

from abx_next.analysis import bernoulli_ci_anytime, diff_ci_anytime_binomial

rng = np.random.default_rng(2024)
daily_conversions = rng.binomial(1, 0.28, size=500)

successes = int(daily_conversions.sum())
interval = bernoulli_ci_anytime(successes, trials=len(daily_conversions), alpha=0.05)
print(interval)

lift_interval = diff_ci_anytime_binomial(
    sc_c=140, n_c=500,
    sc_t=170, n_t=500,
    alpha=0.05,
)
print(lift_interval)
```

Because these intervals are conservative by construction, you should expect wider
intervals early in the experiment that shrink as more data arrives. Logging and
monitoring remain safe—no need to hide dashboards until day 14.
