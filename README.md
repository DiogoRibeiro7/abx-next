# abx-next

Modern A/B experimentation utilities: CUPED/CUPAC hooks, triggered analysis, SRM, switchback helpers, and power simulations.

## Why
Shorter tests, safer reads, and reproducible designs. Focused on variance reduction, guardrails, and interference-aware designs.

## Install
```bash
poetry add abx-next
# or in a dev clone
poetry install
```

## Quickstart
```python
import pandas as pd
from abx_next import ABFrame, cuped_adjust, filter_exposed, diff_in_means, srm_from_frame

df = pd.DataFrame({...})  # group, metric, user_id, exposed, baseline
ab = ABFrame(df); ab.validate()
print(srm_from_frame(df))                  # guardrail
df_adj, theta = cuped_adjust(ab, df["baseline"])
stats = diff_in_means(filter_exposed(ABFrame(df_adj)), use_column="metric_cuped")
print(stats)
```

## Features
- CUPED (hooks for CUPAC)
- Triggered analysis helpers
- SRM (sample-ratio mismatch) test
- Switchback period assignment
- Power simulation utilities

## Contributing
Run `pre-commit`, `ruff`, `mypy`, and `pytest` before PRs. See `CONTRIBUTING.md`.
