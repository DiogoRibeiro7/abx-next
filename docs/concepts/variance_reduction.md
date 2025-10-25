# Variance reduction

Variance reduction techniques squeeze more signal out of noisy experiments without
changing user-facing experiences. `abx_next` ships CUPED/CUPAC helpers and direct
ratio methods so you can plug them into existing reporting pipelines.

## CUPED in a few lines

```python
import pandas as pd

from abx_next import ABFrame, cuped_adjust

raw = pd.DataFrame(
    {
        "user_id": [1, 2, 3, 4],
        "group": ["control", "control", "treatment", "treatment"],
        "metric": [1.2, 0.9, 1.4, 1.1],
        "exposed": [True, True, True, True],
        "baseline": [0.8, 0.7, 0.6, 0.6],
    }
)

ab = ABFrame(raw)
ab.validate()

adjusted, theta = cuped_adjust(ab, covariate=raw["baseline"])
print(theta)
print(adjusted[["user_id", "metric", "metric_cuped"]])
```

## CUPAC with a scikit-learn model

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

from abx_next import ABFrame, cuped_adjust
from abx_next.providers import SklearnCovariateProvider

events = pd.DataFrame(
    {
        "user_id": range(100),
        "group": ["control"] * 50 + ["treatment"] * 50,
        "metric": pd.Series(range(100)).astype(float) * 0.01 + 0.2,
        "exposed": [True] * 100,
    }
)

features = pd.DataFrame(
    {
        "user_id": range(100),
        "baseline": pd.Series(range(100)).astype(float) * 0.02,
        "traffic_source": ["email"] * 40 + ["search"] * 60,
    }
)

model = LinearRegression()
model.fit(features[["baseline"]], events["metric"])

provider = SklearnCovariateProvider(
    model=model,
    feature_df=features,
    key_col="user_id",
    feature_cols=["baseline"],
)

ab = ABFrame(events)
adjusted, _ = cuped_adjust(ab, cov_provider=provider)
print(adjusted.head())
```

## Ratio metrics without bespoke spreadsheets

```python
from abx_next.analysis import ratio_of_means_ci

data = {
    "num_control": [220.0, 210.0, 215.0],
    "den_control": [1000.0, 995.0, 1005.0],
    "num_treatment": [255.0, 250.0, 260.0],
    "den_treatment": [1005.0, 1010.0, 1008.0],
}

summary = ratio_of_means_ci(
    num_c=data["num_control"],
    den_c=data["den_control"],
    num_t=data["num_treatment"],
    den_t=data["den_treatment"],
)
print(summary)
```

The helper returns the point estimate, delta-method standard error, and a
Welch-style confidence interval that stays readable in notebooks or dashboards.
Use these diagnostics together with SRM checks to quickly diagnose uplift wins
without reinventing statistics from scratch.
