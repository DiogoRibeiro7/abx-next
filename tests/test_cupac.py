"""Tests for the scikit-learn based CUPAC provider."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from abx_next import ABFrame, cuped_adjust
from abx_next.providers import SklearnCovariateProvider


def test_sklearn_covariate_provider_reduces_variance() -> None:
    """Ensure CUPAC predictions reduce metric variance relative to the raw metric."""
    rng = np.random.default_rng(2024)
    n = 1500
    user_id = np.arange(n)
    baseline = rng.normal(loc=0.0, scale=1.0, size=n)
    context = rng.normal(loc=0.0, scale=1.0, size=n)
    treatment_mask = rng.random(n) < 0.5
    noise = rng.normal(loc=0.0, scale=0.1, size=n)

    metric = (
        1.0
        + 0.8 * baseline
        + 0.5 * context
        + 0.3 * treatment_mask.astype(float)
        + noise
    )

    df = pd.DataFrame({
        "user_id": user_id,
        "group": np.where(treatment_mask, "treatment", "control"),
        "metric": metric,
        "exposed": np.ones(n, dtype=bool),
    })

    feature_df = pd.DataFrame({
        "user_id": user_id,
        "baseline": baseline,
        "context": context,
    })

    model = LinearRegression()
    model.fit(feature_df[["baseline", "context"]], metric)

    provider = SklearnCovariateProvider(
        model=model,
        feature_df=feature_df,
        key_col="user_id",
        feature_cols=["baseline", "context"],
    )

    ab = ABFrame(df)
    ab.validate()

    adjusted_df, _ = cuped_adjust(ab, cov_provider=provider)

    raw_variance = float(df["metric"].var(ddof=1))

    adjusted_variance = float(adjusted_df["metric_cuped"].var(ddof=1))

    assert adjusted_variance < raw_variance


def test_sklearn_covariate_provider_missing_user_ids() -> None:
    """Ensure missing user identifiers produce a clear error."""
    feature_df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "x": [0.1, 0.2, 0.3],
    })
    model = LinearRegression()
    model.fit(feature_df[["x"]], [0.1, 0.2, 0.3])

    provider = SklearnCovariateProvider(
        model=model,
        feature_df=feature_df,
        key_col="user_id",
        feature_cols=["x"],
    )

    with pytest.raises(ValueError, match="Missing features"):
        provider.get_covariate(pd.Series([4]))

