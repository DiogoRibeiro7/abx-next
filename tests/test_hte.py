
"""Tests for uplift modeling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from sklearn.metrics import roc_auc_score
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pytest.skip("scikit-learn required for uplift tests", allow_module_level=True)

from experimetrics.core.errors import ValidationError
from experimetrics.hte import estimate_uplift


def _make_synthetic(n: int = 2000, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    propensity = 1 / (1 + np.exp(-x1))
    treatment = rng.binomial(1, propensity)
    base = 0.5 * x1 + 0.2 * x2
    tau = np.where(x1 > 0, 1.0, 0.1)
    outcome = base + tau * treatment + rng.normal(0, 0.5, size=n)

    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "group": np.where(treatment == 1, "treatment", "control"),
            "outcome": outcome,
            "true_tau": tau,
        }
    )
    return df


def test_estimate_uplift_auc() -> None:
    df = _make_synthetic()
    uplift = estimate_uplift(df, features=["x1", "x2"], outcome="outcome")
    auc = roc_auc_score(df["true_tau"] > 0.5, uplift)
    assert auc > 0.6


def test_invalid_treatment_column() -> None:
    df = _make_synthetic()
    df["group"] = "control"
    with pytest.raises(ValidationError):
        estimate_uplift(df, features=["x1"], outcome="outcome")

