"""Tests for p-value adjustment utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from experimetrics.analysis import adjust_pvalues, familywise_report
from experimetrics.core.errors import ValidationError


def test_adjust_pvalues_bonferroni_and_holm() -> None:
    pvals = pd.Series([0.01, 0.02, 0.05], index=["A", "B", "C"])
    bonf = adjust_pvalues(pvals, method="bonferroni")
    expected_bonf = np.clip(pvals * len(pvals), 0, 1)
    assert np.allclose(bonf["p_adj"].to_numpy(), expected_bonf.to_numpy())

    holm = adjust_pvalues(pvals, method="holm")
    assert holm.loc["A", "p_adj"] <= holm.loc["B", "p_adj"] <= holm.loc["C", "p_adj"]
    assert holm["p_adj"].max() <= 1.0


def test_adjust_pvalues_bh_known_example() -> None:
    pvals = pd.Series([0.002, 0.008, 0.01, 0.04, 0.05])
    adjusted = adjust_pvalues(pvals, method="bh")
    expected = [0.01, 0.016666666666666666, 0.016666666666666666, 0.05, 0.05]
    assert np.allclose(adjusted["p_adj"].to_numpy(), expected)


def test_familywise_report_flags_hypotheses() -> None:
    df = pd.DataFrame(
        {
            "metric": ["m1", "m2", "m3"],
            "pvalue": [0.01, 0.04, 0.2],
        }
    )

    report = familywise_report(df, method="bh", alpha=0.05)
    assert set(report.columns) >= {"metric", "pvalue", "p_adj", "reject"}
    assert report.loc[0, "reject"]
    assert not bool(report.loc[2, "reject"])


def test_adjust_pvalues_validation() -> None:
    with pytest.raises(ValidationError):
        adjust_pvalues(pd.Series([0.1, -0.2]))
    with pytest.raises(ValidationError):
        adjust_pvalues(pd.Series([], dtype=float))
    with pytest.raises(ValidationError):
        adjust_pvalues(pd.Series([0.1, 0.2]), method="unknown")
