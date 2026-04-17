import pandas as pd
import pytest
from experimetrics.analysis.filtering import filter_by_significance
from experimetrics.core.errors import ValidationError

def test_filter_by_significance_basic():
    df = pd.DataFrame({
        "metric": ["m1", "m2", "m3"],
        "pvalue": [0.01, 0.04, 0.2],
    })
    filtered = filter_by_significance(df, alpha=0.05)
    assert set(filtered["metric"]) == {"m1", "m2"}
    assert all(filtered["pvalue"] <= 0.05)

def test_filter_by_significance_custom_col():
    df = pd.DataFrame({
        "metric": ["m1", "m2"],
        "p": [0.01, 0.2],
    })
    filtered = filter_by_significance(df, alpha=0.05, p_col="p")
    assert list(filtered["metric"]) == ["m1"]

def test_filter_by_significance_invalid_col():
    df = pd.DataFrame({"metric": ["m1"], "pvalue": [0.1]})
    with pytest.raises(ValidationError):
        filter_by_significance(df, p_col="not_a_col")

def test_filter_by_significance_invalid_alpha():
    df = pd.DataFrame({"metric": ["m1"], "pvalue": [0.1]})
    with pytest.raises(ValidationError):
        filter_by_significance(df, alpha=1.5)
