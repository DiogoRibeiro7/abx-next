"""Tests for SRM by strata diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from experimetrics.analysis.srm_diag import srm_by_strata


def test_srm_by_strata_flags_country() -> None:
    rng = np.random.default_rng(42)
    n_control = 200
    n_treatment = 200

    control_countries = np.array(["US"] * 150 + ["CA"] * 50)
    treatment_countries = np.array(["US"] * 80 + ["CA"] * 120)

    control_devices = rng.choice(["mobile", "desktop"], size=n_control)
    treatment_devices = rng.choice(["mobile", "desktop"], size=n_treatment)

    df = pd.DataFrame(
        {
            "group": np.concatenate(
                [np.repeat("control", n_control), np.repeat("treatment", n_treatment)]
            ),
            "country": np.concatenate([control_countries, treatment_countries]),
            "device": np.concatenate([control_devices, treatment_devices]),
        }
    )

    report = srm_by_strata(df, features=["country", "device"])
    assert not report.empty
    country_rows = report[report["feature"] == "country"]
    assert not country_rows.empty
    assert (country_rows["category"] == "US").any()
    assert country_rows[country_rows["category"] == "US"]["pvalue"].iloc[0] < 0.05

    if "device" in report["feature"].values:
        device_min_p = report.loc[report["feature"] == "device", "pvalue"].min()
        assert device_min_p > 0.05
