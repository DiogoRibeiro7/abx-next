"""Tests for SRM diagnostics helper."""

from __future__ import annotations

import pandas as pd

from abx_next.analysis import srm_diagnostics


def test_srm_diagnostics_detects_device_imbalance() -> None:
    """A clear SRM should surface device suspects."""
    records = []
    for _ in range(300):
        records.append({"group": "control", "device": "mobile", "country": "US"})
    for _ in range(50):
        records.append({"group": "control", "device": "mobile", "country": "CA"})
    for _ in range(500):
        records.append({"group": "treatment", "device": "desktop", "country": "US"})
    for _ in range(150):
        records.append({"group": "treatment", "device": "mobile", "country": "US"})

    df = pd.DataFrame(records)
    diag = srm_diagnostics(df, features=["device", "country"])

    assert diag["srm_p"] < 0.001
    devices = [s for s in diag["suspects"] if s["feature"] == "device"]
    assert devices, "Expected device imbalances to be flagged."
    assert any(s["category"] == "desktop" for s in devices)


def test_srm_diagnostics_balanced_returns_empty() -> None:
    """Balanced traffic should yield no suspects."""
    df = pd.DataFrame({
        "group": ["control", "treatment"] * 100,
        "device": ["mobile", "mobile"] * 100,
        "country": ["US", "US"] * 100,
    })
    diag = srm_diagnostics(df, features=["device", "country"])
    assert diag["srm_p"] >= 0.001
    assert diag["suspects"] == []


def test_srm_diagnostics_caps_cardinality() -> None:
    """Only the top 20 categories should be inspected for imbalance."""
    groups = ["control"] * 300 + ["treatment"] * 450
    ids = [f"user_{i % 30}" for i in range(750)]
    df = pd.DataFrame({
        "group": groups,
        "segment": ids,
    })
    diag = srm_diagnostics(df, features=["segment"])
    segment_suspects = [s for s in diag["suspects"] if s["feature"] == "segment"]
    # We should never return more than 20 categories (excluding aggregated other).
    unique_categories = {s["category"] for s in segment_suspects}
    assert len(unique_categories) <= 20
