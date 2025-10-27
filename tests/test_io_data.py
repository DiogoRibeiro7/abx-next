"""Tests for IO helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from abx_next.core.errors import ValidationError
from abx_next.io.data import load_ab_csv, save_results_json


def test_load_ab_csv_round_trip(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"user_id": [1, 2], "group": ["control", "treatment"]})
    df.to_csv(csv_path, index=False)

    loaded = load_ab_csv(csv_path, dtypes={"user_id": "int64", "group": "object"})
    pd.testing.assert_frame_equal(df, loaded)


def test_load_ab_csv_dtype_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"user_id": ["1", "bad"]})
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValidationError):
        load_ab_csv(csv_path, dtypes={"user_id": "int64"})


def test_save_results_json(tmp_path: Path) -> None:
    json_path = tmp_path / "results.json"
    payload = {"metric": 0.123}
    save_results_json(json_path, payload)
    with pytest.raises(ValidationError):
        save_results_json(json_path, payload)
    data = json.loads(json_path.read_text())
    assert data == payload
