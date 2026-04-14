"""Safe I/O helpers for experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

import pandas as pd

from ..core.errors import ValidationError

__all__ = ["load_ab_csv", "save_results_json"]


def load_ab_csv(path: str | Path, dtypes: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Load an A/B dataset from CSV with strict dtype enforcement."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise ValidationError(f"CSV file not found: {csv_path}")

    try:
        if dtypes is not None:
            df = pd.read_csv(csv_path, keep_default_na=False, dtype=dtypes)  # type: ignore[arg-type]
        else:
            df = pd.read_csv(csv_path, keep_default_na=False)
    except ValueError as exc:
        raise ValidationError(f"Failed to parse CSV: {exc}") from exc

    if dtypes:
        for column, dtype in dtypes.items():
            if column not in df.columns:
                raise ValidationError(f"Column '{column}' missing from CSV.")
            if df[column].dtype.name != dtype:
                raise ValidationError(
                    f"Column '{column}' expected dtype '{dtype}', found '{df[column].dtype.name}'."
                )
    return df



def save_results_json(
    path: str | Path,
    obj: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> None:
    """Persist experiment results to a JSON file."""
    json_path = Path(path)
    if json_path.exists() and not overwrite:
        raise ValidationError(f"File already exists: {json_path}")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write(os.linesep)
