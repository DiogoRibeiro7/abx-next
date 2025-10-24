from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd


def assign_switchback(
    timestamps: Iterable[pd.Timestamp],
    period: str = "D",
    seed: int | None = 17,
) -> pd.DataFrame:
    """Assign alternating period-level arms for switchback designs."""
    ts = pd.to_datetime(pd.Series(list(timestamps))).sort_values()
    if ts.empty:
        return pd.DataFrame(columns=["period_start", "group"])

    blocks = ts.dt.floor(period)
    unique_blocks = blocks.drop_duplicates().reset_index(drop=True)

    rng = np.random.default_rng(seed)
    start_treatment = rng.integers(0, 2) == 1
    arms = np.array(["control", "treatment"])
    seq = np.where(
        np.arange(len(unique_blocks)) % 2 == 0,
        arms[int(start_treatment)],
        arms[1 - int(start_treatment)],
    )

    return pd.DataFrame({"period_start": unique_blocks, "group": seq})
