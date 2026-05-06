"""
Load and validate latency telemetry from the C++ market data gateway.
Single responsibility: get the data into a clean DataFrame, fail loud on
schema drift.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np


REQUIRED_COLS = {"queue_transit_ns", "parse_ns"}


def load_latency(path: str | Path) -> pd.DataFrame:
    """
    Load latency.csv and return a cleaned DataFrame with derived columns.

    Returns columns:
        queue_transit_ns, parse_ns           — raw ns values
        queue_transit_us, parse_us           — microsecond views
        end_to_end_us                        — sum of the two stages

    Raises if columns are missing or the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Latency CSV not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"latency.csv missing required columns: {missing}. "
            f"Got: {list(df.columns)}"
        )

    # Drop rows where either column is NaN — happens at start/end of capture
    n_before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLS)).reset_index(drop=True)
    if len(df) < n_before:
        print(f"[loader] dropped {n_before - len(df)} rows with NaN values")

    # Derived microsecond columns
    df["queue_transit_us"] = df["queue_transit_ns"] / 1000.0
    df["parse_us"] = df["parse_ns"] / 1000.0
    df["end_to_end_us"] = df["queue_transit_us"] + df["parse_us"]

    if df.empty:
        raise ValueError("latency.csv contains no usable rows after cleaning")

    return df