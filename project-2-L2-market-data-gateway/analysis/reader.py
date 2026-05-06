"""
Binary file reader for the C++ market data gateway.

Reads two artefacts produced by the gateway:
  1. ticks.bin    — packed NormalizedTick records (50 bytes each)
  2. latency.csv  — per-tick rdtscp deltas (parse_cycles, queue_cycles)

Design choices:
  - np.memmap over read() — zero-copy, OS handles paging, files >RAM are fine
  - Structured dtype matches the C++ struct exactly — no parsing loop
  - itemsize asserted at module load — fail loud if struct layout drifts
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# --- NormalizedTick layout ---
# Mirrors the C++ struct exactly. Field order must match.
# Because C++ uses bit-fields for the first 8 bytes, NumPy must read 
# it as a single 64-bit integer. We unpack the bits later in Pandas/NumPy.
TICK_DTYPE = np.dtype([
    ("packed_word",  "<i8"),  # Contains event_time (56b) and the 3 flags (8b total)
    ("price",        "<i8"),  # scaled int (price * PRICE_SCALE)
    ("qty",          "<i8"),  # scaled int
    ("best_bid",     "<i8"),  # reconstructed top-of-book
    ("best_ask",     "<i8"),  # reconstructed top-of-book
    ("agg_trade_id", "<i8"),  # 0 for depth events
    ("t2_tsc",       "<u8"),  # rdtscp stamp at producer push
    ("u",            "<u8"),  # u (or final_update_id) for contiguity verification
])

# Expected: 8 fields × 8 bytes = 64 bytes exactly.
assert TICK_DTYPE.itemsize == 64, (
    f"TICK_DTYPE.itemsize is {TICK_DTYPE.itemsize}, expected 64. "
    "Check C++ NormalizedTick layout vs Python dtype."
)


# Stream type constants — use these in code, not magic numbers
STREAM_DEPTH = 0
STREAM_TRADE = 1


@dataclass(frozen=True)
class TickFile:
    """Container for a loaded ticks.bin file. Cheap to pass around — the
    underlying ndarray is a memmap, so this is just a view, not a copy."""
    ticks:    np.ndarray         # structured array, dtype=TICK_DTYPE
    path:     Path
    n_ticks:  int

    @property
    def trades(self) -> np.ndarray:
        return self.ticks[self.ticks["stream_type"] == STREAM_TRADE]

    @property
    def depths(self) -> np.ndarray:
        return self.ticks[self.ticks["stream_type"] == STREAM_DEPTH]

    @property
    def n_resyncs(self) -> int:
        """Count of sequence gaps + reconnects observed during capture."""
        return int(self.ticks["is_gap_resync"].sum())

    def summary(self) -> dict:
        """Dataset-level overview, useful for the top of any notebook."""
        first_ts = int(self.ticks["event_time"][0])
        last_ts  = int(self.ticks["event_time"][-1])
        duration_s = (last_ts - first_ts) / 1000.0
        return {
            "path":           str(self.path),
            "file_size_mb":   self.path.stat().st_size / 1024**2,
            "n_ticks":        self.n_ticks,
            "n_trades":       len(self.trades),
            "n_depths":       len(self.depths),
            "n_resyncs":      self.n_resyncs,
            "duration_s":     duration_s,
            "ticks_per_sec":  self.n_ticks / duration_s if duration_s > 0 else 0,
            "first_event_ms": first_ts,
            "last_event_ms":  last_ts,
        }


def read_ticks(path: str | Path) -> TickFile:
    """
    Memory-map a ticks.bin file and return a TickFile view.

    Parameters
    ----------
    path : str | Path
        Path to the binary file written by MmapWriter.

    Returns
    -------
    TickFile
        Lazy view — bytes are only paged in when accessed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ticks.bin not found: {path}")

    # np.memmap returns a structured array view directly — no copy.
    # mode='r' = read-only. Use 'r+' if you need to mutate (you almost never do).
    arr = np.memmap(path, dtype=TICK_DTYPE, mode="r")

    return TickFile(ticks=arr, path=path, n_ticks=len(arr))


# ── Latency CSV reader ──────────────────────────────────────────────────────

def read_latency(
    path:    str | Path,
    cpu_ghz: float = 3.2,
) -> pd.DataFrame:
    """
    Read the latency.csv produced by LatencyStore::dump().
    Auto-detects whether the C++ gateway wrote raw cycles or nanoseconds.
    """
    path = Path(path)
    df = pd.read_csv(path)

    # 1. Handle Schema Drift
    if "parse_cycles" in df.columns and "queue_cycles" in df.columns:
        # Modern schema: gateway dumps raw rdtscp deltas
        df["parse_ns"] = df["parse_cycles"] / cpu_ghz
        df["queue_ns"] = df["queue_cycles"] / cpu_ghz
        
    elif "parse_ns" in df.columns:
        # Legacy schema: gateway dumped nanoseconds
        # Handle naming inconsistencies from older captures
        if "queue_transit_ns" in df.columns:
            df["queue_ns"] = df["queue_transit_ns"]
            
        # Reverse-engineer cycles for the stall matrix
        df["parse_cycles"] = df["parse_ns"] * cpu_ghz
        df["queue_cycles"] = df["queue_ns"] * cpu_ghz
    else:
        raise ValueError(
            f"Unrecognized latency.csv schema. Expected 'parse_cycles' or 'parse_ns'. "
            f"Found: {list(df.columns)}"
        )

    # 2. Derive composite and microsecond columns
    df["end_to_end_ns"] = df["parse_ns"] + df["queue_ns"]
    
    df["parse_us"]      = df["parse_ns"]      / 1000.0
    df["queue_us"]      = df["queue_ns"]      / 1000.0
    df["end_to_end_us"] = df["end_to_end_ns"] / 1000.0

    return df


# ── Convenience joining ─────────────────────────────────────────────────────

def join_ticks_and_latency(
    tick_file: TickFile,
    latency:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Align tick metadata with latency samples by row index.
    Unpacks the 64-byte aligned bit-fields on the fly.
    """
    n_t, n_l = tick_file.n_ticks, len(latency)
    if n_t != n_l:
        n = min(n_t, n_l)
        print(f"[reader] WARNING: ticks.bin has {n_t} rows, "
              f"latency.csv has {n_l}. Truncating to {n}.")
    else:
        n = n_t

    out = latency.iloc[:n].copy()
    
    # ── BITWISE UNPACKING ───────────────────────────────────────────────────
    # Vectorized extraction of the 64-byte aligned bit-field ('packed_word')
    packed = tick_file.ticks["packed_word"][:n]
    
    # event_time: lower 56 bits
    out["event_time"] = packed & 0x00FFFFFFFFFFFFFF
    
    # flags: top 8 bits (shifted right by 56)
    flags = (packed >> 56) & 0xFF
    out["stream_type"]   = flags & 0x01
    out["is_gap_resync"] = (flags >> 1) & 0x01
    
    # 'final_update_id' is mapped to 'u' in the new compact struct
    out["final_update_id"] = tick_file.ticks["u"][:n]

    # Convert exchange ms timestamp to pandas datetime — enables resampling
    out["event_dt"] = pd.to_datetime(out["event_time"], unit="ms", utc=True)

    return out
