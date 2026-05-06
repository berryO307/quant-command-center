"""
Latency statistics and mechanical event detection.

All array ops are vectorized — no Python loops over rows.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# ── Percentile ladder ────────────────────────────────────────────────────────

PERCENTILE_LADDER = [50, 75, 90, 95, 99, 99.9, 99.99]


def summary_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Tail-aware percentile table for one or more latency columns."""
    rows = []
    for col in columns:
        vals = df[col].dropna().values
        row  = {"stage": col, "n": len(vals), "mean_us": vals.mean()}
        for p in PERCENTILE_LADDER:
            row[f"p{p}"] = np.percentile(vals, p)
        row["max"] = vals.max()
        rows.append(row)
    return pd.DataFrame(rows).set_index("stage").round(2)


def compare_stages(stages: dict[str, pd.Series]) -> pd.DataFrame:
    """Side-by-side percentile table for multiple pipeline stages."""
    rows = []
    for name, s in stages.items():
        vals = np.asarray(s.dropna(), dtype=np.float64)
        row  = {"stage": name, "n": len(vals), "mean": vals.mean()}
        for p in PERCENTILE_LADDER:
            row[f"p{p}"] = np.percentile(vals, p)
        row["max"] = vals.max()
        rows.append(row)
    return pd.DataFrame(rows).set_index("stage").round(3)


def jitter_metrics(s: pd.Series) -> dict:
    """
    Multiple jitter definitions — no single metric captures the whole picture.
    Returns std, IQR, MAD, (p99-p50), (p999-p99).
    """
    vals = np.asarray(s.dropna(), dtype=np.float64)
    p    = np.percentile(vals, [50, 75, 25, 99, 99.9])
    return {
        "std":          float(vals.std()),
        "iqr":          float(p[1] - p[2]),
        "mad":          float(np.median(np.abs(vals - np.median(vals)))),
        "p99_minus_p50":  float(p[3] - p[0]),
        "p999_minus_p99": float(p[4] - p[3]),
    }


# ── Rolling percentiles ──────────────────────────────────────────────────────

def rolling_percentiles(
    series:    pd.Series,
    window:    int = 200,
    quantiles: Iterable[float] = (0.50, 0.99, 0.999),
) -> pd.DataFrame:
    """Rolling-window percentile bands. Window in samples, not wall-clock time."""
    out = pd.DataFrame(index=series.index)
    for q in quantiles:
        col      = f"p{q*100:g}".replace(".", "_")
        out[col] = series.rolling(window, min_periods=max(1, window // 4)).quantile(q)
    return out


# ── Burst / regime-shift detection ──────────────────────────────────────────

@dataclass(frozen=True)
class BurstEvent:
    """
    A sustained run of high latency. Distinct from a single-sample outlier.
    Structural cause: OS preemption, cache eviction, NIC backpressure.
    """
    start_idx:  int
    end_idx:    int
    duration:   int
    peak_value: float
    peak_idx:   int
    mean_value: float


# Legacy alias — keep RegimeShift pointing to the same class
RegimeShift = BurstEvent


def detect_bursts(
    values:         np.ndarray,
    threshold_pct:  float = 99.0,
    min_consecutive: int  = 5,
) -> list[BurstEvent]:
    """
    Detect contiguous runs of samples above the global p{threshold_pct}.
    A single outlier is noise; a run of ≥ min_consecutive is structural.
    """
    threshold = np.percentile(values, threshold_pct)
    above     = values > threshold

    padded = np.concatenate(([False], above, [False]))
    edges  = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0] - 1

    events = []
    for s, e in zip(starts, ends):
        if e - s + 1 < min_consecutive:
            continue
        seg          = values[s:e + 1]
        peak_offset  = int(np.argmax(seg))
        events.append(BurstEvent(
            start_idx  = int(s),
            end_idx    = int(e),
            duration   = int(e - s + 1),
            peak_value = float(seg.max()),
            peak_idx   = int(s + peak_offset),
            mean_value = float(seg.mean()),
        ))
    return events


# Keep old name in case anything imports it
def detect_regime_shifts(
    series:         pd.Series,
    threshold_pct:  float = 99.0,
    min_consecutive: int  = 5,
    rolling_window: int   = 50,
) -> list[BurstEvent]:
    """Alias for detect_bursts operating on rolling p99 instead of raw values."""
    rolling_p99      = series.rolling(rolling_window, min_periods=1).quantile(0.99)
    global_threshold = np.percentile(series.values, threshold_pct)
    above            = (rolling_p99 > global_threshold).values

    padded = np.concatenate(([False], above, [False]))
    edges  = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0] - 1

    events = []
    arr    = series.values
    for s, e in zip(starts, ends):
        if e - s + 1 < min_consecutive:
            continue
        seg         = arr[s:e + 1]
        peak_offset = int(np.argmax(seg))
        events.append(BurstEvent(
            start_idx  = int(s),
            end_idx    = int(e),
            duration   = int(e - s + 1),
            peak_value = float(seg.max()),
            peak_idx   = int(s + peak_offset),
            mean_value = float(seg.mean()),
        ))
    return events


def bursts_to_df(bursts: list[BurstEvent]) -> pd.DataFrame:
    """Convert a list of BurstEvent to a summary DataFrame."""
    if not bursts:
        return pd.DataFrame(columns=[
            "start_idx", "end_idx", "duration", "peak_idx",
            "peak_value", "mean_value",
        ])
    return pd.DataFrame([
        {
            "start_idx":  b.start_idx,
            "end_idx":    b.end_idx,
            "duration":   b.duration,
            "peak_idx":   b.peak_idx,
            "peak_value": round(b.peak_value, 2),
            "mean_value": round(b.mean_value, 2),
        }
        for b in bursts
    ])


# ── CCDF ─────────────────────────────────────────────────────────────────────

def ccdf(values: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Empirical survival function P(X > x).
    Uses (n - i) / n indexing — no artificial zero-drop at the maximum.
    """
    a          = np.asarray(values, dtype=np.float64)
    a          = a[np.isfinite(a)]
    sorted_vals = np.sort(a)
    n           = len(sorted_vals)
    survival    = (n - np.arange(n)) / n
    return sorted_vals, survival


# ── Autocorrelation ──────────────────────────────────────────────────────────

def autocorr(x: np.ndarray | pd.Series, max_lag: int = 200) -> np.ndarray:
    """
    Normalized autocorrelation at lags 0..max_lag via FFT convolution.
    O(n log n) — correct for any series length.
    """
    a   = np.asarray(x, dtype=np.float64)
    a   = a[np.isfinite(a)]
    a  -= a.mean()
    var = np.dot(a, a)
    if var == 0:
        return np.zeros(max_lag + 1)

    n      = len(a)
    fsize  = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    F      = np.fft.rfft(a, n=fsize)
    ac_full = np.fft.irfft(F * np.conj(F))[:n] / var
    return ac_full[:max_lag + 1]


# ── NIC burst / inter-arrival metrics ───────────────────────────────────────

def interarrival_us(t2_tsc: np.ndarray, cpu_ghz: float = 3.2) -> np.ndarray:
    """
    Compute per-tick inter-arrival times in µs from raw TSC producer stamps.

    The first element is set to the median (avoids a NaN / large garbage value
    at position 0 poisoning log-scale axes).
    """
    cycles = np.asarray(t2_tsc, dtype=np.float64)
    deltas = np.empty(len(cycles), dtype=np.float64)
    np.subtract(cycles[1:], cycles[:-1], out=deltas[1:])
    deltas[0] = np.median(deltas[1:])               # sentine for index 0
    # Convert cycles → µs
    deltas /= (cpu_ghz * 1e3)
    # Clamp negative deltas (cross-core TSC skew, wrap-around)
    np.clip(deltas, 1e-3, None, out=deltas)
    return deltas


def conditional_percentiles(
    x:        np.ndarray,     # predictor  (inter-arrival µs)
    y:        np.ndarray,     # response   (queue latency µs)
    n_bins:   int   = 40,
    log_x:    bool  = True,
    x_lo_pct: float = 1.0,
    x_hi_pct: float = 99.5,
    quantiles: tuple[float, ...] = (0.50, 0.95, 0.99),
) -> pd.DataFrame:
    """
    Bin x into n_bins (log-spaced if log_x), compute percentiles of y per bin.
    Returns a DataFrame with columns: bin_center, count, p50, p95, p99 (etc.).
    """
    lo = np.percentile(x, x_lo_pct)
    hi = np.percentile(x, x_hi_pct)
    lo = max(lo, 1e-3)

    if log_x:
        edges = np.geomspace(lo, hi, n_bins + 1)
    else:
        edges = np.linspace(lo, hi, n_bins + 1)

    bin_idx    = np.searchsorted(edges, x, side="right") - 1
    bin_idx    = np.clip(bin_idx, 0, n_bins - 1)
    centers    = (edges[:-1] + edges[1:]) / 2

    rows = []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() < 5:          # skip near-empty bins
            continue
        seg  = y[mask]
        row  = {"bin_center": centers[i], "count": int(mask.sum())}
        for q in quantiles:
            row[f"p{q*100:g}"] = float(np.percentile(seg, q * 100))
        rows.append(row)

    return pd.DataFrame(rows)