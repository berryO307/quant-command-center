"""
Generate synthetic ticks.bin + latency.csv that match the C++ output.
Used to smoke-test the analysis pipeline end-to-end.
"""
import numpy as np
import pandas as pd
from pathlib import Path

# Match the dtype exactly — itemsize must equal C++ sizeof(NormalizedTick) = 51
TICK_DTYPE = np.dtype([
    ("event_time",      "<i8"),
    ("price",           "<i8"),
    ("qty",             "<i8"),
    ("best_bid",        "<i8"),
    ("best_ask",        "<i8"),
    ("agg_trade_id",    "<i8"),
    ("t2_tsc",          "<u8"),
    ("final_update_id", "<u8"),
    ("is_buyer_maker",  "i1"),
    ("stream_type",     "i1"),
    ("is_gap_resync",   "i1"),
])
assert TICK_DTYPE.itemsize == 67

OUT = Path("/home/claude/analysis/sample_data")
OUT.mkdir(exist_ok=True)

N = 250_000
rng = np.random.default_rng(42)

# Simulate a 30-min run starting now, with ~140 ticks/sec
start_ms = 1_730_000_000_000
event_time = start_ms + np.cumsum(rng.exponential(7, N)).astype(np.int64)

# Realistic price walk around 42000 USD, scaled ×1e8
price = (42000 * 1e8 + np.cumsum(rng.normal(0, 1e6, N))).astype(np.int64)
qty   = rng.integers(1e6, 5e8, N).astype(np.int64)
best_bid = price - rng.integers(1e4, 5e6, N).astype(np.int64)
best_ask = price + rng.integers(1e4, 5e6, N).astype(np.int64)

stream_type = rng.choice([0, 1], size=N, p=[0.7, 0.3]).astype(np.int8)
is_buyer_maker = np.where(stream_type == 1,
                           rng.choice([0, 1], size=N).astype(np.int8),
                           np.int8(-1))
agg_trade_id = np.where(stream_type == 1,
                         rng.integers(1, 1e9, N), 0).astype(np.int64)

# Simulate occasional gap resyncs (~ 5 events over the run)
is_gap_resync = np.zeros(N, dtype=np.int8)
for idx in rng.choice(N, size=5, replace=False):
    is_gap_resync[idx] = 1

final_update_id = np.cumsum(rng.integers(1, 4, N)).astype(np.uint64)
t2_tsc = (np.cumsum(rng.integers(20_000, 100_000, N))
          + rng.integers(0, 1_000_000_000)).astype(np.uint64)

# Pack into structured array
ticks = np.empty(N, dtype=TICK_DTYPE)
ticks["event_time"]      = event_time
ticks["price"]           = price
ticks["qty"]             = qty
ticks["best_bid"]        = best_bid
ticks["best_ask"]        = best_ask
ticks["agg_trade_id"]    = agg_trade_id
ticks["t2_tsc"]          = t2_tsc
ticks["final_update_id"] = final_update_id
ticks["is_buyer_maker"]  = is_buyer_maker
ticks["stream_type"]     = stream_type
ticks["is_gap_resync"]   = is_gap_resync

ticks.tofile(OUT / "ticks.bin")
print(f"wrote {OUT / 'ticks.bin'}: {N} ticks, {(OUT / 'ticks.bin').stat().st_size} bytes")

# ── Latency CSV — realistic distribution ──────────────────────────────────
# Parse cycles: tight log-normal around 4 µs (12.8k cycles @ 3.2 GHz)
# Queue cycles: log-normal around 30 µs (96k cycles), with occasional
#   tail spikes simulating OS scheduler preemption + cache misses
cpu_ghz = 3.2
parse_us = rng.lognormal(mean=np.log(4), sigma=0.25, size=N)
queue_us = rng.lognormal(mean=np.log(30), sigma=0.40, size=N)

# Inject burst events — 4 sustained high-latency clusters
for _ in range(4):
    burst_start = rng.integers(0, N - 100)
    burst_len   = rng.integers(20, 80)
    queue_us[burst_start:burst_start + burst_len] *= rng.uniform(2.5, 5.0)

# Inject rare extreme outliers (p99.99 spikes)
for _ in range(20):
    idx = rng.integers(0, N)
    queue_us[idx] *= rng.uniform(8, 15)

parse_cycles = (parse_us * 1000 * cpu_ghz).astype(np.uint64)
queue_cycles = (queue_us * 1000 * cpu_ghz).astype(np.uint64)

df = pd.DataFrame({
    "parse_cycles": parse_cycles,
    "queue_cycles": queue_cycles,
})
df.to_csv(OUT / "latency.csv", index=False)
print(f"wrote {OUT / 'latency.csv'}: {len(df)} rows")
