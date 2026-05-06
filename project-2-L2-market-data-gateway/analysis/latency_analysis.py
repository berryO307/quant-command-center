import matplotlib
matplotlib.use("Agg")

import sys
import traceback
from pathlib import Path
import numpy as np

from analysis.style import apply_style
from analysis.reader import read_ticks, read_latency, join_ticks_and_latency
from analysis.visualization import TradingStoryPlotter

apply_style()

HERE        = Path(__file__).resolve().parent
OUTPUT_DIR  = HERE / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR    = HERE.parent / "data"
CPU_GHZ     = 3.2

ticks   = read_ticks(DATA_DIR / "ticks.bin")
latency = read_latency(DATA_DIR / "latency.csv", cpu_ghz=CPU_GHZ)
df      = join_ticks_and_latency(ticks, latency)

n = len(df)
df["t2_tsc"]           = ticks.ticks["t2_tsc"][-n:]   # tail-align: latency rows = recent ticks
df["queue_transit_us"] = df["queue_us"]

print(f"rows={n:,}  p99={df['queue_transit_us'].quantile(0.99):.2f}µs  p999={df['queue_transit_us'].quantile(0.999):.2f}µs")

import matplotlib.pyplot as plt

plotter = TradingStoryPlotter(df, instrument="BTCUSDT", venue="Bybit")

for name, fn in [
    ("stall_matrix",             lambda: plotter.plot_stall_matrix(cpu_ghz=CPU_GHZ, save_to=str(OUTPUT_DIR / "stall_matrix.png"))),
    ("latency_regime_dashboard", lambda: plotter.plot_latency_regime_dashboard(save_to=str(OUTPUT_DIR / "latency_regime_dashboard.png"))),
    ("tail_risk",                lambda: plotter.plot_tail_risk(save_to=str(OUTPUT_DIR / "tail_risk.png"))),
]:
    try:
        fn()
        plt.close("all")
        print(f"saved: {name}.png")
    except Exception:
        traceback.print_exc()