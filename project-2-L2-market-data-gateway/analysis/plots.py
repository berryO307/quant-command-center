"""
Quant-grade latency visualizations.

Each function returns the (fig, ax) tuple so the notebook can compose them.
All plots respect the project style (style.apply_style()) and avoid clutter.

Key plots:
  - Histogram with linear bulk + log-scale tail (CCDF)
  - Time-series with rolling percentile bands
  - Density heatmap with percentile overlay (Jane Street style)
  - Stage-by-stage CCDF comparison
  - Burst timeline
  - Autocorrelation
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from . import metrics
from .style import PALETTE, STAGE_COLORS, annotate_value


# ── 1. Histogram + log-scale tail ───────────────────────────────────────────

def plot_histogram_with_tail(
    x:        np.ndarray | pd.Series,
    title:    str = "Latency distribution",
    unit:     str = "µs",
    color:    str = PALETTE["ink"],
    bins:     int = 100,
):
    """
    Two-panel plot:
      (left)  linear histogram of the bulk — clipped at p99.5 for readability
      (right) CCDF on log-log axes — exposes the tail shape
    """
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]

    fig, (ax_hist, ax_tail) = plt.subplots(
        1, 2, figsize=(11, 4),
        gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.25},
    )

    # ── Left: bulk histogram ────────────────────────────────────────────────
    clip = np.percentile(a, 99.5)

    ax_hist.hist(
        a[a <= clip],
        bins=bins,
        color=color,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.9,
    )

    ax_hist.set_xlabel(f"Latency ({unit})")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title(f"{title} — bulk (≤ p99.5)", loc="left")

    median = np.median(a)
    p99    = np.percentile(a, 99)

    ax_hist.axvline(
        median,
        color=PALETTE["ink"],
        linewidth=1.0,
        linestyle="--",
        alpha=0.85,
    )

    ax_hist.axvline(
        p99,
        color=PALETTE["tail"],
        linewidth=1.0,
        linestyle="--",
        alpha=0.85,
    )

    ax_hist.text(
        median,
        ax_hist.get_ylim()[1] * 0.92,
        f" p50: {median:.1f}",
        color=PALETTE["ink"],
        fontsize=8.5,
        va="top",
    )

    ax_hist.text(
        p99,
        ax_hist.get_ylim()[1] * 0.82,
        f" p99: {p99:.1f}",
        color=PALETTE["tail"],
        fontsize=8.5,
        va="top",
    )

    # ── Right: CCDF ─────────────────────────────────────────────────────────
    vals, surv = metrics.ccdf(a)

    mask = surv < 0.5

    ax_tail.loglog(
        vals[mask],
        surv[mask],
        color=color,
        linewidth=1.6,
    )

    ax_tail.set_xlabel(f"Latency ({unit})")
    ax_tail.set_ylabel("P(X > x)")
    ax_tail.set_title("Survival function (log-log)", loc="left")

    for q, label in [
        (0.01, "p99"),
        (0.001, "p99.9"),
        (0.0001, "p99.99"),
    ]:
        ax_tail.axhline(
            q,
            color=PALETTE["muted"],
            linewidth=0.5,
            linestyle=":",
            zorder=1,
        )

        ax_tail.text(
            ax_tail.get_xlim()[1],
            q,
            f" {label} ",
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=PALETTE["muted"],
        )

    return fig, (ax_hist, ax_tail)


# ── 2. Time-series with rolling percentile bands ────────────────────────────

def plot_timeseries_percentiles(
    x:           np.ndarray | pd.Series,
    window:      int = 1000,
    quantiles:   Iterable[float] = (0.50, 0.99, 0.999),
    title:       str = "Latency over time",
    unit:        str = "µs",
):
    """
    Rolling-window percentile bands.
    """
    rp = metrics.rolling_percentiles(
        x,
        window=window,
        quantiles=quantiles,
    )

    fig, ax = plt.subplots(figsize=(11, 4))

    color_map = {
        "p50":   PALETTE["ink"],
        "p99":   PALETTE["tail"],
        "p99_9": PALETTE["alert"],
    }

    for col in rp.columns:
        ax.plot(
            rp.index,
            rp[col],
            color=color_map.get(col, PALETTE["muted"]),
            linewidth=1.1,
            label=col.replace("_", "."),
            alpha=0.9,
        )

    ax.set_xlabel("Sample index")
    ax.set_ylabel(f"Latency ({unit})")

    ax.set_title(
        f"{title} — rolling window = {window:,}",
        loc="left",
    )

    ax.legend(loc="upper right", ncol=len(rp.columns))

    return fig, ax


# ── 3. Density heatmap ──────────────────────────────────────────────────────

def plot_density_heatmap(
    x:              np.ndarray | pd.Series,
    timestamps:     np.ndarray | pd.Series | None = None,
    n_time_bins:    int   = 200,
    n_value_bins:   int   = 80,
    log_y:          bool  = False,
    overlay_pct:    float = 50.0,
    title:          str   = "Latency density over time",
    unit:           str   = "µs",
    cmap:           str   = "Blues",
):
    """
    2-D density plot.
    """
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    n = a.size

    if timestamps is not None:
        t = np.asarray(timestamps, dtype=np.float64)
        t = t[:n]
        x_label = "Time"
    else:
        t = np.arange(n, dtype=np.float64)
        x_label = "Sample index"

    y_lo = max(np.percentile(a, 0.1), 1e-9)
    y_hi = np.percentile(a, 99.5)

    if log_y:
        y_edges = np.geomspace(y_lo, y_hi, n_value_bins + 1)
    else:
        y_edges = np.linspace(y_lo, y_hi, n_value_bins + 1)

    x_edges = np.linspace(t.min(), t.max(), n_time_bins + 1)

    H, xe, ye = np.histogram2d(
        t,
        a,
        bins=[x_edges, y_edges],
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    H_plot = H.T
    H_plot_masked = np.ma.masked_where(H_plot == 0, H_plot)

    mesh = ax.pcolormesh(
        xe,
        ye,
        H_plot_masked,
        cmap=cmap,
        norm=LogNorm(vmin=1, vmax=max(H_plot.max(), 2)),
        shading="auto",
        rasterized=True,
    )

    bin_idx = np.clip(
        np.searchsorted(x_edges, t, side="right") - 1,
        0,
        n_time_bins - 1,
    )

    overlay = np.full(n_time_bins, np.nan)

    for i in range(n_time_bins):
        cell = a[bin_idx == i]

        if cell.size > 0:
            overlay[i] = np.percentile(cell, overlay_pct)

    bin_centers = (x_edges[:-1] + x_edges[1:]) / 2

    ax.plot(
        bin_centers,
        overlay,
        color=PALETTE["tail"],
        linewidth=1.3,
        label=f"p{overlay_pct:g}",
    )

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Latency ({unit})")
    ax.set_title(title, loc="left")

    ax.legend(loc="upper right")

    cbar = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.04)

    cbar.set_label("Samples per cell (log scale)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    return fig, ax


# ── 4. Stage comparison ─────────────────────────────────────────────────────

def plot_stage_comparison(
    samples:   dict[str, np.ndarray | pd.Series],
    title:     str = "Stage-by-stage latency comparison",
    unit:      str = "µs",
):
    """
    Multi-stage CCDF comparison.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, arr in samples.items():
        vals, surv = metrics.ccdf(arr)

        mask = surv < 0.5

        ax.loglog(
            vals[mask],
            surv[mask],
            color=STAGE_COLORS.get(name, PALETTE["muted"]),
            linewidth=1.6,
            label=name,
            alpha=0.9,
        )

    for q, label in [
        (0.01, "p99"),
        (0.001, "p99.9"),
        (0.0001, "p99.99"),
    ]:
        ax.axhline(
            q,
            color=PALETTE["muted"],
            linewidth=0.5,
            linestyle=":",
            zorder=1,
        )

        ax.text(
            ax.get_xlim()[1],
            q,
            f" {label} ",
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=PALETTE["muted"],
        )

    ax.set_xlabel(f"Latency ({unit})")
    ax.set_ylabel("P(X > x)")
    ax.set_title(title, loc="left")

    ax.legend(loc="lower left")

    return fig, ax


# ── 5. Burst timeline ───────────────────────────────────────────────────────

def plot_burst_timeline(
    x:                np.ndarray | pd.Series,
    bursts:           list[metrics.BurstEvent],
    title:            str  = "Latency with burst events highlighted",
    unit:             str  = "µs",
    threshold_pct:    float = 99.0,
):
    """
    Time series with burst regions highlighted.
    """
    a = np.asarray(x, dtype=np.float64)

    threshold = np.percentile(a, threshold_pct)

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(
        a,
        color=PALETTE["muted"],
        linewidth=0.4,
        alpha=0.55,
    )

    for b in bursts:
        ax.axvspan(
            b.start_idx,
            b.end_idx,
            color=PALETTE["density"],
            alpha=0.18,
            zorder=1,
        )

    ax.axhline(
        threshold,
        color=PALETTE["tail"],
        linewidth=1.0,
        linestyle="--",
        alpha=0.9,
    )

    ax.set_xlabel("Sample index")
    ax.set_ylabel(f"Latency ({unit})")

    ax.set_title(
        f"{title} — {len(bursts)} burst(s) detected "
        f"(threshold > p{threshold_pct:g})",
        loc="left",
    )

    return fig, ax


# ── 6. Autocorrelation ──────────────────────────────────────────────────────

def plot_autocorrelation(
    x:        np.ndarray | pd.Series,
    max_lag:  int = 200,
    title:    str = "Latency autocorrelation",
):
    """
    Autocorrelation stems plot.
    """
    ac = metrics.autocorr(x, max_lag=max_lag)

    fig, ax = plt.subplots(figsize=(11, 3.5))

    lags = np.arange(len(ac))

    ax.vlines(
        lags,
        0,
        ac,
        color=PALETTE["ink"],
        linewidth=0.9,
    )

    ax.axhline(
        0,
        color=PALETTE["muted"],
        linewidth=0.5,
    )

    n = len(np.asarray(x))

    if n > 0:
        ci = 1.96 / np.sqrt(n)

        ax.axhspan(
            -ci,
            ci,
            color=PALETTE["muted"],
            alpha=0.25,
            zorder=0,
        )

    ax.set_xlabel("Lag (samples)")
    ax.set_ylabel("Correlation")
    ax.set_title(title, loc="left")

    return fig, ax

# ── 7. NIC micro-burst diagnostic ───────────────────────────────────────────

def plot_nic_burst_diagnostic(
    t2_tsc:     np.ndarray,
    queue_us:   np.ndarray,
    cpu_ghz:    float = 3.2,
    title:      str   = "NIC micro-burst diagnostic",
    unit:       str   = "µs",
    n_bins:     int   = 40,
) -> tuple:
    """
    Diagnose whether NIC micro-bursts (GRO coalescing / bursty arrival) are
    the root cause of queue-transit tail latency.

    Mechanical hypothesis
    ─────────────────────
    When the NIC delivers packets in tight clusters (inter-arrival < ~200 µs),
    the producer can push several ticks onto the SPSC ring before the consumer
    drains the previous ones. Each queued tick then experiences a growing
    wake-up delay — the first tick in the burst sees normal latency, the last
    tick waits for all predecessors to be processed. On a log-log scatter this
    appears as a rising cloud of queue_us values at small inter-arrival times.

    A C-state wakeup signature is the *opposite*: latency spikes appear at
    LARGE inter-arrival times (the CPU entered a sleep state during a quiet
    period), not small ones.

    Layout (three panels)
    ─────────────────────
    Top-left  : 2-D log-density heatmap of inter-arrival_us vs queue_us.
                Both axes log-scaled. A rising ridge towards the bottom-left
                (small inter-arrival → high latency) confirms burst-driven tails.

    Top-right : Conditional p50 / p99 ribbon binned by inter-arrival.
                Readable at-a-glance: does queue p99 rise as inter-arrival falls?

    Bottom    : Inter-arrival CCDF.
                Reference for how often bursts actually occur in this capture.
                Mark the 1 ms boundary — arrivals below this are almost
                certainly GRO-batched (kernel coalesced ≥2 frames into one wakeup).
    """
    ia_us  = metrics.interarrival_us(t2_tsc, cpu_ghz=cpu_ghz)
    q      = np.asarray(queue_us, dtype=np.float64)

    # Align lengths (t2_tsc and queue_us must have same length)
    n = min(len(ia_us), len(q))
    ia_us, q = ia_us[:n], q[:n]

    fig = plt.figure(figsize=(13, 8))
    gs  = fig.add_gridspec(
        2, 2,
        height_ratios=[1.4, 1.0],
        hspace=0.38,
        wspace=0.32,
    )
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_cond = fig.add_subplot(gs[0, 1])
    ax_ccdf = fig.add_subplot(gs[1, :])

    # ── Panel 1: 2-D log-log density heatmap ───────────────────────────────
    ia_lo   = max(np.percentile(ia_us, 0.5), 0.05)
    ia_hi   = np.percentile(ia_us, 99.5)
    q_lo    = max(np.percentile(q,    0.5), 0.05)
    q_hi    = np.percentile(q,    99.5)

    x_edges = np.geomspace(ia_lo, ia_hi, 60)
    y_edges = np.geomspace(q_lo,  q_hi,  50)

    H, xe, ye = np.histogram2d(ia_us, q, bins=[x_edges, y_edges])

    from matplotlib.colors import LogNorm
    mesh = ax_heat.pcolormesh(
        xe, ye, H.T,
        cmap="Blues",
        norm=LogNorm(vmin=1, vmax=max(H.max(), 2)),
        shading="auto",
        rasterized=True,
    )
    ax_heat.set_xscale("log")
    ax_heat.set_yscale("log")
    ax_heat.set_xlabel(f"Inter-arrival ({unit})")
    ax_heat.set_ylabel(f"Queue transit ({unit})")
    ax_heat.set_title("Density: inter-arrival vs queue transit", loc="left")

    # 1 ms GRO boundary marker
    ax_heat.axvline(1000, color=PALETTE["alert"], linewidth=0.9,
                    linestyle="--", alpha=0.85, zorder=3)
    ax_heat.text(1000, q_hi * 0.6, " 1 ms\nGRO\nboundary",
                 color=PALETTE["alert"], fontsize=7.5, va="top")

    cb = fig.colorbar(mesh, ax=ax_heat, pad=0.01, fraction=0.045)
    cb.set_label("Samples (log)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # ── Panel 2: Conditional p50 / p99 ribbon ──────────────────────────────
    cond = metrics.conditional_percentiles(
        ia_us, q, n_bins=n_bins, log_x=True, quantiles=(0.50, 0.99),
    )

    if not cond.empty:
        ax_cond.fill_between(
            cond["bin_center"],
            cond["p50"],
            cond["p99"],
            color=PALETTE["density"],
            alpha=0.25,
            label="p50–p99 band",
        )
        ax_cond.plot(
            cond["bin_center"], cond["p50"],
            color=PALETTE["ink"],
            linewidth=1.5,
            label="p50",
        )
        ax_cond.plot(
            cond["bin_center"], cond["p99"],
            color=PALETTE["tail"],
            linewidth=1.5,
            label="p99",
        )

    ax_cond.set_xscale("log")
    ax_cond.set_yscale("log")
    ax_cond.set_xlabel(f"Inter-arrival ({unit})")
    ax_cond.set_ylabel(f"Queue transit ({unit})")
    ax_cond.set_title("Conditional queue latency by arrival gap", loc="left")
    ax_cond.axvline(1000, color=PALETTE["alert"], linewidth=0.9,
                    linestyle="--", alpha=0.75, zorder=3)
    ax_cond.legend(loc="upper right", fontsize=8)

    # Slope annotation — is p99 rising as inter-arrival falls?
    if len(cond) >= 4:
        x_vals = cond["bin_center"].values
        y_vals = cond["p99"].values
        valid  = np.isfinite(np.log(x_vals)) & np.isfinite(np.log(y_vals))
        if valid.sum() >= 4:
            slope, _ = np.polyfit(np.log(x_vals[valid]), np.log(y_vals[valid]), 1)
            sign     = "↑ burst-driven" if slope < -0.05 else "↔ flat (not burst)"
            ax_cond.annotate(
                f"slope ≈ {slope:.2f}  {sign}",
                xy=(0.04, 0.06), xycoords="axes fraction",
                fontsize=8, color=PALETTE["tail"],
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=PALETTE["paper"],
                          edgecolor=PALETTE["grid"], alpha=0.9),
            )

    # ── Panel 3: Inter-arrival CCDF ─────────────────────────────────────────
    ia_sorted, ia_surv = metrics.ccdf(ia_us)

    ax_ccdf.loglog(
        ia_sorted, ia_surv,
        color=PALETTE["ink"],
        linewidth=1.4,
    )
    ax_ccdf.axvline(1000, color=PALETTE["alert"], linewidth=0.9,
                    linestyle="--", alpha=0.85, zorder=3,
                    label="1 ms (GRO boundary)")

    # Annotate fraction of ticks that arrived in < 1 ms gap
    frac_burst = float((ia_us < 1000).mean()) * 100
    ax_ccdf.annotate(
        f"{frac_burst:.1f}% of ticks arrived\nwithin 1 ms of the prior tick",
        xy=(1000, float(np.interp(1000, ia_sorted, ia_surv))),
        xytext=(40, 12), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=PALETTE["alert"], lw=0.9),
        fontsize=8.5, color=PALETTE["alert"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["paper"],
                  edgecolor=PALETTE["grid"], alpha=0.9),
    )

    for q_ref, label in [(0.01, "p99"), (0.001, "p99.9")]:
        ax_ccdf.axhline(q_ref, color=PALETTE["muted"],
                        linewidth=0.5, linestyle=":", zorder=1)
        ax_ccdf.text(ax_ccdf.get_xlim()[1], q_ref, f" {label}",
                     ha="right", va="bottom", fontsize=7.5, color=PALETTE["muted"])

    ax_ccdf.set_xlabel(f"Inter-arrival time ({unit})")
    ax_ccdf.set_ylabel("P(IA > x)")
    ax_ccdf.set_title("Inter-arrival time survival function — burst frequency reference", loc="left")
    ax_ccdf.legend(loc="lower left", fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="semibold", x=0.02, ha="left")
    return fig, (ax_heat, ax_cond, ax_ccdf)