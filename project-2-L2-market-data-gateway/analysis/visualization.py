"""
Design: a single class that owns the figure and dispatches plot methods.
Each method produces a single coherent narrative chart. The dashboard
methods compose these into multi-panel layouts using GridSpec.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from pathlib import Path

from .style import PALETTE, DENSITY_CMAP, STAGE_COLORS, annotate_value
from .metrics import (
    summary_table, rolling_percentiles, detect_regime_shifts,
    ccdf, RegimeShift,
)


class TradingStoryPlotter:
    """
    Compose Jane Street-style latency dashboards from gateway telemetry.

    Usage:
        plotter = TradingStoryPlotter(latency_df)
        plotter.plot_latency_regime_dashboard(save_to="output/dashboard.png")
        plotter.plot_tail_risk(save_to="output/tail.png")
    """

    def __init__(self, df: pd.DataFrame, instrument: str = "BTCUSDT", venue: str = "Bybit"):
        self.df = df
        self.instrument = instrument
        self.venue = venue

    # Hero dashboard — replaces your 2x2 grid with a storytelling layout

    def plot_latency_regime_dashboard(self, save_to: Optional[str] = None):
        """
        Storytelling layout:
          ┌──────────────────────────────────────────────────────┐
          │ Hero: latency density heatmap with p50 overlay       │
          │ (full width, dominant)                               │
          ├────────────────────────┬─────────────────────────────┤
          │ Parse distribution     │ Queue distribution          │
          │ (with percentile       │ (with percentile callouts)  │
          │  callouts)             │                             │
          ├────────────────────────┴─────────────────────────────┤
          │ Time-series with auto-detected regime shifts         │
          │ (full width — narrative payoff)                      │
          └──────────────────────────────────────────────────────┘
        """
        fig = plt.figure(figsize=(15, 11))
        gs = gridspec.GridSpec(
            3, 2,
            figure=fig,
            height_ratios=[1.4, 1.0, 1.1],
            hspace=0.45,
            wspace=0.18,
        )

        # ── Row 0: hero density ────────────────────────────────────────────
        ax_hero = fig.add_subplot(gs[0, :])
        self._draw_density_heatmap(ax_hero, self.df["queue_transit_us"])

        # ── Row 1: distributions ───────────────────────────────────────────
        ax_parse = fig.add_subplot(gs[1, 0])
        self._draw_distribution(ax_parse, self.df["parse_us"], "Parse Latency")

        ax_queue = fig.add_subplot(gs[1, 1])
        self._draw_distribution(ax_queue, self.df["queue_transit_us"], "Queue Transit")

        # ── Row 2: regime timeline ─────────────────────────────────────────
        ax_time = fig.add_subplot(gs[2, :])
        self._draw_regime_timeline(ax_time, self.df["queue_transit_us"])

        # ── Suptitle ───────────────────────────────────────────────────────
        fig.suptitle(
            f"Latency Characterization · {self.venue} {self.instrument} L2 Gateway",
            x=0.04, y=0.985, ha="left",
            fontsize=15, weight="semibold", color=PALETTE["ink"],
        )
        fig.text(
            0.04, 0.957,
            f"n = {len(self.df):,} ticks  ·  measured via rdtscp on calibrated TSC",
            fontsize=10, color=PALETTE["muted"], ha="left",
        )
    
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)

        if save_to:
            fig.savefig(save_to, dpi=200)
        return fig

    # Tail risk view — separate dashboard focused on CCDF

    def plot_tail_risk(self, save_to: Optional[str] = None):
        """
        Two-panel tail analysis:
          left:  CCDF on log-log axes — bug-fixed version with reference lines
          right: percentile ladder bar chart — explicit numbers
        """
        fig = plt.figure(figsize=(13, 5.5))
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.4, 1.0], wspace=0.25)

        ax_ccdf = fig.add_subplot(gs[0, 0])
        self._draw_ccdf(ax_ccdf, {
            "Parse": self.df["parse_us"].values,
            "Queue": self.df["queue_transit_us"].values,
            "End-to-end": self.df["end_to_end_us"].values,
        })

        ax_ladder = fig.add_subplot(gs[0, 1])
        self._draw_percentile_ladder(ax_ladder)

        fig.suptitle(
            f"Tail Risk Analysis · {self.venue} {self.instrument}",
            x=0.04, y=0.97, ha="left",
            fontsize=14, weight="semibold", color=PALETTE["ink"],
        )

        if save_to:
            fig.savefig(save_to, dpi=200)
        return fig

    # Specialized diagnostic view for investigating latency spikes

    def plot_stall_matrix(self, cpu_ghz: float = 3.2, save_to: Optional[str] = None):
            fig, ax = plt.subplots(figsize=(9, 7))

            t2_tsc       = self.df["t2_tsc"].values.astype(np.float64)
            queue_cycles = self.df["queue_cycles"].values.astype(np.float64)
            t1_tsc       = t2_tsc - queue_cycles

            # np.diff drops row 0 — correct. prepend=t1_tsc[0] was making delta[0]=0 always.
            delta_t1_cycles  = np.diff(t1_tsc)
            queue_transit_all = self.df["queue_transit_us"].values[1:]

            valid_mask       = (delta_t1_cycles > -32_000_000) & (queue_transit_all > 0)
            delta_t1_us      = (delta_t1_cycles[valid_mask] / cpu_ghz) / 1000.0
            queue_transit_us = queue_transit_all[valid_mask]

            if len(queue_transit_us) == 0:
                print("[plot_stall_matrix] no valid data")
                return fig, ax

            ax.scatter(delta_t1_us, queue_transit_us, s=6, alpha=0.35, color=PALETTE["density"], edgecolors="none")
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_yscale("log")

            p99_transit      = np.percentile(queue_transit_us, 99)
            positive         = delta_t1_us[delta_t1_us > 0]
            median_arrival   = np.median(positive) if len(positive) > 0 else 1.0

            ax.axhline(p99_transit, color=PALETTE["alert"], linestyle="--", alpha=0.7, linewidth=1.2)
            ax.axvline(median_arrival, color=PALETTE["ink"], linestyle=":", alpha=0.5, linewidth=1.2)

            # Use data bounds — get_xlim() before draw() is unreliable
            x_lo = float(np.percentile(delta_t1_us, 0.5))
            x_hi = float(np.percentile(delta_t1_us, 99.5))

            annotate_value(ax, x_hi, p99_transit, f"p99 Transit ({p99_transit:.1f} µs)", color=PALETTE["alert"])
            ax.text(x_lo, p99_transit * 1.5, "Consumer Stalls\n(OS Preemption)",
                    color=PALETTE["alert"], fontsize=9, ha="left", va="bottom", clip_on=True)
            ax.text(median_arrival * 2.0, p99_transit * 1.5, "Producer / NIC Stalls\n(mmap page faults / GRO)",
                    color=PALETTE["tail"], fontsize=9, ha="left", va="bottom", clip_on=True)

            ax.set_xlabel(r"Producer Inter-arrival Time $\Delta t_1$ (µs, symlog)")
            ax.set_ylabel(r"Queue Transit Latency $t_2 - t_1$ (µs, log)")
            ax.set_title("Stall Matrix: Inter-arrival vs. Queue Transit", loc="left")

            try:
                from matplotlib.ticker import SymmetricalLogLocator
                ax.xaxis.set_major_locator(SymmetricalLogLocator(linthresh=1.0, base=10))
            except Exception:
                pass

            if save_to:
                Path(save_to).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_to, dpi=200)

            return fig, ax

    # Drawing primitives — single chart per method

    def _draw_density_heatmap(self, ax, series: pd.Series):
        """
        2D histogram: time on x, latency value on y, density coloured.
        Median overlaid in amber. This is the Jane Street signature chart.
        """
        a = series.values
        n = len(a)
        t = np.arange(n)

        # Clip y range at p99.5 so the bulk is visible — extreme outliers
        # would otherwise compress the meaningful range to a single row
        y_lo = max(np.percentile(a, 0.1), 0.1)
        y_hi = np.percentile(a, 99.5)

        # Log-spaced y bins reveal both the dense bulk and the upper tail
        n_y_bins = 80
        n_x_bins = 200
        y_edges = np.geomspace(y_lo, y_hi, n_y_bins + 1)
        x_edges = np.linspace(0, n, n_x_bins + 1)

        H, _, _ = np.histogram2d(t, a, bins=[x_edges, y_edges])
        H_masked = np.ma.masked_where(H.T == 0, H.T)

        mesh = ax.pcolormesh(
            x_edges, y_edges, H_masked,
            cmap=DENSITY_CMAP,
            norm=LogNorm(vmin=1, vmax=max(H.max(), 2)),
            shading="auto",
            rasterized=True,
        )

        # Per-bin median — recompute properly per time bucket
        bin_idx = np.clip(np.searchsorted(x_edges, t, side="right") - 1, 0, n_x_bins - 1)
        median_per_bin = np.full(n_x_bins, np.nan)
        for i in range(n_x_bins):
            cell = a[bin_idx == i]
            if len(cell) > 0:
                median_per_bin[i] = np.median(cell)
        bin_centers = (x_edges[:-1] + x_edges[1:]) / 2

        ax.plot(
            bin_centers, median_per_bin,
            color=PALETTE["tail"], linewidth=2.0,
            label="rolling p50",
            solid_capstyle="round",
        )

        ax.set_yscale("log")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Queue transit (µs)")
        ax.set_title("Queue transit density over capture lifetime", loc="left")
        ax.legend(loc="upper right")
        ax.grid(True, axis="both", alpha=0.3)

        # Slim colorbar — embedded right edge
        cbar = ax.figure.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025)
        cbar.set_label("samples per cell (log scale)", fontsize=8.5)
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_linewidth(0.5)

    def _draw_distribution(self, ax, series: pd.Series, title: str):
        """
        Histogram with inline percentile callouts at p50, p99.
        Clipped at p99.5 so the bulk is legible — the tail goes to the
        CCDF panel where it belongs.
        """
        a = series.values
        clip = np.percentile(a, 99.5)
        bulk = a[a <= clip]

        p50 = np.percentile(a, 50)
        p99 = np.percentile(a, 99)

        ax.hist(
            bulk, bins=70,
            color=PALETTE["density"],
            edgecolor=PALETTE["paper"], linewidth=0.4,
            alpha=0.92,
        )

        # Percentile reference lines, in-place labels
        ymax = ax.get_ylim()[1]
        ax.axvline(p50, color=PALETTE["ink"], linewidth=1.0, linestyle="--", alpha=0.7, zorder=2)
        ax.axvline(p99, color=PALETTE["alert"], linewidth=1.0, linestyle="--", alpha=0.85, zorder=2)

        annotate_value(ax, p50, ymax * 0.92, f"p50  {p50:.1f} µs",
                       color=PALETTE["ink"])
        annotate_value(ax, p99, ymax * 0.78, f"p99  {p99:.1f} µs",
                       color=PALETTE["alert"])

        ax.set_xlabel(f"{title.lower()} (µs)")
        ax.set_ylabel("Frequency")
        ax.set_title(title, loc="left")

    def _draw_regime_timeline(self, ax, series: pd.Series):
        """
        Time-series with rolling p50 + p99 lines. Auto-detected regime
        shifts shown as shaded red bands with arrow annotations.
        """
        roll = rolling_percentiles(series, window=200, quantiles=(0.50, 0.99))
        x = np.arange(len(series))

        # Threshold for shaded "high latency" regions: 100µs above global p50
        threshold = np.percentile(series.values, 50) + 100

        # Shade regions where rolling p99 exceeds the threshold
        above = (roll["p99"] > threshold).values
        if above.any():
            padded = np.concatenate(([False], above, [False]))
            edges = np.diff(padded.astype(np.int8))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            for s, e in zip(starts, ends):
                ax.axvspan(s, e, color=PALETTE["alert"], alpha=0.07, zorder=1)

        # Rolling lines
        ax.plot(x, roll["p50"], color=PALETTE["ink"], linewidth=1.3,
                label="rolling p50", zorder=3)
        ax.plot(x, roll["p99"], color=PALETTE["tail"], linewidth=1.3,
                label="rolling p99", alpha=0.9, zorder=3)

        # Auto-detect regime shifts and annotate with arrows
        shifts = detect_regime_shifts(series, threshold_pct=99.0, min_consecutive=5)
        # Annotate top 3 by peak value to avoid clutter
        top_shifts = sorted(shifts, key=lambda s: -s.peak_value)[:3]

        ymax = max(roll["p99"].max(), threshold * 1.5)
        ax.set_ylim(0, min(ymax, np.percentile(series.values, 99.9)))

        for shift in top_shifts:
            ax.annotate(
                f"⚠  p99 spike\n{shift.peak_value:.0f} µs",
                xy=(shift.peak_idx, min(shift.peak_value, ax.get_ylim()[1] * 0.95)),
                xytext=(shift.peak_idx, ax.get_ylim()[1] * 0.85),
                fontsize=8.5, color=PALETTE["alert"], weight="medium",
                ha="center", va="top",
                arrowprops=dict(
                    arrowstyle="->", color=PALETTE["alert"],
                    lw=0.8, alpha=0.7,
                    connectionstyle="arc3,rad=0",
                ),
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=PALETTE["paper"],
                    edgecolor=PALETTE["alert"],
                    linewidth=0.6, alpha=0.95,
                ),
            )

        ax.set_xlabel("Sample index")
        ax.set_ylabel("Latency (µs)")
        ax.set_title("Latency over time with detected regime shifts", loc="left")
        ax.legend(loc="upper right", ncol=2)

    def _draw_ccdf(self, ax, stages: dict[str, np.ndarray]):
        """
        CCDF on log-log axes. Bug-fixed survival function (no artificial
        drop at max value). Reference lines at p99/p99.9/p99.99.
        """
        for name, arr in stages.items():
            vals, surv = ccdf(arr)
            color = STAGE_COLORS.get(name.lower().replace(" ", "_").replace("-", "_"),
                                      PALETTE["ink"])
            # Plot only the upper tail (survival < 0.5) to avoid the
            # uninteresting bulk and keep the log-log shape readable
            mask = surv < 0.5
            ax.loglog(vals[mask], surv[mask], color=color, linewidth=1.7,
                      label=name, alpha=0.92)

        # Reference percentile bands
        for q, label in [(0.01, "p99"), (0.001, "p99.9"), (0.0001, "p99.99")]:
            ax.axhline(q, color=PALETTE["grid"], linewidth=0.6,
                       linestyle=":", zorder=1)
            ax.text(
                ax.get_xlim()[1], q, f" {label} ",
                ha="right", va="bottom",
                fontsize=8, color=PALETTE["muted"],
                bbox=dict(facecolor=PALETTE["paper"], edgecolor="none",
                          alpha=0.85, pad=1.0),
            )

        ax.set_xlabel("Latency (µs)")
        ax.set_ylabel("P(X > x)")
        ax.set_title("Survival function — log-log tail", loc="left")
        ax.legend(loc="lower left")
        ax.grid(True, which="both", alpha=0.3)

    def _draw_percentile_ladder(self, ax):
        """Horizontal bar chart: explicit percentiles for each stage."""
        percentiles = [50, 90, 95, 99, 99.9]
        stages = ["parse_us", "queue_transit_us", "end_to_end_us"]
        labels = ["Parse", "Queue", "End-to-end"]
        colors = [PALETTE["density"], PALETTE["tail"], PALETTE["ink"]]

        y_pos = np.arange(len(percentiles))
        bar_height = 0.25

        for i, (stage, label, color) in enumerate(zip(stages, labels, colors)):
            vals = [np.percentile(self.df[stage].values, p) for p in percentiles]
            offset = (i - 1) * bar_height
            ax.barh(y_pos + offset, vals, bar_height,
                    color=color, alpha=0.85, label=label,
                    edgecolor=PALETTE["paper"], linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"p{p}" for p in percentiles])
        ax.set_xscale("log")
        ax.set_xlabel("Latency (µs, log scale)")
        ax.set_title("Percentile ladder by stage", loc="left")
        ax.legend(loc="lower right")
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()  # p50 at top, tail at bottom — reads top-down

    # Public summary helper

    def print_summary(self):
        table = summary_table(self.df, ["parse_us", "queue_transit_us", "end_to_end_us"])
        print(table.to_string())