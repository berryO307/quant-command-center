import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
import pandas as pd

from src.dynamics import REGIME_STATES
from src.environment import ACTIONS, State

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGIME_COLORS = {
    "normal":   "#d4e6f1",
    "peak":     "#fde8d8",
    "volatile": "#e8d5f0",
}

ACTION_CMAP = mcolors.ListedColormap(["#e74c3c", "#bdc3c7", "#27ae60"])
ACTION_NORM = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)
ACTION_LABELS = {-1: "Discharge", 0: "Hold", 1: "Charge"}

_ROW_KEYS   = [("low", "low"), ("low", "high"), ("high", "low"), ("high", "high")]
_ROW_LABELS = ["P=Low,  G=Low", "P=Low,  G=High", "P=High, G=Low", "P=High, G=High"]


# ---------------------------------------------------------------------------
# Plot 1 — Regime-colored cumulative PnL
# ---------------------------------------------------------------------------

def plot_regime_pnl(
    trajectory: pd.DataFrame,
    save_path:  str,
    show_raw:   bool = False,
) -> None:
    """
    Regime-banded PnL plot.

    show_raw=False (default): plots alpha_pnl — battery contribution only.
    show_raw=True            : plots cumulative_reward (raw active reward).

    trajectory must contain columns: t, regime, alpha_pnl, cumulative_reward.
    """
    col    = "cumulative_reward" if show_raw else "alpha_pnl"
    title  = ("Cumulative PnL with Market Regime Background" if show_raw
               else "Cumulative Alpha (Battery Contribution)")
    ylabel = "Cumulative Reward" if show_raw else "Cumulative Alpha"

    df      = trajectory.reset_index(drop=True)
    t       = df["t"].to_numpy()
    pnl     = df[col].to_numpy()
    regimes = df["regime"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))

    # Background bands: scan for consecutive runs of same regime
    i = 0
    while i < len(regimes):
        r = regimes[i]
        j = i + 1
        while j < len(regimes) and regimes[j] == r:
            j += 1
        ax.axvspan(t[i], t[j - 1], color=REGIME_COLORS[r], alpha=0.45, linewidth=0)
        i = j

    ax.plot(t, pnl, color="#2c3e50", linewidth=1.8, zorder=5)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

    patches = [
        mpatches.Patch(color=REGIME_COLORS[r], label=r.capitalize())
        for r in REGIME_STATES
    ]
    ax.legend(handles=patches, title="Regime", loc="upper left", framealpha=0.9)

    ax.set_xlabel("Time Step (t)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Policy heatmap
# ---------------------------------------------------------------------------

def plot_policy_heatmap(policy: dict, save_path: str) -> None:
    """
    One subplot per regime. Rows: (price, gen) combinations. Columns: battery level.
    Cell color and label: the action the policy selects.
    """
    battery_levels = sorted({s.battery_level for s in policy})
    n_bat = len(battery_levels)

    fig, axes = plt.subplots(1, len(REGIME_STATES), figsize=(14, 4), sharey=True,
                             constrained_layout=True)

    for ax, regime in zip(axes, REGIME_STATES):
        matrix = np.zeros((len(_ROW_KEYS), n_bat))
        for i, (price, gen) in enumerate(_ROW_KEYS):
            for j, b in enumerate(battery_levels):
                matrix[i, j] = policy[State(b, price, gen, regime)]

        ax.imshow(matrix, cmap=ACTION_CMAP, norm=ACTION_NORM, aspect="auto")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j, i, ACTION_LABELS[int(matrix[i, j])],
                    ha="center", va="center", fontsize=7.5, color="#1a1a1a"
                )

        ax.set_xticks(range(n_bat))
        ax.set_xticklabels(battery_levels)
        ax.set_yticks(range(len(_ROW_KEYS)))
        ax.set_yticklabels(_ROW_LABELS, fontsize=8.5)
        ax.set_xlabel("Battery Level")
        ax.set_title(f"Regime: {regime.capitalize()}", fontweight="bold")

    cbar = fig.colorbar(
        ScalarMappable(norm=ACTION_NORM, cmap=ACTION_CMAP),
        ax=axes,
        orientation="vertical",
        fraction=0.015,
        pad=0.04,
    )
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["Discharge", "Hold", "Charge"])

    fig.suptitle("Converged MDP Policy — Action by State", fontsize=13, fontweight="bold")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 3 — Action distribution histogram
# ---------------------------------------------------------------------------

def plot_action_histogram(results: dict, save_path: str, title: str = "Action Distribution") -> None:
    """
    Grouped bar chart of action counts.
    results: dict[label -> SimulationSummary]
    """
    actions = [-1, 0, 1]
    keys    = list(results.keys())
    n       = len(keys)
    x       = np.arange(len(actions))
    width   = 0.65 / n

    fig, ax = plt.subplots(figsize=(9, 5))

    palette = ["#2980b9", "#e67e22", "#8e44ad", "#27ae60"]
    for i, key in enumerate(keys):
        counts = [results[key].action_counts[a] for a in actions]
        ax.bar(
            x + i * width,
            counts,
            width,
            label=str(key),
            color=palette[i % len(palette)],
            edgecolor="white",
        )

    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels([ACTION_LABELS[a] for a in actions])
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 4 — Inventory sensitivity
# ---------------------------------------------------------------------------

def plot_inventory_sensitivity(results: dict, save_path: str) -> None:
    """
    Dual-axis chart showing the tradeoff introduced by inventory penalty.
    Left  axis: cumulative alpha (reward)
    Right axis: hold action count
    results: dict[inventory_coeff (float) -> SimulationSummary]
    """
    keys        = sorted(results.keys())
    alphas      = [results[k].alpha_pnl        for k in keys]
    hold_counts = [results[k].action_counts[0] for k in keys]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    c_alpha = "#2980b9"
    c_hold  = "#e74c3c"

    ax1.plot(keys, alphas,      color=c_alpha, marker="o", linewidth=2, label="Alpha PnL")
    ax1.set_xlabel("inventory_coeff")
    ax1.set_ylabel("Cumulative Alpha", color=c_alpha)
    ax1.tick_params(axis="y", labelcolor=c_alpha)
    ax1.set_xticks(keys)

    ax2 = ax1.twinx()
    ax2.plot(keys, hold_counts, color=c_hold, marker="s", linewidth=2,
             linestyle="--", label="Hold count")
    ax2.set_ylabel("Hold Action Count", color=c_hold)
    ax2.tick_params(axis="y", labelcolor=c_hold)

    lines  = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", framealpha=0.9)

    ax1.set_title("Inventory Sensitivity: Alpha vs Hold Behaviour")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
