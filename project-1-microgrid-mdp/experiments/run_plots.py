"""
run_plots.py — generates every figure used in the report.

Companion to run_experiments.py:
  - run_experiments.py produces the terminal report (no files)
  - run_plots.py produces all 5 figures in results/plots/

    PYTHONPATH=. python experiments/run_plots.py

Outputs:
  regime_pnl.png             — cumulative alpha with regime-colored bands
  policy_heatmap.png         — converged action for every state, per regime
  action_histogram.png       — action distribution shift under rising friction
  pnl_comparison.png         — cumulative alpha: upper bound vs naive vs cost-aware
  inventory_sensitivity.png  — alpha and behavior vs L2 inventory penalty
"""

import os

import matplotlib.pyplot as plt

from src.environment import MDPEnvironment, State
from src.solver import value_iteration
from src.simulator import Simulator
from src.plot_results import (
    plot_regime_pnl,
    plot_policy_heatmap,
    plot_action_histogram,
    plot_inventory_sensitivity,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED          = 42
HORIZON       = 200
COST_COEFF    = 0.1
PLOTS_DIR     = "results/plots"
INITIAL_STATE = State(2, "low", "low", "normal")
TRAIN_STATE   = State(0, "low", "low", "normal")

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train(cost_coeff: float, inventory_coeff: float = 0.05):
    env = MDPEnvironment(
        TRAIN_STATE,
        cost_coeff=cost_coeff,
        inventory_coeff=inventory_coeff,
    )
    _, policy = value_iteration(env=env)
    return policy


def simulate(policy, cost_coeff: float, inventory_coeff: float = 0.05,
             initial_state: State = INITIAL_STATE):
    env = MDPEnvironment(
        initial_state,
        seed=SEED,
        cost_coeff=cost_coeff,
        inventory_coeff=inventory_coeff,
    )
    return Simulator(env, policy, HORIZON).run_policy()


# ---------------------------------------------------------------------------
# Plot 1 — Regime-colored cumulative alpha
# ---------------------------------------------------------------------------

def plot_1_regime_pnl(policy_no_cost):
    print("  [1/5] regime_pnl.png — cumulative alpha with regime bands")
    result = simulate(policy_no_cost, cost_coeff=0.0)
    plot_regime_pnl(
        result.trajectory,
        os.path.join(PLOTS_DIR, "regime_pnl.png"),
    )


# ---------------------------------------------------------------------------
# Plot 2 — Policy heatmap
# ---------------------------------------------------------------------------

def plot_2_policy_heatmap(policy_no_cost):
    print("  [2/5] policy_heatmap.png — converged policy action per state")
    plot_policy_heatmap(
        policy_no_cost,
        os.path.join(PLOTS_DIR, "policy_heatmap.png"),
    )


# ---------------------------------------------------------------------------
# Plot 3 — Action histogram across friction levels
# ---------------------------------------------------------------------------

def plot_3_action_histogram():
    print("  [3/5] action_histogram.png — action distribution vs friction")
    cost_results = {}
    for cost in [0.0, 0.1, 0.3]:
        policy = train(cost_coeff=cost)
        cost_results[f"cost = {cost:.1f}"] = simulate(policy, cost_coeff=cost)

    plot_action_histogram(
        cost_results,
        os.path.join(PLOTS_DIR, "action_histogram.png"),
        title="Action Distribution by Transaction Cost Level",
    )


# ---------------------------------------------------------------------------
# Plot 4 — PnL comparison: upper bound vs naive vs cost-aware
# ---------------------------------------------------------------------------

def plot_4_pnl_comparison(policy_no_cost, policy_with_cost):
    print("  [4/5] pnl_comparison.png — upper bound vs naive vs cost-aware")

    r_upper = simulate(policy_no_cost,   cost_coeff=0.0)         # ceiling
    r_naive = simulate(policy_no_cost,   cost_coeff=COST_COEFF)  # naive
    r_aware = simulate(policy_with_cost, cost_coeff=COST_COEFF)  # correct

    t = r_upper.trajectory["t"]

    plt.figure(figsize=(10, 5))
    plt.plot(t, r_upper.trajectory["alpha_pnl"],
             label="Upper bound  (no-cost policy, no-cost env)",
             linewidth=2)
    plt.plot(t, r_naive.trajectory["alpha_pnl"],
             label="Naive        (no-cost policy, cost env)",
             linewidth=2, linestyle="--")
    plt.plot(t, r_aware.trajectory["alpha_pnl"],
             label="Cost-aware   (cost policy,    cost env)",
             linewidth=2, linestyle=":")
    plt.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

    plt.xlabel("Time step $t$")
    plt.ylabel("Cumulative alpha $\\alpha_t$")
    plt.title("Battery Alpha Over Time: Cost-Aware vs Naive Policy")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pnl_comparison.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Plot 5 — Inventory penalty sensitivity
# ---------------------------------------------------------------------------

def plot_5_inventory_sensitivity():
    print("  [5/5] inventory_sensitivity.png — alpha vs L2 penalty strength")

    coeffs = [0.0, 0.02, 0.05, 0.10]
    inv_results = {}

    for ic in coeffs:
        policy = train(cost_coeff=COST_COEFF, inventory_coeff=ic)
        inv_results[ic] = simulate(policy,
                                   cost_coeff=COST_COEFF,
                                   inventory_coeff=ic)

    plot_inventory_sensitivity(
        inv_results,
        os.path.join(PLOTS_DIR, "inventory_sensitivity.png"),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Training canonical policies (reused across plots)...")
    policy_no_cost   = train(cost_coeff=0.0)
    policy_with_cost = train(cost_coeff=COST_COEFF)

    print(f"\nGenerating 5 figures in {PLOTS_DIR}/\n")
    plot_1_regime_pnl(policy_no_cost)
    plot_2_policy_heatmap(policy_no_cost)
    plot_3_action_histogram()
    plot_4_pnl_comparison(policy_no_cost, policy_with_cost)
    plot_5_inventory_sensitivity()

    print(f"\nAll plots written to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
