import os

from src.environment import MDPEnvironment, State
from src.solver import value_iteration
from src.simulator import Simulator
from src.plot_results import plot_regime_pnl, plot_policy_heatmap, plot_action_histogram

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED      = 42
HORIZON   = 200
PLOTS_DIR = "results/plots"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Shared: train baseline policy (no transaction cost)
    # ------------------------------------------------------------------
    print("Training baseline policy...")
    _, policy = value_iteration(
        env=MDPEnvironment(State(0, "low", "low", "normal"), cost_coeff=0.0)
    )

    # ------------------------------------------------------------------
    # Plot 1 — Regime-colored PnL
    # ------------------------------------------------------------------
    print("\nGenerating regime PnL plot...")
    env = MDPEnvironment(State(2, "low", "low", "normal"), seed=SEED, cost_coeff=0.0)
    result = Simulator(env, policy, HORIZON).run_policy()
    plot_regime_pnl(
        result.trajectory,
        os.path.join(PLOTS_DIR, "regime_pnl.png"),
    )

    # ------------------------------------------------------------------
    # Plot 2 — Policy heatmap
    # ------------------------------------------------------------------
    print("\nGenerating policy heatmap...")
    plot_policy_heatmap(
        policy,
        os.path.join(PLOTS_DIR, "policy_heatmap.png"),
    )

    # ------------------------------------------------------------------
    # Plot 3 — Action histogram (cost sensitivity, Scenario C)
    # ------------------------------------------------------------------
    print("\nRunning cost-sensitivity simulations for action histogram...")
    cost_results = {}
    for cost in [0.0, 0.1, 0.3]:
        train_env = MDPEnvironment(State(0, "low", "low", "normal"), cost_coeff=cost)
        _, pol    = value_iteration(env=train_env)
        sim_env   = MDPEnvironment(State(2, "low", "low", "normal"), seed=SEED, cost_coeff=cost)
        cost_results[f"cost = {cost:.1f}"] = Simulator(sim_env, pol, HORIZON).run_policy()

    plot_action_histogram(
        cost_results,
        os.path.join(PLOTS_DIR, "action_histogram.png"),
        title="Action Distribution by Transaction Cost Level",
    )

    print(f"\nAll plots written to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
