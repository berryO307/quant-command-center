import os

import matplotlib.pyplot as plt

from src.environment import MDPEnvironment, State
from src.solver import value_iteration
from src.simulator import Simulator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INITIAL_STATE = State(2, "low", "low", "normal")
SEED          = 42
HORIZON       = 100
COST_COEFF    = 0.1
SAVE_DATA     = False

PLOTS_DIR = "results/plots"
DATA_DIR  = "results/data"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Train two policies
    # ------------------------------------------------------------------

    print("Training policy (no transaction cost)...")
    _, policy_no_cost = value_iteration(
        env=MDPEnvironment(State(0, "low", "low", "normal"), cost_coeff=0.0)
    )

    print("Training policy (with transaction cost)...")
    _, policy_with_cost = value_iteration(
        env=MDPEnvironment(State(0, "low", "low", "normal"), cost_coeff=COST_COEFF)
    )

    # ------------------------------------------------------------------
    # Simulate three scenarios — same seed, same initial state
    # ------------------------------------------------------------------

    # Curve 1: no-cost policy, no-cost env  (clean baseline)
    env_A = MDPEnvironment(INITIAL_STATE, seed=SEED, cost_coeff=0.0)
    result_A = Simulator(env_A, policy_no_cost, HORIZON).run_policy()

    # Curve 2: no-cost policy, cost env  (naive — ignores friction at training)
    env_B = MDPEnvironment(INITIAL_STATE, seed=SEED, cost_coeff=COST_COEFF)
    result_B = Simulator(env_B, policy_no_cost, HORIZON).run_policy()

    # Curve 3: cost-aware policy, cost env  (correct)
    env_C = MDPEnvironment(INITIAL_STATE, seed=SEED, cost_coeff=COST_COEFF)
    result_C = Simulator(env_C, policy_with_cost, HORIZON).run_policy()

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def _mdp(r):
        return r.baseline_pnl + r.alpha_pnl

    print("\n===== RESULTS =====\n")
    header = f"{'Scenario':<42}  {'Baseline':>10}  {'MDP PnL':>10}  {'Alpha':>10}"
    print(header)
    print("-" * len(header))

    rows = [
        ("No-cost policy,   no-cost env  (baseline)", result_A),
        ("No-cost policy,   cost env     (naive)",    result_B),
        ("Cost-aware policy, cost env    (fix)",      result_C),
    ]
    for label, r in rows:
        print(f"{label:<42}  {r.baseline_pnl:>10.4f}  {_mdp(r):>10.4f}  {r.alpha_pnl:>10.4f}")

    print("\nAction counts:")
    for label, r in rows:
        print(f"  {label:<42}: {r.action_counts}")

    # ------------------------------------------------------------------
    # Plot — alpha PnL comparison
    # ------------------------------------------------------------------

    os.makedirs(PLOTS_DIR, exist_ok=True)

    t = result_A.trajectory["t"]

    plt.figure(figsize=(10, 5))
    plt.plot(t, result_A.trajectory["alpha_pnl"],
             label="No-cost policy, no-cost env (baseline)", linewidth=2)
    plt.plot(t, result_B.trajectory["alpha_pnl"],
             label="No-cost policy, cost env (naive)",       linewidth=2, linestyle="--")
    plt.plot(t, result_C.trajectory["alpha_pnl"],
             label="Cost-aware policy, cost env (fix)",      linewidth=2, linestyle=":")
    plt.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    plt.xlabel("Time Step (t)")
    plt.ylabel("Cumulative Alpha")
    plt.title("Battery Alpha: Cost-Aware vs Naive Policy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pnl_comparison.png"), dpi=150)
    plt.close()
    print(f"\nPlot saved to {PLOTS_DIR}/pnl_comparison.png")

    # ------------------------------------------------------------------
    # Optional CSV saving
    # ------------------------------------------------------------------

    if SAVE_DATA:
        os.makedirs(DATA_DIR, exist_ok=True)
        result_A.trajectory.to_csv(os.path.join(DATA_DIR, "no_cost_policy_no_cost_env.csv"),   index=False)
        result_B.trajectory.to_csv(os.path.join(DATA_DIR, "no_cost_policy_with_cost_env.csv"), index=False)
        result_C.trajectory.to_csv(os.path.join(DATA_DIR, "cost_aware_policy_cost_env.csv"),   index=False)
        print(f"Data saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()
