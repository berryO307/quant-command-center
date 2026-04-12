import os

from src.environment import MDPEnvironment, State
from src.solver import value_iteration
from src.simulator import Simulator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED        = 42
HORIZON     = 200
RESULTS_DIR = "results/experiments"
SAVE_DATA   = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _row(label: str, result) -> None:
    print(
        f"  {label:<35} | "
        f"total_reward = {result.total_reward:8.3f} | "
        f"actions = {result.action_counts}"
    )


def _save(results: dict, prefix: str) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for label, result in results.items():
        safe = str(label).replace(".", "_")
        path = os.path.join(RESULTS_DIR, f"{prefix}_{safe}.csv")
        result.trajectory.to_csv(path, index=False)
    print(f"  Data saved to {RESULTS_DIR}/")


# ---------------------------------------------------------------------------
# Scenario A — High Volatility vs Stable Market
#
# One policy, trained without bias on all regimes.
# Simulated from two initial conditions: normal vs volatile.
# Shows how market regime at episode start affects achievable PnL.
# ---------------------------------------------------------------------------

def scenario_A() -> dict:
    _header("SCENARIO A: Stable vs Volatile Market")

    _, policy = value_iteration(
        env=MDPEnvironment(State(0, "low", "low", "normal"), cost_coeff=0.0)
    )

    configs = {
        "stable   (init: normal)":   State(2, "low", "low", "normal"),
        "volatile (init: volatile)": State(2, "low", "low", "volatile"),
    }

    results = {}
    for label, init_state in configs.items():
        env = MDPEnvironment(init_state, seed=SEED, cost_coeff=0.0)
        results[label] = Simulator(env, policy, HORIZON).run_policy()
        _row(label, results[label])

    if SAVE_DATA:
        _save(results, "scenario_A")

    return results


# ---------------------------------------------------------------------------
# Scenario B — Varying Battery Capacity
#
# A fresh policy is trained for each battery size.
# Larger capacity = more arbitrage opportunity between low/high price states.
# Shows the marginal value of additional storage.
# ---------------------------------------------------------------------------

def scenario_B() -> dict:
    _header("SCENARIO B: Varying Battery Capacity (battery_max)")

    battery_sizes = [2, 4, 8]
    results = {}

    for bmax in battery_sizes:
        train_env = MDPEnvironment(
            State(0, "low", "low", "normal"),
            cost_coeff=0.0,
            battery_max=bmax,
        )
        _, policy = value_iteration(env=train_env)

        sim_env = MDPEnvironment(
            State(bmax // 2, "low", "low", "normal"),
            seed=SEED,
            cost_coeff=0.0,
            battery_max=bmax,
        )
        results[bmax] = Simulator(sim_env, policy, HORIZON).run_policy()
        _row(f"battery_max = {bmax}", results[bmax])

    if SAVE_DATA:
        _save(results, "scenario_B")

    return results


# ---------------------------------------------------------------------------
# Scenario C — High vs Zero Transaction Costs
#
# Each policy is trained at its own cost level (cost-aware training).
# Compares achievable PnL as friction increases.
# Also shows the shift in action distribution: high costs deter cycling.
# ---------------------------------------------------------------------------

def scenario_C() -> dict:
    _header("SCENARIO C: Transaction Cost Sensitivity")

    cost_levels = [0.0, 0.1, 0.3]
    results = {}

    for cost in cost_levels:
        train_env = MDPEnvironment(
            State(0, "low", "low", "normal"),
            cost_coeff=cost,
        )
        _, policy = value_iteration(env=train_env)

        sim_env = MDPEnvironment(
            State(2, "low", "low", "normal"),
            seed=SEED,
            cost_coeff=cost,
        )
        results[cost] = Simulator(sim_env, policy, HORIZON).run_policy()
        _row(f"cost_coeff = {cost:.1f}", results[cost])

    if SAVE_DATA:
        _save(results, "scenario_C")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    results_A = scenario_A()
    results_B = scenario_B()
    results_C = scenario_C()

    print("\nAll scenarios complete.")
    return results_A, results_B, results_C


if __name__ == "__main__":
    main()
