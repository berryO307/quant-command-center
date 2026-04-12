from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.environment import ACTIONS, DEMAND, GEN_VALUES, MDPEnvironment, PRICE_VALUES, State


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SimulationSummary:
    total_reward:  float          # cumulative R_active (= alpha_pnl)
    action_counts: dict           # {-1: int, 0: int, 1: int}
    trajectory:    pd.DataFrame
    baseline_pnl:  float          # cumulative PnL without battery
    alpha_pnl:     float          # battery contribution = mdp_pnl - baseline_pnl


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    def __init__(
        self,
        env:     MDPEnvironment,
        policy:  dict,
        horizon: int,
    ) -> None:
        self.env     = env
        self.policy  = policy
        self.horizon = horizon

    # ------------------------------------------------------------------
    # Core runner
    # ------------------------------------------------------------------

    def _run(self, action_fn) -> SimulationSummary:
        records             = []
        cumulative_reward   = 0.0   # = cumulative R_active
        cumulative_baseline = 0.0
        cumulative_alpha    = 0.0

        for t in range(self.horizon):
            state  = self.env.state
            action = action_fn(state)

            # Baseline: what the site earns/pays with no battery
            price         = PRICE_VALUES[state.price_state]
            generation    = GEN_VALUES[state.generation_state]
            baseline_step = price * (generation - DEMAND)

            next_state, reward = self.env.step(action)   # reward = R_active

            cumulative_baseline += baseline_step
            cumulative_alpha    += reward
            cumulative_reward   += reward   # kept for backward compatibility
            cumulative_mdp       = cumulative_baseline + cumulative_alpha

            records.append({
                "t":                 t,
                "regime":            state.regime,
                "battery_level":     state.battery_level,
                "price_state":       state.price_state,
                "generation_state":  state.generation_state,
                "action":            action,
                "reward":            reward,
                "cumulative_reward": cumulative_reward,
                "baseline_pnl":      cumulative_baseline,
                "mdp_pnl":           cumulative_mdp,
                "alpha_pnl":         cumulative_alpha,
            })

        df = pd.DataFrame(records)
        action_counts = {a: int((df["action"] == a).sum()) for a in ACTIONS}

        return SimulationSummary(
            total_reward  = cumulative_reward,
            action_counts = action_counts,
            trajectory    = df,
            baseline_pnl  = cumulative_baseline,
            alpha_pnl     = cumulative_alpha,
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def run_policy(self) -> SimulationSummary:
        return self._run(action_fn=lambda s: self.policy[s])

    def run_random_policy(self, seed: int = 0) -> SimulationSummary:
        rng = np.random.default_rng(seed)
        return self._run(action_fn=lambda s: int(rng.choice(ACTIONS)))
