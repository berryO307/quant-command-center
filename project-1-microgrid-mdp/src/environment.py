import numpy as np
from typing import NamedTuple

from src.dynamics import sample_next_exogenous

BATTERY_MIN = 0
BATTERY_MAX = 4

PRICE_VALUES = {"low": 1.0, "high": 3.0}
GEN_VALUES   = {"low": 0.0, "high": 1.0}
DEMAND       = 1.0

ACTIONS = (-1, 0, 1)  # discharge, hold, charge


class State(NamedTuple):
    battery_level:    int
    price_state:      str
    generation_state: str
    regime:           str


class MDPEnvironment:
    def __init__(self, initial_state: State, seed: int = 42, cost_coeff: float = 0.1,
                 battery_max: int = BATTERY_MAX, sell_discount: float = 0.6) -> None:
        self.state         = initial_state
        self.rng           = np.random.default_rng(seed)
        self.cost_coeff    = cost_coeff
        self.battery_max   = battery_max
        self.sell_discount = sell_discount

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clamp_battery(self, battery: int, action: int) -> int:
        return int(np.clip(battery + action, BATTERY_MIN, self.battery_max))

    def _effective_action(self, battery: int, action: int) -> int:
        """Actual change in battery level after applying constraints."""
        return self._clamp_battery(battery, action) - battery

    def _reward(self, state: State, action: int) -> float:
        """
        Battery trading alpha only — generation/demand excluded (tracked as baseline).

        P_buy  = price            (cost to charge)
        P_sell = price * sell_discount  (revenue from discharge, bid-ask spread)

        R_active = -(P_buy * max(action, 0) + P_sell * min(action, 0)) - cost_coeff * |action|
          action > 0 (charge)    : pays P_buy  per unit → negative contribution
          action < 0 (discharge) : earns P_sell per unit → positive contribution
          action = 0 (hold)      : zero
        """
        price      = PRICE_VALUES[state.price_state]
        p_buy      = price
        p_sell     = price * self.sell_discount
        eff        = self._effective_action(state.battery_level, action)   # clamped delta
        r_active   = -(p_buy * max(eff, 0) + p_sell * min(eff, 0))
        return r_active - self.cost_coeff * abs(eff)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[State, float]:
        if action not in ACTIONS:
            raise ValueError(f"Invalid action {action}. Must be one of {ACTIONS}.")

        reward      = self._reward(self.state, action)
        new_battery = self._clamp_battery(self.state.battery_level, action)

        next_price, next_gen, next_regime = sample_next_exogenous(
            self.state.price_state,
            self.state.generation_state,
            self.state.regime,
            self.rng,
        )

        next_state = State(new_battery, next_price, next_gen, next_regime)
        self.state = next_state
        return next_state, reward

    def reset(self, state: State) -> State:
        self.state = state
        return self.state
