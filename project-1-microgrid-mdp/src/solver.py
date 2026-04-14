import numpy as np
from itertools import product

from src.dynamics import PRICE_STATES, GEN_STATES, REGIME_STATES, sample_next_exogenous
from src.environment import ACTIONS, BATTERY_MIN, BATTERY_MAX, MDPEnvironment, State

def all_states(battery_max: int = BATTERY_MAX) -> list[State]:
    return [
        State(b, p, g, r)
        for b, p, g, r in product(
            range(BATTERY_MIN, battery_max + 1),
            PRICE_STATES,
            GEN_STATES,
            REGIME_STATES,
        )
    ]

def value_iteration(
    gamma:     float          = 0.95,
    theta:     float          = 1e-4,
    n_samples: int            = 20,
    max_iters: int            = 50,
    seed:      int            = 0,
    env:       MDPEnvironment = None,
) -> tuple[dict, dict]:
    """
    Approximate value iteration using Monte-Carlo expectation.

    Pass an MDPEnvironment to train with a specific cost_coeff.
    If omitted, a default environment (cost_coeff=0.0) is used.

    Returns
    -------
    value_table  : dict[State, float]
    policy_table : dict[State, int]   action in {-1, 0, 1}
    """
    rng    = np.random.default_rng(seed)
    if env is None:
        env = MDPEnvironment(State(0, "low", "low", "normal"), cost_coeff=0.0)
    states = all_states(battery_max=env.battery_max)

    V      = {s: 0.0 for s in states}
    policy = {s: 0   for s in states}

    for _ in range(max_iters):
        delta = 0.0

        for s in states:
            q_values = np.empty(len(ACTIONS))

            for i, a in enumerate(ACTIONS):
                # Reward is deterministic in (s, a)
                r = env._reward(s, a)

                # Next battery level is also deterministic
                b_next = env._clamp_battery(s.battery_level, a)

                # Approximate E[V(s')] by sampling exogenous transitions
                v_next = np.mean([
                    V[State(b_next, *sample_next_exogenous(s.price_state, s.generation_state, s.regime, rng))]
                    for _ in range(n_samples)
                ])

                q_values[i] = r + gamma * v_next

            best_q    = float(q_values.max())
            delta     = max(delta, abs(best_q - V[s]))
            V[s]      = best_q
            policy[s] = ACTIONS[int(q_values.argmax())]

        if delta < theta:
            break

    return V, policy
