# MDP Environment — Mathematical & Code Inference

This document unpacks the battery-trading environment I built. Part I covers the **mathematical and statistical theory** driving the simulator. Part II walks through the **Python code block-by-block**, mapping every line back to the theory in Part I.

---

# Part I — Mathematical and Statistical Inference

At its core, my code models a **Markov Decision Process (MDP)** — the mathematical framework for decision-making where outcomes are partly random and partly under the control of an agent. Understanding *why* this works requires three pillars: **Reinforcement Learning math**, **Optimization math**, and **Stochastic Processes**.

---

## 1. Core Reinforcement Learning Mathematics

### 1.1 State Space ($S$)

The set of all possible situations the environment can be in.

My state is a **vector combining endogenous and exogenous variables**:

- **Endogenous (internal, agent-controlled):** `battery_level` — an integer.
- **Exogenous (external, market-driven):** `price_state`, `generation_state`, `regime` — categorical strings.

### 1.2 Action Space ($A$)

The set of all valid decisions the agent can make from a given state.

I use a **discrete, finite** action space:

$$A = \{-1,\ 0,\ +1\}$$

The agent can only pull one of three levers at a time — *discharge*, *hold*, or *charge*.

### 1.3 Reward Function ($R$)

A scalar feedback signal $R(s, a)$ that indicates how good an action was. The agent's goal is to maximize cumulative reward.

I am using a **dense reward function** that calculates immediate financial return based on the action taken, minus operational penalties. "Dense" means the agent gets feedback at every step, not just at the end of an episode.

### 1.4 Transition Dynamics ($P$)

The probability distribution $P(s' \mid s, a)$ of moving to next state $s'$ given current state $s$ and action $a$.

My system is **mixed**:

- The battery transition is purely **deterministic** — if I charge by $+1$, the battery goes up by $1$ (subject to bounds).
- The market transitions (price, generation, regime) are **stochastic** — sampled from the Markov Chain I defined in the dynamics file.

> **Why this mix matters:** The agent has *full control* over its battery but *zero control* over the market. This is the entire point of trading — you decide how to react to randomness.

---

## 2. Financial & Optimization Mathematics

The `_reward` function is the engine of the environment. It relies on specific mathematical formulations to simulate realistic market constraints and penalize bad behavior.

### 2.1 Piecewise / Asymmetric Functions

Functions that apply different rules depending on the input's sign:

$$\max(a, 0) \quad \text{and} \quad \min(a, 0)$$

I use these to **separate buying logic from selling logic**. Buying costs $P_{\text{buy}}$, but selling only yields $P_{\text{sell}}$ due to my bid-ask spread (the `sell_discount`).

### 2.2 $L_1$ Norm — Absolute Value Penalty

A penalty that scales **linearly** with magnitude:

$$\lambda \, \|x\|_1 = \lambda \, |x|$$

My transaction cost uses $\lambda \, |a|$ (`abs(eff)`). Every unit of energy moved incurs a flat rate of friction, discouraging the agent from needlessly cycling the battery.

### 2.3 $L_2$ Norm — Quadratic Penalty

A penalty that scales with the **square** of the variable:

$$\lambda \, x^2$$

This heavily punishes large deviations while forgiving small ones. My `inventory_penalty` squares the battery percentage — holding at 100% is penalized *much* more harshly than holding at 50%, which discourages hoarding.

### 2.4 Clamping / Bounding Functions

Restricting a variable to a closed interval $[a, b]$:

$$f(x) = \max(\min(x, b),\ a)$$

The battery physically cannot exceed `BATTERY_MAX` or drop below `BATTERY_MIN`. This math enforces the boundary condition before executing actions.

---

## 3. Statistical & Stochastic Theory

The environment leans heavily on randomness to simulate the unpredictable nature of energy markets.

### 3.1 Stochastic Exogenous Processes

*Exogenous* means external factors that affect the system but cannot be controlled by the agent. A **stochastic process** is a family of random variables indexed by time.

`sample_next_exogenous` (imported from my dynamics file) handles the random progression of prices, generation, and regimes — completely independent of the agent's battery actions.

### 3.2 Markov Chains

A stochastic model where the probability of the next event depends *only* on the current state:

$$P(S_{t+1} \mid S_t, S_{t-1}, \ldots) = P(S_{t+1} \mid S_t)$$

Sampling the next price state based on the current price state is the textbook definition of a Markov Chain — exactly what my dynamics module implements.

### 3.3 Pseudo-Random Number Generation & Seeding

PRNG algorithms use deterministic formulas to produce sequences that *appear* random. A **seed** is the starting point for that sequence.

I use `np.random.default_rng(seed)`. Setting a seed ensures my market fluctuations are **exactly the same every run**, letting me isolate and evaluate the agent's performance properly. Two agents tested on seed `42` face identical markets — any difference in performance is purely the policy.

---

# Part II — Line-by-Line Code Inference

Now I'll pull the code apart block by block to see how the theory above translates into Python.

---

## 1. Imports and Global Constants

```python
import numpy as np
from typing import NamedTuple
from src.dynamics import sample_next_exogenous

BATTERY_MIN = 0
BATTERY_MAX = 4
PRICE_VALUES = {"low": 1.0, "high": 3.0}
GEN_VALUES   = {"low": 0.0, "high": 1.0}
DEMAND       = 1.0
ACTIONS = (-1, 0, 1)  # discharge, hold, charge
```

- **`NamedTuple`** — I use this for memory efficiency and immutability. In RL, state shouldn't be accidentally modified in place; it should be overwritten entirely.
- **`sample_next_exogenous`** — my stochastic transition function for the environment (the Markov Chain from the dynamics file).
- **The Constants** — these define the absolute boundaries of my MDP. The battery has 5 discrete levels ($0, 1, 2, 3, 4$), and the agent only ever has 3 choices.

---

## 2. The State Representation

```python
class State(NamedTuple):
    battery_level:    int
    price_state:      str
    generation_state: str
    regime:           str
```

This is my state vector $S$. Notice the **mix of data types**: the battery is an `int` (endogenous — changed by the agent), while the market factors are `str` categories (exogenous — driven by the stochastic engine).

---

## 3. Environment Initialization

```python
class MDPEnvironment:
    def __init__(self, initial_state: State, seed: int = 42, cost_coeff: float = 0.1,
                 battery_max: int = BATTERY_MAX, sell_discount: float = 0.6,
                 inventory_coeff: float = 0.05) -> None:
        self.state           = initial_state
        self.rng             = np.random.default_rng(seed)
        self.cost_coeff      = cost_coeff
        self.battery_max     = battery_max
        self.sell_discount   = sell_discount
        self.inventory_coeff = inventory_coeff
```

- **`self.rng = np.random.default_rng(seed)`** — my Pseudo-Random Number Generator. I use the modern `default_rng` rather than the legacy `np.random.seed()` because it prevents global state contamination, which is crucial when running thousands of RL episodes in parallel.
- The coefficients (`cost_coeff`, `sell_discount`, `inventory_coeff`) are my **hyperparameters**. They tune the strictness of the mathematical penalties.

---

## 4. Physical Constraints (Internal Helpers)

```python
    def _clamp_battery(self, battery: int, action: int) -> int:
        return int(np.clip(battery + action, BATTERY_MIN, self.battery_max))

    def _effective_action(self, battery: int, action: int) -> int:
        """Actual change in battery level after applying constraints."""
        return self._clamp_battery(battery, action) - battery
```

- **`_clamp_battery`** — this is the bounding function from §2.4. If the battery is at $4$ and the agent chooses action $+1$, `np.clip` catches it and forces it back to $4$.
- **`_effective_action`** — this calculates the **delta**. In the scenario above, the agent *tried* to charge ($+1$), but the effective action was $0$. This is critical because I don't want to charge the agent transaction fees for energy that was never physically moved.

> **Why the distinction matters:** If I skipped `_effective_action` and used the raw action in the cost calculation, the agent would be penalized for attempted but impossible actions, which distorts learning.

---

## 5. The Reward Function

```python
    def _reward(self, state: State, action: int) -> float:
        price             = PRICE_VALUES[state.price_state]
        p_buy             = price
        p_sell            = price * self.sell_discount
        eff               = self._effective_action(state.battery_level, action)

        r_active          = -(p_buy * max(eff, 0) + p_sell * min(eff, 0))
        transaction_cost  = self.cost_coeff * abs(eff)
        inventory_penalty = self.inventory_coeff * (state.battery_level / self.battery_max) ** 2

        return r_active - transaction_cost - inventory_penalty
```

This is where all three mathematical pillars collide.

- **`p_sell = price * self.sell_discount`** — creates the **asymmetric bid-ask spread**. Selling is inherently less profitable per unit than buying costs.
- **`r_active`** — implements the **piecewise logic** from §2.1:
  - If `eff > 0` (charging): `min(eff, 0)` becomes $0$. The formula reduces to $-p_{\text{buy}} \cdot \text{eff}$ → the agent **loses money** to buy power.
  - If `eff < 0` (discharging): `max(eff, 0)` becomes $0$. The formula reduces to $-p_{\text{sell}} \cdot \text{eff}$ → because `eff` is negative, the double negative turns into **positive profit**.
- **`transaction_cost`** — the **$L_1$ norm penalty** from §2.2. `abs(eff)` ensures friction is applied whether charging or discharging.
- **`inventory_penalty`** — the **$L_2$ quadratic penalty** from §2.3. Squaring the battery percentage means the penalty curve gets steeper as the battery approaches 100%.

The final reward is a single scalar:

$$R(s, a) = r_{\text{active}} - \text{Transaction Cost} - \text{Inventory Penalty}$$

$$R(s, a) = r_{\text{active}} - \lambda_1 \, |\text{eff}| - \lambda_2 \left(\frac{b}{b_{\max}}\right)^2$$


The agent wants this final number to be as high as possible. To win, it must time the market (buy low, sell high) while minimizing hardware degradation (transactions) and avoiding hoarding (inventory).

---

## 6. The Core RL Loop (Public Interface)

```python
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
```

This is the heartbeat of the environment — one call per timestep.

- **Validation** — fails fast if a rogue action is passed.
- **Determinism vs. Stochasticism** — both live in the same function:
  - `new_battery` is strictly calculated from the agent's action and physics constraints → **deterministic**.
  - `next_price`, `next_gen`, `next_regime` are sampled via the PRNG → **stochastic**.
- **State Transition** — a new `State` tuple is minted, the environment's internal state is updated, and `(next_state, reward)` is yielded to whatever algorithm (Q-learning, PPO, etc.) is driving the agent.

---

## 7. Reset

```python
    def reset(self, state: State) -> State:
        self.state = state
        return self.state
```

Essential for **episodic RL training**. When an episode ends, this resets the environment to a fresh starting state so the agent can try again. Without `reset`, the agent would be stuck in one infinite trajectory — no batching, no averaging, no learning.

---

## Closing Note

The elegance of this setup is that the **math and the code are one-to-one**. Every clamp, every max/min, every PRNG call maps cleanly back to a named mathematical object in Part I. That tight coupling is the property I want: it means when I debug a misbehaving agent, I can reason about it in terms of $L_1$ penalties and transition probabilities, not just Python.