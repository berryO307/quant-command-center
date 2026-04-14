# Value Iteration — Mathematical & Code Inference

This document covers the **solver** — the piece that actually *computes* the optimal policy that the simulator later executes. The environment defines the rules, the simulator plays the game, and this layer is what figures out **how to play well**.

I won't re-derive the MDP tuple, state/action spaces, reward function mechanics, Markov chains, policies, or the baseline/alpha decomposition here — those live in `dynamics_foundations.md`, `environment_foundations.md`, and `simulator_foundations.md`. This file focuses on what's **new at the solver layer**: **Bellman equations**, **Value & Q-functions**, **Value Iteration as a fixed-point algorithm**, and **Monte Carlo expectation** as a statistical shortcut.

---

# Part I — Mathematical and Statistical Inference

## 1. The Value Function $V(s)$

Up to this point, I've only talked about rewards one step at a time. The solver needs something bigger — a function that tells me **how good a state is in the long run**, assuming I act optimally from there onward.

$$V(s) = \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \,\Big|\, S_0 = s \right]$$

In plain terms: $V(s)$ is the expected cumulative discounted reward of *starting* in state $s$ and playing optimally forever.

> **Why I care about $V$ and not just $R$:** The reward function only scores one action. The value function scores an entire *future*. That's what I actually want the agent to optimize — not the next trade, but the chain of all trades that follow.

Note that $\gamma$ — the discount factor I deferred in `simulator.md` — comes back here. At the solver level, I'm reasoning over an infinite horizon (the fixed-point math assumes it), so $\gamma < 1$ is required for convergence.

---

## 2. The Action-Value Function $Q(s, a)$

$V(s)$ tells me *how good the state is*. $Q(s, a)$ tells me *how good a specific action is*, in that state, assuming I act optimally afterward:

$$Q(s, a) = R(s, a) + \gamma \, \mathbb{E}_{s' \sim P(\cdot \mid s, a)} \big[ V(s') \big]$$

Breaking this down:

- $R(s, a)$ — the immediate reward (the alpha from one transaction).
- $\gamma \, \mathbb{E}[V(s')]$ — the discounted expected value of wherever I land next.

$Q$ is the quantity I actually compare across actions. The best action in state $s$ is just the one with the highest $Q$.

---

## 3. The Bellman Optimality Equation

Richard Bellman's insight from the 1950s: a complex multi-step optimization problem can be broken into overlapping subproblems linked by a **recursive equation**. For the optimal value function $V^*$:

$$V^*(s) = \max_{a \in A} \Big[ R(s, a) + \gamma \, \mathbb{E}_{s' \sim P(\cdot \mid s, a)} [V^*(s')] \Big]$$

Equivalently:

$$V^*(s) = \max_{a} Q^*(s, a)$$

This is a **fixed-point equation** — $V^*$ is the function that equals its own right-hand-side. The whole game of Value Iteration is to *find* this fixed point numerically.

The optimal policy falls out for free:

$$\pi^*(s) = \arg\max_{a} Q^*(s, a)$$

---

## 4. Value Iteration as a Fixed-Point Algorithm

I don't know $V^*$ ahead of time, so I start with a terrible guess ($V_0(s) = 0$ everywhere) and **apply the Bellman equation repeatedly as an update rule**:

$$V_{k+1}(s) \leftarrow \max_{a} \Big[ R(s, a) + \gamma \, \mathbb{E}[V_k(s')] \Big]$$

Every iteration, the approximation $V_k$ gets closer to the true $V^*$. The **Banach fixed-point theorem** guarantees this — as long as $\gamma < 1$, the Bellman operator is a contraction mapping, and repeated application converges geometrically to a unique fixed point.

### 4.1 Convergence via the Infinity Norm

The stopping condition is:

$$\|V_{k+1} - V_k\|_\infty < \theta$$

The **infinity norm** is just *the largest absolute change across any state*:

$$\|V_{k+1} - V_k\|_\infty = \max_s |V_{k+1}(s) - V_k(s)|$$

When the biggest update anywhere is smaller than a tiny threshold $\theta$, the values have stabilized and I stop.

> **Why infinity norm specifically:** I care about the *worst-case* state, not the average. If 99% of states have settled but one state is still moving by 10 units, the policy could still be wrong. The infinity norm catches that straggler.

---

## 5. Monte Carlo Expectation — The Statistical Shortcut

Textbook Value Iteration computes the expectation exactly:

$$\mathbb{E}[V(s')] = \sum_{s' \in S} P(s' \mid s, a) \, V(s')$$

This requires knowing — and summing over — the full transition distribution. For a rich exogenous process (price × generation × regime, each with their own Markov dynamics), writing out $P(s' \mid s, a)$ explicitly is a mess.

So I **swap exact probability for sampling**:

$$\mathbb{E}[V(s')] \approx \frac{1}{N} \sum_{i=1}^{N} V(s'_i), \quad s'_i \sim P(\cdot \mid s, a)$$

I draw $N$ samples from the transition distribution (using the same `sample_next_exogenous` from the dynamics file) and average their values.

### 5.1 Why This Works — Law of Large Numbers

$$\lim_{N \to \infty} \frac{1}{N} \sum_{i=1}^{N} V(s'_i) = \mathbb{E}[V(s')]$$

As $N$ grows, the sample mean provably converges to the true expectation. The tradeoff is obvious: higher $N$ → better approximation → slower iteration. My `n_samples = 20` is a compromise — small enough to stay fast, large enough that the noise doesn't wreck convergence.

> **The bigger picture:** This is *approximate* Value Iteration, not exact. I'm trading a small statistical error for a massive reduction in modeling complexity. I never have to write down the transition matrix — I only have to be able to *sample from it*. That's a much weaker requirement, and it's why Monte Carlo methods are the workhorse of modern RL.

---

# Part II — Line-by-Line Code Inference

## 1. Building the Universe — The State Space $S$

```python
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
```

This function materializes the entire state space as a concrete Python list.

- **`itertools.product`** computes the Cartesian product across all four state dimensions. Every legal combination becomes a `State` tuple.
- With $5$ battery levels, $2$ prices, $2$ generation states, and $3$ regimes, I get $5 \times 2 \times 2 \times 3 = 60$ unique states.

> **Why enumerate explicitly:** Value Iteration needs to visit every state each sweep. For a tabular method like this, the state space has to be small and fully enumerable. This is what makes the algorithm tractable here and intractable for continuous problems.

---

## 2. Initialization

```python
def value_iteration(gamma=0.95, theta=1e-4, n_samples=20, max_iters=50, seed=0, env=None):
    # ... setup code ...
    V      = {s: 0.0 for s in states}
    policy = {s: 0   for s in states}
```

- **Hyperparameters** — $\gamma = 0.95$ (§1), $\theta = 10^{-4}$ (convergence threshold, §4.1), $N = 20$ (Monte Carlo samples, §5), and `max_iters` as an iteration cap.
- **`V`** — the value function table, starting from the naive guess $V_0(s) = 0$ for all $s$. Fixed-point iteration will correct this.
- **`policy`** — starts as "do nothing everywhere." Also gets overwritten as the algorithm discovers better actions.

> **Why zero-initialize:** With $\gamma < 1$ and bounded rewards, the Bellman operator is a contraction *regardless of starting point*. Zero is just the cleanest uninformative prior.

---

## 3. The Outer Loops — Sweeps and Convergence Tracking

```python
    for _ in range(max_iters):
        delta = 0.0

        for s in states:
            q_values = np.empty(len(ACTIONS))
```

- **`for _ in range(max_iters):`** — the iteration counter $k$ from §4. It's a failsafe; contraction-mapping theory guarantees convergence, but a bounded loop protects against pathological configurations.
- **`delta = 0.0`** — this is $\|V_{k+1} - V_k\|_\infty$, reset at the start of every sweep.
- **`for s in states:`** — a **full sweep** across the state space. One inner loop = one application of the Bellman operator.
- **`q_values`** — preallocated array of size $|A| = 3$, ready to hold $Q(s, a)$ for each action.

---

## 4. Evaluating Actions — Deterministic Components

```python
            for i, a in enumerate(ACTIONS):
                # Reward is deterministic in (s, a)
                r = env._reward(s, a)

                # Next battery level is also deterministic
                b_next = env._clamp_battery(s.battery_level, a)
```

Inside the sweep, for each action $a$:

- **`r = env._reward(s, a)`** — the immediate reward $R(s, a)$. This is deterministic given $(s, a)$, so I only call it once.
- **`b_next = env._clamp_battery(...)`** — the deterministic part of the transition. The battery physics are fully known: if I charge with enough headroom, the level goes up by 1, period.

This is an important exploitation of structure: because the battery transition is deterministic, I **don't need to sample it**. I only need Monte Carlo for the random bits.

---

## 5. Monte Carlo Expectation — Approximating $\mathbb{E}[V(s')]$

```python
                # Approximate E[V(s')] by sampling exogenous transitions
                v_next = np.mean([
                    V[State(b_next, *sample_next_exogenous(s.price_state, s.generation_state, s.regime, rng))]
                    for _ in range(n_samples)
                ])
```

This is §5 in Python, and it's the cleverest block in the file.

- For each of `n_samples` iterations, I call `sample_next_exogenous` to draw one possible next $(p', g', r')$ from the market dynamics.
- I glue the deterministic `b_next` onto each sampled exogenous tuple to reconstruct a complete next-state `State(b_next, p', g', r')`.
- I look up its value in the current `V` table.
- `np.mean(...)` collapses the $N$ samples into a single number — the approximate $\mathbb{E}[V(s')]$.

> **Note on the sample source:** the samples come from `self.rng`, a seeded PRNG. That's what makes Value Iteration **reproducible** — re-running with the same seed produces the same policy, which matters for debugging.

---

## 6. The Bellman Update

```python
                q_values[i] = r + gamma * v_next

            best_q    = float(q_values.max())
            delta     = max(delta, abs(best_q - V[s]))
            V[s]      = best_q
            policy[s] = ACTIONS[int(q_values.argmax())]
```

The Bellman Optimality Equation, executed line by line:

- **`q_values[i] = r + gamma * v_next`** — this is literally $Q(s, a) = R(s, a) + \gamma \, \mathbb{E}[V(s')]$ from §2.
- **`best_q = q_values.max()`** — $\max_a Q(s, a)$ from §3. The best achievable value from state $s$.
- **`delta = max(delta, abs(best_q - V[s]))`** — updates the infinity norm tracker. I compare the new value against the old and keep the largest change seen so far this sweep.
- **`V[s] = best_q`** — overwrite the old estimate. This is the fixed-point update $V_{k+1}(s) \leftarrow \max_a Q(s, a)$.
- **`policy[s] = ACTIONS[q_values.argmax()]`** — extract $\pi^*(s) = \arg\max_a Q(s, a)$. Every value update produces a policy update for free.

> **Subtle point:** `V[s]` gets updated *during* the sweep, which means later states in the same sweep see the already-updated values of earlier states. This is technically **Gauss-Seidel-style in-place Value Iteration**, which typically converges *faster* than synchronous updates. It's a small detail, but it's why my `max_iters = 50` is usually enough.

---

## 7. The Stopping Rule

```python
        if delta < theta:
            break

    return V, policy
```

At the end of each full sweep, I check the convergence criterion from §4.1: if no state's value changed by more than $\theta$, the fixed point has been reached (to within tolerance) and I break out early.

When the loop exits — either by converging or hitting `max_iters` — I return the final `V` (how good every state is) and `policy` (what to do in every state). That `policy` dictionary is exactly the object the simulator's `run_policy` consumes.

---

## Closing Note

What makes this solver feel almost magical is that it **never sees the transition matrix**. It only ever calls `sample_next_exogenous`. The solver is model-*aware* (it knows rewards and battery physics) but transition-*sampling* (it estimates market dynamics via Monte Carlo). That hybrid is exactly what lets tabular methods scale beyond textbook toy problems — and it's the same hybrid that carries over cleanly into Q-learning and the deep RL methods I plan to build on top of this scaffolding.
