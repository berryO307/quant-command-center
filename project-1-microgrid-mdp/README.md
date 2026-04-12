# Stochastic Energy Management via MDP — Microgrid Optimization

> Capstone project — Electrical Engineering, 8th Semester  
> Stochastic control of a battery-storage microgrid using a Markov Decision Process and Value Iteration.

---

## Executive Summary

This project formulates the dispatch problem of a grid-connected battery microgrid as a finite Markov Decision Process. The agent observes the current market price, solar generation, market regime, and battery state-of-charge, and decides each timestep whether to charge, hold, or discharge the battery.

A tabular Value Iteration algorithm converges to an optimal policy that maximises discounted cumulative profit from energy arbitrage, explicitly penalising battery cycling via a configurable transaction cost term. Structured experiments demonstrate that the MDP policy outperforms a random baseline, that larger storage capacity yields diminishing marginal returns, and that cost-aware training is necessary when friction is high.

---

## Mathematical Formulation

### State Space

$$\mathcal{S} = \mathcal{B} \times \mathcal{P} \times \mathcal{G} \times \mathcal{R}$$

| Variable | Symbol | Values |
|---|---|---|
| Battery level | $b$ | $\{0, 1, 2, 3, 4\}$ |
| Price state | $p$ | $\{\text{low}, \text{high}\}$ |
| Generation state | $g$ | $\{\text{low}, \text{high}\}$ |
| Market regime | $r$ | $\{\text{normal}, \text{peak}, \text{volatile}\}$ |

$$|\mathcal{S}| = 5 \times 2 \times 2 \times 3 = 60 \text{ states}$$

### Action Space

$$\mathcal{A} = \{-1,\ 0,\ +1\} \quad \text{(discharge, hold, charge)}$$

Battery transitions are deterministic and constrained:

$$b' = \text{clip}(b + a,\ 0,\ b_{\max})$$

### Reward Function

$$R(s, a) = p \cdot \underbrace{(g - d - a^{\text{eff}})}_{\text{net energy}} - c \cdot |a|$$

where $p$ is the electricity price, $g$ is generation, $d = 1$ is fixed demand, $a^{\text{eff}} = b' - b$ is the effective battery change after clamping, and $c$ is the transaction cost coefficient.

- Net energy $> 0$: surplus sold to grid (revenue)
- Net energy $< 0$: deficit bought from grid (cost)

### Stochastic Transition

The exogenous state $(p, g, r)$ evolves as a regime-modulated Markov chain:

$$r' \sim P_r(\cdot \mid r), \qquad p' \sim P_p(\cdot \mid p,\ r'), \qquad g' \sim P_g(\cdot \mid g,\ r')$$

The regime drives the price and generation dynamics — peak regimes push prices high with probability 0.90–0.95, while volatile regimes make both price and generation uniformly random.

### Bellman Optimality Equation

$$V^*(s) = \max_{a \in \mathcal{A}} \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a)\ V^*(s') \right]$$

The expectation over $s'$ is approximated via Monte-Carlo sampling (N = 20 draws per state-action pair). The algorithm iterates until the max Bellman residual falls below $\theta = 10^{-4}$.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        dynamics.py                           │
│  REGIME_TRANSITION  ──►  PRICE_TRANSITION[regime]            │
│                     └──►  GEN_TRANSITION[regime]             │
│  sample_next_exogenous(price, gen, regime) → (p', g', r')    │
└───────────────────────────┬──────────────────────────────────┘
                            │ stochastic transitions
┌───────────────────────────▼──────────────────────────────────┐
│                      environment.py                          │
│  State(battery, price, gen, regime)                          │
│  MDPEnvironment.step(action) → (next_state, reward)          │
│  _reward = price × net_energy − cost_coeff × |action|        │
└───────────────┬──────────────────────┬───────────────────────┘
                │ env (for reward)     │ env (for simulation)
┌───────────────▼──────────┐  ┌───────▼───────────────────────┐
│        solver.py         │  │        simulator.py            │
│  value_iteration(env)    │  │  Simulator(env, policy, T)     │
│  → V*, policy*           │  │  .run_policy() → DataFrame     │
└──────────────────────────┘  └───────────────────────────────┘
```

---

## Project Structure

```
project-1-microgrid-mdp/
├── src/
│   ├── dynamics.py        # Markov transition matrices (regime-modulated)
│   ├── environment.py     # MDP state, reward, step()
│   ├── solver.py          # Value Iteration
│   ├── simulator.py       # Policy rollout → trajectory DataFrame
│   └── plot_results.py    # All plotting utilities
├── experiments/
│   ├── run_simulation.py  # Baseline: cost-aware vs naive policy comparison
│   ├── experiments.py     # Structured scenarios A / B / C
│   └── run_plots.py       # Generate all publication plots
├── results/
│   ├── plots/             # regime_pnl.png, policy_heatmap.png, action_histogram.png
│   └── experiments/       # Optional CSV output
├── notes/
│   ├── mdp_formulation.md
│   └── transition_matrices.md
├── requirements.txt
└── README.md
```

---

## How to Run

### Install dependencies

```bash
pip install numpy pandas matplotlib
```

### 1 — Compare cost-aware vs naive policy

```bash
python -m experiments.run_simulation
```

Trains two policies (with and without transaction cost) and plots three PnL curves to `results/plots/pnl_comparison.png`.

### 2 — Run structured experiments

```bash
python -m experiments.experiments
```

Runs three scenarios and prints a results table:

| Scenario | Variable | Description |
|---|---|---|
| A | Initial regime | Stable (`normal`) vs volatile start |
| B | `battery_max` | Capacity sweep: 2 / 4 / 8 |
| C | `cost_coeff` | Friction sweep: 0.0 / 0.1 / 0.3 |

Set `SAVE_DATA = True` inside the script to write per-scenario CSVs to `results/experiments/`.

### 3 — Generate publication plots

```bash
python -m experiments.run_plots
```

Writes three figures to `results/plots/`:

| File | Description |
|---|---|
| `regime_pnl.png` | Cumulative PnL with regime-coloured background bands |
| `policy_heatmap.png` | Converged policy action for every state, per regime |
| `action_histogram.png` | Action distribution shift under increasing transaction costs |

---

## Results & Insights

### Scenario A — Market Regime Robustness

Starting in a volatile regime versus a normal regime produces near-identical cumulative reward over 200 steps. This confirms that the Markov chain mixes to its stationary distribution well within the simulation horizon — the policy is regime-robust.

### Scenario B — Battery Capacity (Diminishing Returns)

| `battery_max` | Total Reward | Marginal Gain |
|---|---|---|
| 2 | −166.0 | — |
| 4 | −141.0 | +25.0 |
| 8 | −133.0 | +8.0 |

Doubling capacity from 2 → 4 yields a 25-unit gain. Doubling again to 8 yields only 8 additional units. Once the battery is large enough to exploit all viable price spreads within the planning horizon, further capacity adds little — a result consistent with energy storage investment theory.

### Scenario C — Transaction Cost Sensitivity

| `cost_coeff` | Total Reward | Hold actions | Active actions |
|---|---|---|---|
| 0.0 | −141.0 | 14 | 186 |
| 0.1 | −166.2 | 28 | 172 |
| 0.3 | −201.5 | 45 | 155 |

Two effects are visible:

1. **PnL degrades monotonically** with cost — friction directly erodes arbitrage profit.
2. **The policy internalises the cost**: as `cost_coeff` rises, hold actions nearly triple. The agent learns that unless the price spread is wide enough to cover the round-trip cost, staying idle is optimal. This is the discrete analogue of the no-trade region in continuous-time transaction cost models.

### Policy Structure

The converged policy (visible in `policy_heatmap.png`) has a clear structure:

- **Low battery + high price → Discharge** (sell while price is high)
- **High battery + low price → Hold** (no benefit selling at low price)
- **Low battery + low price + high generation → Charge** (store cheap surplus)
- **Peak regime** shifts the discharge threshold — the agent discharges more aggressively because high prices are more persistent under peak dynamics.

---

## Key Parameters

| Parameter | Default | Effect |
|---|---|---|
| `gamma` | 0.95 | Discount factor — higher values favour long-term arbitrage |
| `cost_coeff` | 0.1 | Transaction friction — raises the no-action threshold |
| `battery_max` | 4 | Storage capacity — diminishing returns above ~4 |
| `n_samples` | 20 | MC draws per Bellman update — increase for tighter convergence |
| `horizon` | 200 | Simulation length |
