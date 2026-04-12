import numpy as np

PRICE_STATES  = ["low", "high"]
GEN_STATES    = ["low", "high"]
REGIME_STATES = ["normal", "peak", "volatile"]

PRICE_IDX  = {s: i for i, s in enumerate(PRICE_STATES)}
GEN_IDX    = {s: i for i, s in enumerate(GEN_STATES)}
REGIME_IDX = {s: i for i, s in enumerate(REGIME_STATES)}

# ---------------------------------------------------------------------------
# Regime transition
# T[i, j] = P(next_regime = j | regime = i)
# ---------------------------------------------------------------------------

REGIME_TRANSITION = np.array([
    [0.70, 0.20, 0.10],   # normal   -> normal 0.70, peak 0.20, volatile 0.10
    [0.30, 0.50, 0.20],   # peak     -> normal 0.30, peak 0.50, volatile 0.20
    [0.40, 0.20, 0.40],   # volatile -> normal 0.40, peak 0.20, volatile 0.40
])

# ---------------------------------------------------------------------------
# Price transitions — conditioned on the NEXT regime
# T[regime][i, j] = P(price' = j | price = i, next_regime)
# ---------------------------------------------------------------------------

PRICE_TRANSITION = {
    "normal":   np.array([
        [0.70, 0.30],   # low  -> low 0.70, high 0.30
        [0.40, 0.60],   # high -> low 0.40, high 0.60
    ]),
    "peak":     np.array([
        [0.10, 0.90],   # low  -> high with high probability (peak demand drives prices up)
        [0.05, 0.95],   # high -> stays high almost certainly
    ]),
    "volatile": np.array([
        [0.50, 0.50],   # uniform — price can go either way
        [0.50, 0.50],
    ]),
}

# ---------------------------------------------------------------------------
# Generation transitions — conditioned on the NEXT regime
# T[regime][i, j] = P(gen' = j | gen = i, next_regime)
# ---------------------------------------------------------------------------

GEN_TRANSITION = {
    "normal":   np.array([
        [0.60, 0.40],   # low  -> stays low 0.60, goes high 0.40
        [0.50, 0.50],   # high -> equal chance
    ]),
    "peak":     np.array([
        [0.80, 0.20],   # low generation during peak periods (cloud cover / high demand)
        [0.70, 0.30],   # high -> likely drops
    ]),
    "volatile": np.array([
        [0.50, 0.50],   # uniform
        [0.50, 0.50],
    ]),
}


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def sample_next_regime(current_regime: str, rng: np.random.Generator) -> str:
    idx = REGIME_IDX[current_regime]
    next_idx = rng.choice(len(REGIME_STATES), p=REGIME_TRANSITION[idx])
    return REGIME_STATES[next_idx]


def sample_next_price(current_price: str, next_regime: str, rng: np.random.Generator) -> str:
    idx = PRICE_IDX[current_price]
    next_idx = rng.choice(len(PRICE_STATES), p=PRICE_TRANSITION[next_regime][idx])
    return PRICE_STATES[next_idx]


def sample_next_gen(current_gen: str, next_regime: str, rng: np.random.Generator) -> str:
    idx = GEN_IDX[current_gen]
    next_idx = rng.choice(len(GEN_STATES), p=GEN_TRANSITION[next_regime][idx])
    return GEN_STATES[next_idx]


def sample_next_exogenous(
    price:  str,
    gen:    str,
    regime: str,
    rng:    np.random.Generator,
) -> tuple[str, str, str]:
    """Returns (next_price, next_gen, next_regime)."""
    next_regime = sample_next_regime(regime, rng)
    next_price  = sample_next_price(price, next_regime, rng)
    next_gen    = sample_next_gen(gen, next_regime, rng)
    return next_price, next_gen, next_regime
