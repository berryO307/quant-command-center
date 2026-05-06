# Barinder Singh — Quant Finance Portfolio

**Electrical Engineering undergrad (final year) · 22 · targeting Quant Developer and Quant Researcher roles**

I build quantitative systems from the hardware up. My background in EE gives me a natural home in the low-level end of quant infrastructure — cache-line-aware C++, memory-mapped I/O, hardware timing — and I pair that with rigorous Python research and modelling work. This repository is a sequenced record of that work, built in public.

---

## Projects

| # | Project | Role Target | Stack | Headline |
|---|---------|-------------|-------|----------|
| 2 | [**L2 Market Data Gateway**](./project-2-L2-market-data-gateway) | **Quant Developer** | C++20, Boost.Beast, simdjson, Python | Lock-free SPSC ingestion pipeline for crypto L2 data. Queue transit **p99 = 7.4 µs** on live Bybit Futures. Hierarchical bitboard order book with O(1) best-price lookup via hardware intrinsics. |
| 1 | [**Microgrid MDP**](./project-1-microgrid-mdp) | **Quant Researcher** | Python, NumPy, Pandas | Markov Decision Process model for optimal energy dispatch in a microgrid. Dynamic programming solution with sensitivity analysis across demand and generation scenarios. |

Projects are numbered in build order. Each has its own README, deep technical notes, and analytical instrumentation.

---

## What I Can Do

**Systems side (QD)**
- Lock-free concurrent data structures — SPSC ring buffers, atomic memory ordering, cache-line discipline
- Hardware-aware C++ — `alignas(64)`, `__rdtscp` timing, `mlock`'d memory, thread affinity, real-time scheduling
- Market microstructure engineering — L2 order book reconstruction, sequence-gap detection and recovery, fixed-point arithmetic
- Cross-platform build systems (CMake, GCC/Clang/MSVC)

**Research side (QR)**
- Stochastic modelling — Markov chains, dynamic programming, MDP formulations
- Statistical analysis and backtesting in Python — pandas, NumPy, SciPy
- Data pipeline design — ingestion, normalization, storage, visualization
- Latency analysis and instrumentation — Plotly dashboards, percentile decomposition, root-cause attribution

---

## Background

I am a final-year Electrical Engineering student with a strong pull toward quantitative finance. EE gave me the foundation: signals, systems, linear algebra, probability, and a habit of thinking in hardware constraints rather than abstractions. I have redirected that toward financial systems — first through research-oriented modelling (Project 1), then through production-infrastructure engineering (Project 2).

The projects are intentionally sequenced to cover both sides of a quant firm's hiring surface. I understand that QR and QD are distinct roles with distinct skill demands, and the work here reflects that distinction rather than blurring it.

---

## Structure

```
quant-portfolio/
├── project-1-microgrid-mdp/          # QR: Python, MDP, dynamic programming
└── project-2-L2-market-data-gateway/ # QD: C++20, HFT, lock-free systems
    ├── include/                       # C++ headers
    ├── src/                           # C++ implementations
    ├── analysis/                      # Python latency analysis
    │   └── notebooks/latency_analysis.ipynb
    └── notes/Engineering_Notes.md     # Architectural thesis
```

---

## Contact

**Barinder Singh**
- GitHub: [@berryO307](https://github.com/berryO307)
- Email: barindersinghdhanoa@gmail.com
- LinkedIn: *www.linkedin.com/in/berry07*

Open to internship and full-time opportunities in quant finance — both QR and QD tracks. Happy to walk through any part of the work in technical detail.






