# Reinforcement Learning for Inventory Management under Supply Chain Disruptions

Simulation code for the paper:

> Zaczyk, M. (2025). *Reinforcement Learning for Inventory Management under Supply Chain
> Disruptions: A Multi-Agent Simulation Study with Stochastic Lead Times.*
> Submitted to Computers & Operations Research.

---

## Overview

This repository contains all simulation code and experimental output data used in the paper.
The implementation uses **pure Python with NumPy** — no GPU, no PyTorch, no TensorFlow required.

The simulation models a **three-echelon supply chain** (Supplier → Manufacturer → Warehouse →
Customer) with:

- Stochastic lead times: `Triangular(a=2, b ∈ {3, 7, 14}, c=14)` days
- Poisson customer demand: `μ ∈ {15, 20, 25}` units/day
- Supplier disruptions: `D ∈ {10, 30, 60, 90}` days, randomised start

**Five ordering policies** are compared:

| Method | Type | Description |
|---|---|---|
| ROP | Classical | Reorder Point (R=80, Q=150) |
| (s,S) | Classical | Two-threshold policy (s=80, S=300) |
| Q-Learning | Tabular RL | 60-state Q-table, ε-greedy |
| SARSA | Tabular RL | On-policy, same hyperparameters as QL |
| DQN | Deep RL | 2×64 hidden layers, pure NumPy |

---

## Requirements

Python 3.8 or later. Install dependencies with:

```bash
pip install numpy pandas scipy
```

---

## Repository structure

```
rl-inventory-scm/
├── README.md
├── requirements.txt
│
├── core/
│   └── dqn_numpy.py             # SupplyChainEnv + DQN agent (NumPy)
│                                #   also used by QL/SARSA experiments
│
├── experiments/
│   ├── run_dqn_experiment.py    # Main DQN experiment (36×30×600)
│   ├── run_dqn_extended.py      # DQN robustness: 1200 episodes, D=60/90
│   ├── run_cost_sensitivity.py  # Cost ratio sensitivity: c_l/c_h ∈ {5,20,50}
│   └── run_rop_sensitivity.py   # ROP threshold sensitivity: R ∈ {60,80,100}
│
└── data/
    ├── results_raw.csv              # Raw results: ROP, (s,S), QL, SARSA (4320 runs)
    ├── results_agg.csv              # Aggregate: means, SDs, CIs, p-values
    ├── dqn_results_raw.csv          # Raw results: DQN (1080 runs)
    ├── dqn_results_agg.csv          # Aggregate DQN results
    ├── dqn_extended_results_raw.csv # DQN 1200-episode results (540 runs)
    ├── dqn_extended_results_agg.csv # Aggregate DQN extended
    ├── cost_sensitivity_results.csv # Cost sensitivity (3 ratios × 36 combos)
    ├── cost_sensitivity_breakeven.csv
    ├── rop_sensitivity_results.csv  # ROP threshold sensitivity
    └── rop_sensitivity_summary.csv
```

---

## Column names in data files

| Column | Description |
|---|---|
| `MEAN_DEMAND` | Mean daily demand μ ∈ {15, 20, 25} |
| `LEAD_MODE` | Lead time modal value b ∈ {3, 7, 14} days |
| `DISRUPT_DAYS` | Disruption duration D ∈ {10, 30, 60, 90} days |
| `Method` | Ordering policy name |
| `Run` | Replication index (0-indexed) |
| `fill_rate` | Fill Rate (%) |
| `cost` | Total operating cost (PLN) |
| `lost_demand` | Total unmet demand (units) |
| `bullwhip` | Bullwhip Effect Index |

---

## How to reproduce the results

All scripts must be run from the `experiments/` directory,
with `core/` and `data/` present at the same level.

### Step 1 — QL, SARSA, ROP, (s,S) baseline

The baseline results for classical and tabular RL methods are provided in
`data/results_raw.csv` and `data/results_agg.csv`.
The simulation script for these methods is not included separately —
the environment class in `core/dqn_numpy.py` (`SupplyChainEnv`) implements
identical dynamics. Results were generated with `n=30` replications per cell
and evaluation seeds `1000 + i`.

### Step 2 — DQN experiment

```bash
cd experiments/
python run_dqn_experiment.py
```

Reads: `data/results_raw.csv` (for p-value comparisons vs ROP)  
Writes: `dqn_results_raw.csv`, `dqn_results_agg.csv`  
Time: approximately 30–90 minutes on CPU

### Step 3 — DQN robustness (extended training)

```bash
python run_dqn_extended.py
```

Reads: `data/results_raw.csv`, `data/dqn_results_raw.csv`  
Writes: `dqn_extended_results_raw.csv`, `dqn_extended_results_agg.csv`  
Purpose: Confirms that QL > DQN at D ≥ 60 days is structural, not undertrained

### Step 4 — Cost sensitivity analysis

```bash
python run_cost_sensitivity.py
```

Reads: `data/results_raw.csv`, `data/dqn_results_raw.csv`  
Writes: `cost_sensitivity_results.csv`, `cost_sensitivity_breakeven.csv`  
Variants: `c_l/c_h ∈ {5, 20, 50}`  
Time: approximately 2–3 hours

### Step 5 — ROP threshold sensitivity

```bash
python run_rop_sensitivity.py
```

Reads: `data/results_raw.csv` (for QL comparison)  
Writes: `rop_sensitivity_results.csv`, `rop_sensitivity_summary.csv`  
Variants: `R ∈ {60, 80, 100}`  
Time: approximately 2–5 minutes (classical methods only)

---

## Key results

Aggregate Fill Rate averaged across all 36 parameter combinations
(n = 1,080 replications per method):

| Method | Fill Rate | Cost (PLN) | BEI |
|---|---|---|---|
| ROP | 68.83% | 36,311 | 9.51 |
| (s,S) | 74.43% | 35,745 | 13.53 |
| Q-Learning | 93.37% | 79,231 | 19.21 |
| SARSA | 93.38% | 91,233 | 18.74 |
| DQN | 92.89% | 110,844 | 13.77 |

Q-Learning significantly outperforms ROP in all 36 parameter combinations
(Mann-Whitney U, p < 0.001 in 36/36 cases).

---

## Reproducibility notes

- All random seeds are fixed per replication: evaluation seed = `1000 + i`,
  DQN training seed = `4000 + combo_index × 1000`.
- Results may differ slightly across platforms due to floating-point
  differences in NumPy, but aggregate statistics should match reported
  values within ±0.1 pp Fill Rate.
- Tested on Python 3.10 and 3.12 with NumPy 1.24–2.4, pandas 1.5–2.2,
  scipy 1.10–1.13.

---

## Computational complexity

| Agent | Memory | Per-step update | Training time (per combo) |
|---|---|---|---|
| Q-Learning / SARSA | O(360) entries | O(1) | ≈ 8 s |
| DQN | O(8,900) parameters + O(30,000) buffer | O(B·d·H) | ≈ 45 s |

All experiments run on standard CPU hardware (Intel Core i7, 16 GB RAM).
No GPU required.

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

Mateusz Zaczyk  
Faculty of Organization and Management  
Silesian University of Technology  
mzaczyk@polsl.pl  
ORCID: 0000-0002-3206-4784
