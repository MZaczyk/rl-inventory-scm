"""
run_dqn_experiment.py
=====================
Runs the full DQN experiment (36 combinations x 30 runs x 600 episodes)
and produces two CSV files compatible with the existing QL/SARSA results:

  dqn_results_raw.csv  – 1,080 raw rows (one per replication)
  dqn_results_agg.csv – 36 aggregate rows with means, SDs, CIs, p-values

Usage:
    python run_dqn_experiment.py

Requirements: numpy, pandas, scipy  (no PyTorch/TensorFlow needed)
Estimated runtime: 30–90 min depending on hardware.

After the run, merge with your existing results_agg.csv using:
    merged = pd.merge(results_agg, dqn_results_agg,
                      on=['MEAN_DEMAND','LEAD_MODE','DISRUPT_DAYS'])
"""

import numpy as np
import pandas as pd
from scipy import stats
import time
import os

# ── import agent (put dqn_numpy.py in the same folder) ───────────
from dqn_numpy import DQNAgent, SupplyChainEnv

# ── experiment parameters ─────────────────────────────────────────
DEMANDS     = [15, 20, 25]
LEAD_MODES  = [3, 7, 14]
DISRUPTIONS = [10, 30, 60, 90]
N_RUNS      = 30
EPISODES    = 600
EVAL_SEED   = 1000   # same seeds as QL/SARSA evaluation

# ── output paths ──────────────────────────────────────────────────
OUT_RAW = 'dqn_results_raw.csv'
OUT_AGG = 'dqn_results_agg.csv'

# also load existing QL/SARSA raw results for p-value comparison
EXISTING_RAW = 'results_raw.csv'   # adjust path if needed

# ── helpers ───────────────────────────────────────────────────────
def ci95(vals):
    return 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))

def mwu_p(a, b):
    """Mann-Whitney U two-sided p-value."""
    _, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    return p

# ── main loop ─────────────────────────────────────────────────────
rows_raw = []
rows_agg = []
total    = len(DEMANDS) * len(LEAD_MODES) * len(DISRUPTIONS)
done_n   = 0
t0       = time.time()

print(f"Starting DQN experiment: {total} combinations × {N_RUNS} runs × "
      f"{EPISODES} training episodes")
print(f"Output: {OUT_RAW}, {OUT_AGG}\n")

# load existing raw data for p-value comparisons (optional)
try:
    existing = pd.read_csv(EXISTING_RAW)
    has_existing = True
    print(f"Loaded existing results: {len(existing)} rows from {EXISTING_RAW}\n")
except FileNotFoundError:
    has_existing = False
    print(f"Note: {EXISTING_RAW} not found – p-values vs. ROP will be skipped.\n")

for mu in DEMANDS:
    for b in LEAD_MODES:
        for D in DISRUPTIONS:
            done_n  += 1
            elapsed  = time.time() - t0
            eta      = (elapsed / done_n * (total - done_n)) if done_n > 1 else 0
            print(f"[{done_n:2d}/{total}]  μ={mu}  b={b:2d}  D={D:2d}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min",
                  flush=True)

            # ── train a fresh agent per combination ───────────────
            # seed is offset so each combination gets a unique RNG trajectory
            agent = DQNAgent(seed=4000 + done_n)
            agent.train(mu, b, D,
                        episodes=EPISODES,
                        train_seed_base=4000 + done_n * 1000)

            # ── evaluate ──────────────────────────────────────────
            results = agent.evaluate(mu, b, D,
                                     n_runs=N_RUNS,
                                     eval_seed_base=EVAL_SEED)

            # raw rows
            for i, r in enumerate(results):
                rows_raw.append({
                    'MEAN_DEMAND': mu,
                    'LEAD_MODE': b,
                    'DISRUPT_DAYS': D,
                    'Method':       'DQN',
                    'Run':     i,
                    'fill_rate':    r['fill_rate'],
                    'cost':        r['cost'],
                    'lost_demand':     r['lost_demand'],
                    'bullwhip':     r['bullwhip'],
                })

            frs   = np.array([r['fill_rate'] for r in results])
            costs = np.array([r['cost']     for r in results])
            losts = np.array([r['lost_demand']  for r in results])
            beis  = np.array([r['bullwhip']  for r in results])

            # p-values vs ROP (if existing data available)
            p_fr = p_k = p_u = p_bw = np.nan
            if has_existing:
                mask = ((existing['MEAN_DEMAND'] == mu) &
                        (existing['LEAD_MODE'] == b)  &
                        (existing['DISRUPT_DAYS'] == D)  &
                        (existing['Method'] == 'ROP'))
                rop = existing[mask]
                if len(rop) == N_RUNS:
                    p_fr = mwu_p(frs,   rop['fill_rate'].values)
                    p_k  = mwu_p(costs, rop['cost'].values)
                    p_u  = mwu_p(losts, rop['lost_demand'].values)
                    p_bw = mwu_p(beis,  rop['bullwhip'].values)

            rows_agg.append({
                'MEAN_DEMAND':          mu,
                'LEAD_MODE':          b,
                'DISRUPT_DAYS':          D,
                'DQN_fill_rate_mean':    np.mean(frs),
                'DQN_fill_rate_sd':      np.std(frs, ddof=1),
                'DQN_fill_rate_ci95':    ci95(frs),
                'DQN_cost_mean':        np.mean(costs),
                'DQN_cost_sd':          np.std(costs, ddof=1),
                'DQN_cost_ci95':        ci95(costs),
                'DQN_lost_demand_mean':     np.mean(losts),
                'DQN_lost_demand_sd':       np.std(losts, ddof=1),
                'DQN_lost_demand_ci95':     ci95(losts),
                'DQN_bullwhip_mean':     np.mean(beis),
                'DQN_bullwhip_sd':       np.std(beis, ddof=1),
                'DQN_bullwhip_ci95':     ci95(beis),
                'p_DQN_vs_ROP_fill_rate': p_fr,
                'p_DQN_vs_ROP_cost':     p_k,
                'p_DQN_vs_ROP_lost_demand':  p_u,
                'p_DQN_vs_ROP_bullwhip':  p_bw,
            })

            # ── quick summary per combination ─────────────────────
            print(f"         FR={np.mean(frs):.1f}%  "
                  f"Cost={np.mean(costs):,.0f}  "
                  f"BEI={np.mean(beis):.2f}  "
                  f"p_vs_ROP={p_fr:.4f}" if not np.isnan(p_fr)
                  else f"         FR={np.mean(frs):.1f}%  "
                       f"Cost={np.mean(costs):,.0f}  "
                       f"BEI={np.mean(beis):.2f}")

# ── save results ──────────────────────────────────────────────────
df_raw = pd.DataFrame(rows_raw)
df_agg = pd.DataFrame(rows_agg)
df_raw.to_csv(OUT_RAW, index=False)
df_agg.to_csv(OUT_AGG, index=False)

total_time = time.time() - t0
print(f"\n{'='*60}")
print(f"Experiment complete in {total_time/60:.1f} minutes")
print(f"Raw rows : {len(df_raw):,}  ->  {OUT_RAW}")
print(f"Agg rows : {len(df_agg):,}  ->  {OUT_AGG}")

# ── summary table ─────────────────────────────────────────────────
print(f"\n{'─'*50}")
print("AGGREGATE SUMMARY (all 36 combinations):")
print(f"  Fill Rate : {df_agg['DQN_fill_rate_mean'].mean():.2f}% "
      f"± {df_agg['DQN_fill_rate_sd'].mean():.2f}")
print(f"  Cost      : {df_agg['DQN_cost_mean'].mean():,.0f} PLN")
print(f"  Lost dem. : {df_agg['DQN_lost_demand_mean'].mean():.0f} units")
print(f"  BEI       : {df_agg['DQN_bullwhip_mean'].mean():.2f}")
if has_existing and not df_agg['p_DQN_vs_ROP_fill_rate'].isna().all():
    sig = (df_agg['p_DQN_vs_ROP_fill_rate'] < 0.05).sum()
    print(f"  p<0.05 vs ROP (Fill Rate): {sig}/36 combinations")

print(f"\nNext step: merge dqn_results_agg.csv with your existing")
print(f"results_agg.csv using:")
print(f"  pd.merge(df_existing, df_dqn, on=['MEAN_DEMAND','LEAD_MODE','DISRUPT_DAYS'])")
