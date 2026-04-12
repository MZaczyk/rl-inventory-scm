"""
run_dqn_extended.py
===================
DQN experiment with 1200 training episodes for D=60 and D=90 scenarios.
Purpose: verify that the QL > DQN result at long disruptions is a structural
property of the architecture, not an artefact of insufficient training.

Covers: 3 demand levels × 3 lead times × 2 disruption durations = 18 combinations
Each: 1200 training episodes + 30 evaluation runs
Seeds: offset from original experiment to ensure independence

Output files:
  dqn_extended_results_raw.csv   – 540 raw rows
  dqn_extended_results_agg.csv  – 18 aggregate rows + p-values vs QL (original)

Usage:
    # Put this file in the same folder as dqn_numpy.py and results_raw.csv
    python run_dqn_extended.py

Requirements: numpy, pandas, scipy
"""

import numpy as np
import pandas as pd
from scipy import stats
import time

from dqn_numpy import DQNAgent, SupplyChainEnv

# ── parameters ────────────────────────────────────────────────────
DEMANDS      = [15, 20, 25]
LEAD_MODES   = [3, 7, 14]
DISRUPTIONS  = [60, 90]          # only long disruptions
N_RUNS       = 30
EPISODES     = 1200              # double the original 600
EVAL_SEED    = 1000              # same as original for fair comparison

OUT_RAW = 'dqn_extended_results_raw.csv'
OUT_AGG = 'dqn_extended_results_agg.csv'

# original results for p-value comparison
ORIGINAL_RAW = 'results_raw.csv'
DQN_ORIG_RAW = 'dqn_results_raw.csv'

def ci95(vals):
    return 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))

def mwu_p(a, b):
    _, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    return p

# ── load existing data ────────────────────────────────────────────
try:
    existing     = pd.read_csv(ORIGINAL_RAW)
    dqn_original = pd.read_csv(DQN_ORIG_RAW)
    has_existing = True
    print(f"Loaded: {ORIGINAL_RAW} ({len(existing)} rows)")
    print(f"Loaded: {DQN_ORIG_RAW} ({len(dqn_original)} rows)")
except FileNotFoundError as e:
    has_existing = False
    print(f"Warning: {e} — p-values will be skipped")

# ── experiment ────────────────────────────────────────────────────
rows_raw = []
rows_agg = []
total    = len(DEMANDS) * len(LEAD_MODES) * len(DISRUPTIONS)
done_n   = 0
t0       = time.time()

print(f"\nStarting extended DQN experiment:")
print(f"  {total} combinations × {N_RUNS} runs × {EPISODES} training episodes")
print(f"  Disruptions: {DISRUPTIONS} days only (long disruption scenarios)\n")

for mu in DEMANDS:
    for b in LEAD_MODES:
        for D in DISRUPTIONS:
            done_n  += 1
            elapsed  = time.time() - t0
            eta      = (elapsed / done_n * (total - done_n)) if done_n > 1 else 0
            print(f"[{done_n:2d}/{total}]  μ={mu}  b={b:2d}  D={D}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min",
                  flush=True)

            # seed offset: 9000-series to avoid collision with original (4000-series)
            agent = DQNAgent(seed=9000 + done_n)
            agent.train(mu, b, D,
                        episodes=EPISODES,
                        train_seed_base=9000 + done_n * 1000)

            results = agent.evaluate(mu, b, D,
                                     n_runs=N_RUNS,
                                     eval_seed_base=EVAL_SEED)

            for i, r in enumerate(results):
                rows_raw.append({
                    'MEAN_DEMAND':  mu,
                    'LEAD_MODE':  b,
                    'DISRUPT_DAYS':  D,
                    'Method':        'DQN_1200',
                    'Run':      i,
                    'fill_rate':     r['fill_rate'],
                    'cost':         r['cost'],
                    'lost_demand':      r['lost_demand'],
                    'bullwhip':      r['bullwhip'],
                })

            frs   = np.array([r['fill_rate'] for r in results])
            costs = np.array([r['cost']     for r in results])
            losts = np.array([r['lost_demand']  for r in results])
            beis  = np.array([r['bullwhip']  for r in results])

            # p-values vs QL original and vs DQN-600 original
            p_vs_ql_fr = p_vs_ql_k = np.nan
            p_vs_dqn600_fr = p_vs_dqn600_k = np.nan

            if has_existing:
                # vs QL
                mask_ql = ((existing['MEAN_DEMAND'] == mu) &
                           (existing['LEAD_MODE'] == b)  &
                           (existing['DISRUPT_DAYS'] == D)  &
                           (existing['Method'] == 'QL'))
                ql_data = existing[mask_ql]
                if len(ql_data) == N_RUNS:
                    p_vs_ql_fr = mwu_p(frs, ql_data['fill_rate'].values)
                    p_vs_ql_k  = mwu_p(costs, ql_data['cost'].values)

                # vs DQN-600 (original)
                mask_d600 = ((dqn_original['MEAN_DEMAND'] == mu) &
                             (dqn_original['LEAD_MODE'] == b)  &
                             (dqn_original['DISRUPT_DAYS'] == D))
                d600_data = dqn_original[mask_d600]
                if len(d600_data) == N_RUNS:
                    p_vs_dqn600_fr = mwu_p(frs, d600_data['fill_rate'].values)
                    p_vs_dqn600_k  = mwu_p(costs, d600_data['cost'].values)

            rows_agg.append({
                'MEAN_DEMAND':             mu,
                'LEAD_MODE':             b,
                'DISRUPT_DAYS':             D,
                'DQN1200_fill_rate_mean':   np.mean(frs),
                'DQN1200_fill_rate_sd':     np.std(frs, ddof=1),
                'DQN1200_fill_rate_ci95':   ci95(frs),
                'DQN1200_cost_mean':       np.mean(costs),
                'DQN1200_cost_sd':         np.std(costs, ddof=1),
                'DQN1200_cost_ci95':       ci95(costs),
                'DQN1200_lost_demand_mean':    np.mean(losts),
                'DQN1200_lost_demand_sd':      np.std(losts, ddof=1),
                'DQN1200_lost_demand_ci95':    ci95(losts),
                'DQN1200_bullwhip_mean':    np.mean(beis),
                'DQN1200_bullwhip_sd':      np.std(beis, ddof=1),
                'DQN1200_bullwhip_ci95':    ci95(beis),
                'p_DQN1200_vs_QL_fill_rate': p_vs_ql_fr,
                'p_DQN1200_vs_QL_cost':     p_vs_ql_k,
                'p_DQN1200_vs_DQN600_fill_rate': p_vs_dqn600_fr,
                'p_DQN1200_vs_DQN600_cost':     p_vs_dqn600_k,
            })

            # inline summary
            ql_fr_ref = (existing[mask_ql]['fill_rate'].mean()
                         if has_existing and len(ql_data)==N_RUNS else np.nan)
            d600_fr_ref = (dqn_original[mask_d600]['fill_rate'].mean()
                           if has_existing and len(d600_data)==N_RUNS else np.nan)
            print(f"         DQN-1200: FR={np.mean(frs):.2f}%  "
                  f"QL: {ql_fr_ref:.2f}%  "
                  f"DQN-600: {d600_fr_ref:.2f}%  "
                  f"p(vs QL)={p_vs_ql_fr:.4f}  "
                  f"p(vs DQN-600)={p_vs_dqn600_fr:.4f}")

# ── save ──────────────────────────────────────────────────────────
df_raw = pd.DataFrame(rows_raw)
df_agg = pd.DataFrame(rows_agg)
df_raw.to_csv(OUT_RAW, index=False)
df_agg.to_csv(OUT_AGG, index=False)
total_time = time.time() - t0

print(f"\n{'='*65}")
print(f"Done in {total_time/60:.1f} minutes")
print(f"Raw:  {len(df_raw)} rows  ->  {OUT_RAW}")
print(f"Agg:  {len(df_agg)} rows  ->  {OUT_AGG}")

# ── summary table ─────────────────────────────────────────────────
print(f"\n{'─'*65}")
print("KEY RESULT — DQN-1200 vs QL vs DQN-600 (Fill Rate %):")
print(f"{'Combo':<22} {'DQN-1200':>10} {'DQN-600':>10} {'QL':>10} {'p(1200 vs QL)':>15}")
for _, row in df_agg.iterrows():
    mu = int(row['MEAN_DEMAND']); b = int(row['LEAD_MODE'])
    D  = int(row['DISRUPT_DAYS'])
    combo = f"μ={mu} b={b} D={D}"

    if has_existing:
        mask_ql  = ((existing['MEAN_DEMAND']==mu) & (existing['LEAD_MODE']==b) &
                    (existing['DISRUPT_DAYS']==D)  & (existing['Method']=='QL'))
        mask_d60 = ((dqn_original['MEAN_DEMAND']==mu) & (dqn_original['LEAD_MODE']==b) &
                    (dqn_original['DISRUPT_DAYS']==D))
        ql_fr  = existing[mask_ql]['fill_rate'].mean()
        d60_fr = dqn_original[mask_d60]['fill_rate'].mean()
    else:
        ql_fr = d60_fr = np.nan

    p_str = f"{row['p_DQN1200_vs_QL_fill_rate']:.4f}"
    print(f"{combo:<22} {row['DQN1200_fill_rate_mean']:>9.2f}% "
          f"{d60_fr:>9.2f}% {ql_fr:>9.2f}% {p_str:>15}")

print(f"\nInterpretation guide:")
print(f"  If DQN-1200 ≈ DQN-600 (p_vs_DQN600 > 0.05): "
      f"training length is NOT the issue -> structural QL advantage confirmed.")
print(f"  If DQN-1200 >> DQN-600 (p_vs_DQN600 < 0.05) and DQN-1200 ≈ QL: "
      f"original DQN was undertrained -> Finding 3 needs revision.")
