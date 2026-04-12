"""
run_rop_sensitivity.py
======================
Sensitivity analysis on ROP reorder threshold R ∈ {60, 80, 100}.
Purpose: demonstrate that QL advantage over ROP is robust to threshold
calibration, not an artefact of suboptimal ROP parameterisation.

Classical methods only (ROP and (s,S)) — no RL retraining needed.
Fast: ~2-5 minutes total.

Design:
  3 R values × 36 combinations × 30 runs = 3,240 simulations
  (s,S) also tested with matching s values: s ∈ {60, 80, 100}, S=300 fixed

Output:
  rop_sensitivity_results.csv  – 108 aggregate rows (3 R × 36 combos × ROP + (s,S))
  rop_sensitivity_summary.csv  – mean FR and cost per R value

Place in same folder as results_raw.csv.
"""

import numpy as np
import pandas as pd
from scipy import stats
import time

DEMANDS     = [15, 20, 25]
LEAD_MODES  = [3, 7, 14]
DISRUPTIONS = [10, 30, 60, 90]
N_RUNS      = 30
EVAL_SEED   = 1000

R_VALUES    = [60, 80, 100]   # reorder thresholds to test
Q_FIXED     = 150             # order quantity fixed
S_FIXED     = 300             # (s,S) upper level fixed
C_H = 0.50; C_L = 10.0; C_ORDER = 100.0

OUT_AGG  = 'rop_sensitivity_results.csv'
OUT_SUM  = 'rop_sensitivity_summary.csv'

# load QL results for comparison
try:
    old = pd.read_csv('results_raw.csv')
    has_ql = True
    print(f"Loaded existing results: {len(old)} rows")
except FileNotFoundError:
    has_ql = False
    print("Warning: results_raw.csv not found – QL comparison skipped")


# ── Environment ───────────────────────────────────────────────────
class Env:
    def __init__(self, mu, b, D, seed):
        self.rng = np.random.default_rng(seed)
        self.mu  = mu; self.b = b; self.D = D
        self.d0  = int(self.rng.integers(60, 201))
        self.stock = 200; self.in_transit = []
        self.day = 0; self.sold = 0; self.dem = 0
        self.lost = 0; self.cost = 0.0; self.orders = []

    def _lt(self):
        return int(np.ceil(self.rng.triangular(2, self.b, 14)))

    def _disrupted(self):
        return self.d0 <= self.day < self.d0 + self.D

    @property
    def in_tr_qty(self):
        return sum(q for _, q in self.in_transit)

    def step(self, qty):
        self.orders.append(qty)
        if qty > 0 and not self._disrupted():
            self.in_transit.append((self.day + self._lt(), qty))
        recv = sum(q for d,q in self.in_transit if d <= self.day)
        self.in_transit = [(d,q) for d,q in self.in_transit if d > self.day]
        self.stock = min(self.stock + recv, 600)
        dem = int(self.rng.poisson(self.mu))
        self.dem  += dem
        sold = min(dem, self.stock); lost = dem - sold
        self.stock -= sold; self.sold += sold; self.lost += lost
        self.cost += (C_H*self.stock +
                      (C_ORDER if qty>0 else 0) +
                      C_L*lost)
        self.day += 1

    def metrics(self):
        fr  = self.sold/self.dem*100 if self.dem else 0
        bei = np.std(self.orders)/np.sqrt(self.mu) if self.orders else 0
        return {'fill_rate':fr,'cost':self.cost,
                'lost_demand':self.lost,'bullwhip':bei}


def run_rop(R, mu, b, D, n_runs):
    results = []
    for i in range(n_runs):
        env = Env(mu, b, D, EVAL_SEED+i)
        for _ in range(365):
            eff = env.stock + env.in_tr_qty
            qty = Q_FIXED if eff <= R else 0
            env.step(qty)
        results.append(env.metrics())
    return results


def run_ss(s, mu, b, D, n_runs):
    results = []
    for i in range(n_runs):
        env = Env(mu, b, D, EVAL_SEED+i)
        for _ in range(365):
            eff = env.stock + env.in_tr_qty
            if eff < s:
                qty_need = S_FIXED - eff
                # round to nearest 50
                qty = int(round(qty_need/50)*50)
                qty = max(0, min(qty, 250))
            else:
                qty = 0
            env.step(qty)
        results.append(env.metrics())
    return results


def ci95(v): return 1.96*np.std(v,ddof=1)/np.sqrt(len(v))
def mwu(a,b):
    _,p = stats.mannwhitneyu(a,b,alternative='two-sided')
    return p


# ── Main loop ─────────────────────────────────────────────────────
rows_agg = []
total = len(R_VALUES)*len(DEMANDS)*len(LEAD_MODES)*len(DISRUPTIONS)
done_n = 0
t0 = time.time()

print(f"\nROP sensitivity: {len(R_VALUES)} R-values × 36 combinations × {N_RUNS} runs")

for R in R_VALUES:
    print(f"\n--- R = {R} ---")
    for mu in DEMANDS:
        for b in LEAD_MODES:
            for D in DISRUPTIONS:
                done_n += 1
                elapsed = time.time()-t0
                eta = (elapsed/done_n*(total-done_n)) if done_n>1 else 0
                print(f"[{done_n:3d}/{total}] R={R} μ={mu} b={b} D={D}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

                rop_res = run_rop(R,  mu, b, D, N_RUNS)
                ss_res  = run_ss( R,  mu, b, D, N_RUNS)

                rop_frs  = np.array([r['fill_rate'] for r in rop_res])
                rop_ksts = np.array([r['cost']     for r in rop_res])

                # QL comparison p-value
                p_ql_vs_rop = np.nan
                if has_ql:
                    mask = ((old['MEAN_DEMAND']==mu)&(old['LEAD_MODE']==b)&
                            (old['DISRUPT_DAYS']==D)&(old['Method']=='QL'))
                    ql_sub = old[mask]
                    if len(ql_sub)==N_RUNS:
                        p_ql_vs_rop = mwu(ql_sub['fill_rate'].values, rop_frs)

                for method, res in [('ROP',rop_res),('(s,S)',ss_res)]:
                    frs = np.array([r['fill_rate'] for r in res])
                    kst = np.array([r['cost']     for r in res])
                    rows_agg.append({
                        'R_value':        R,
                        's_value':        R,
                        'MEAN_DEMAND':   mu,
                        'LEAD_MODE':   b,
                        'DISRUPT_DAYS':   D,
                        'Method':         method,
                        'fill_rate_mean': np.mean(frs),
                        'fill_rate_sd':   np.std(frs,ddof=1),
                        'fill_rate_ci95': ci95(frs),
                        'koszt_mean':     np.mean(kst),
                        'koszt_ci95':     ci95(kst),
                        'p_QL_vs_ROP':    p_ql_vs_rop if method=='ROP' else np.nan,
                    })

# ── Save ──────────────────────────────────────────────────────────
df = pd.DataFrame(rows_agg)
df.to_csv(OUT_AGG, index=False)
print(f"\nDone in {time.time()-t0:.0f}s")
print(f"Saved: {OUT_AGG} ({len(df)} rows)")

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'─'*55}")
print("SUMMARY — ROP Fill Rate by threshold R:")
print(f"{'R':>5} {'FR mean':>10} {'FR range':>18} "
      f"{'Cost mean':>12} {'p(QL>ROP) sig':>15}")

for R in R_VALUES:
    sub = df[(df['R_value']==R)&(df['Method']=='ROP')]
    fr_m = sub['fill_rate_mean'].mean()
    fr_min = sub['fill_rate_mean'].min()
    fr_max = sub['fill_rate_mean'].max()
    k_m  = sub['koszt_mean'].mean()
    p_col = sub['p_QL_vs_ROP'].dropna()
    sig  = (p_col < 0.05).sum()
    n    = len(p_col)
    print(f"{R:>5}  {fr_m:>9.2f}%  "
          f"[{fr_min:.1f}–{fr_max:.1f}]%  "
          f"{k_m:>12,.0f}  {sig}/{n} combos")

# Save concise summary
rows_sum = []
for R in R_VALUES:
    for method in ['ROP','(s,S)']:
        sub = df[(df['R_value']==R)&(df['Method']==method)]
        if has_ql and method == 'ROP':
            # QL baseline for comparison
            ql_fr = old[old['Method']=='QL']['fill_rate'].mean()
        rows_sum.append({
            'R_value': R,
            'Method': method,
            'fill_rate_mean': sub['fill_rate_mean'].mean(),
            'fill_rate_min':  sub['fill_rate_mean'].min(),
            'fill_rate_max':  sub['fill_rate_mean'].max(),
            'koszt_mean':     sub['koszt_mean'].mean(),
            'sig_QL_vs_ROP':  int((sub['p_QL_vs_ROP'].dropna()<0.05).sum()),
        })

pd.DataFrame(rows_sum).to_csv(OUT_SUM, index=False)
print(f"Summary saved: {OUT_SUM}")
