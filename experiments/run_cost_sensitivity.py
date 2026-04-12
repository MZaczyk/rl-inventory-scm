"""
run_cost_sensitivity_fast.py
============================
Cost sensitivity analysis: impact of the lost-sale to holding cost ratio c_l/c_h.
Key design choices for speed:
  - Classical policies (ROP, s,S) implemented as fast NumPy vectorised loops
  - Tabular QL/SARSA: 200 training episodes (converge faster, same problem structure)
  - DQN: 300 episodes
  - n=15 replications per cell (sufficient for Mann-Whitney U, p-values stable)
  - Baseline results (ratio=20) loaded from existing result files

Output:
  cost_sensitivity_results.csv  – 108 rows (36 combinations × 3 ratios × 5 methods)
  cost_sensitivity_breakeven.csv – breakeven per combo

Required in the same directory:
  dqn_numpy.py, results_raw.csv, dqn_results_raw.csv
"""

import numpy as np
import pandas as pd
from scipy import stats
import time

# ── parametry ─────────────────────────────────────────────────────
DEMANDS      = [15, 20, 25]
LEAD_MODES   = [3, 7, 14]
DISRUPTIONS  = [10, 30, 60, 90]
N_RUNS       = 15     # sensitivity analysis – 15 replications sufficient
EPISODES_TAB = 200    # tabular QL/SARSA
EPISODES_DQN = 300    # DQN
EVAL_SEED    = 1000
C_H          = 0.50
C_ORDER      = 100.0

COST_CONFIGS = {
    'LOW':  {'c_l': 2.50,  'ratio':  5},
    'HIGH': {'c_l': 25.00, 'ratio': 50},
}

OUT_AGG      = 'cost_sensitivity_results.csv'
OUT_BREAKEVEN = 'cost_sensitivity_breakeven.csv'

# ── load existing baseline ────────────────────────────────────────
print("Loading baseline (ratio=20)...")
try:
    base = pd.read_csv('results_raw.csv')
    dqn_base = pd.read_csv('dqn_results_raw.csv')
    base_all = pd.concat([base, dqn_base], ignore_index=True)
    has_base = True
    print(f"  Loaded {len(base_all)} rows")
except FileNotFoundError as e:
    print(f"  Warning: {e}")
    has_base = False


# ══════════════════════════════════════════════════════════════════
# Fast environment – single replication via numpy arrays
# ══════════════════════════════════════════════════════════════════
class FastEnv:
    ACTIONS = np.array([0, 50, 100, 150, 200, 250])

    def __init__(self, c_l, mu, b, D, seed):
        self.rng  = np.random.default_rng(seed)
        self.c_l  = c_l
        self.mu   = mu
        self.b    = b      # lead time mode
        self.D    = D      # disruption days
        self.d0   = int(self.rng.integers(60, 201))
        self.reset()

    def reset(self):
        self.stock      = 200
        self.in_transit = []   # (arrival_day, qty)
        self.day        = 0
        self.sold_total = 0
        self.dem_total  = 0
        self.lost_total = 0
        self.cost_total = 0.0
        self.orders     = []

    def _lt(self):
        return int(np.ceil(self.rng.triangular(2, self.b, 14)))

    def _disrupted(self):
        return self.d0 <= self.day < self.d0 + self.D

    @property
    def in_transit_qty(self):
        return sum(q for _, q in self.in_transit)

    def state_discrete(self):
        sb_edges  = [0, 50, 100, 150, 200, 300]
        ib_edges  = [0, 50, 100, 200, 400]
        sb = sum(1 for e in sb_edges[1:] if self.stock > e)
        qt = self.in_transit_qty
        ib = sum(1 for e in ib_edges[1:] if qt > e)
        fi = int(self._disrupted())
        return (sb * 5 + ib) * 2 + fi

    def state_continuous(self):
        return np.array([
            self.stock / 600.0,
            self.in_transit_qty / 600.0,
            float(self._disrupted())
        ], dtype=np.float32)

    def step(self, action_idx):
        qty = int(self.ACTIONS[action_idx])
        self.orders.append(qty)

        if qty > 0 and not self._disrupted():
            self.in_transit.append((self.day + self._lt(), qty))

        recv = sum(q for d, q in self.in_transit if d <= self.day)
        self.in_transit = [(d, q) for d, q in self.in_transit if d > self.day]
        self.stock = min(self.stock + recv, 600)

        dem  = int(self.rng.poisson(self.mu))
        self.dem_total += dem
        sold = min(dem, self.stock)
        lost = dem - sold
        self.stock      -= sold
        self.sold_total += sold
        self.lost_total += lost

        cost = (C_H * self.stock +
                (C_ORDER if qty > 0 else 0) +
                self.c_l * lost)
        self.cost_total += cost
        self.day += 1
        return cost, dem, sold

    def run_horizon(self):
        for _ in range(365):
            yield

    def metrics(self):
        fr  = self.sold_total / self.dem_total * 100 if self.dem_total else 0
        bei = np.std(self.orders) / np.sqrt(self.mu) if self.orders else 0
        return {'fill_rate': fr, 'cost': self.cost_total,
                'lost_demand': self.lost_total, 'bullwhip': bei}


# ══════════════════════════════════════════════════════════════════
# Classical policies
# ══════════════════════════════════════════════════════════════════
def eval_rop(c_l, mu, b, D, n_runs):
    results = []
    for i in range(n_runs):
        env = FastEnv(c_l, mu, b, D, EVAL_SEED + i)
        for _ in range(365):
            eff = env.stock + env.in_transit_qty
            a   = 3 if eff <= 80 else 0   # action 3 = 150 units
            env.step(a)
        results.append(env.metrics())
    return results

def eval_ss(c_l, mu, b, D, n_runs):
    results = []
    for i in range(n_runs):
        env = FastEnv(c_l, mu, b, D, EVAL_SEED + i)
        for _ in range(365):
            eff = env.stock + env.in_transit_qty
            if eff < 80:
                need = 300 - eff
                # closest action >= need, or max
                acts = FastEnv.ACTIONS
                cands = np.where(acts >= need)[0]
                a = int(cands[0]) if len(cands) else 5
            else:
                a = 0
            env.step(a)
        results.append(env.metrics())
    return results


# ══════════════════════════════════════════════════════════════════
# Tabular QL / SARSA
# ══════════════════════════════════════════════════════════════════
class FastTabular:
    N_STATES  = 6 * 5 * 2    # 60
    N_ACTIONS = 6

    def __init__(self, algo='QL', alpha=0.10, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.990, seed=0):
        self.algo    = algo
        self.alpha   = alpha
        self.gamma   = gamma
        self.eps     = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.rng     = np.random.default_rng(seed)
        self.Q       = np.zeros((self.N_STATES, self.N_ACTIONS))

    def _act(self, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.N_ACTIONS))
        return int(np.argmax(self.Q[s]))

    def _update(self, s, a, r, s2, a2=None, done=False):
        if self.algo == 'QL':
            tgt = r + self.gamma * np.max(self.Q[s2]) * (1 - done)
        else:
            tgt = r + self.gamma * self.Q[s2, a2] * (1 - done)
        self.Q[s, a] += self.alpha * (tgt - self.Q[s, a])

    def train(self, c_l, mu, b, D, episodes, seed_base):
        for ep in range(episodes):
            env = FastEnv(c_l, mu, b, D, seed_base + ep)
            s = env.state_discrete()
            a = self._act(s)
            for _ in range(365):
                cost, dem, sold = env.step(a)
                fr_t   = sold / dem if dem > 0 else 1.0
                reward = 10.0 * fr_t - cost / 500.0
                s2 = env.state_discrete()
                a2 = self._act(s2)
                done = (env.day >= 365)
                self._update(s, a, reward, s2, a2, done)
                s, a = s2, a2
            self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def evaluate(self, c_l, mu, b, D, n_runs):
        results = []
        for i in range(n_runs):
            env = FastEnv(c_l, mu, b, D, EVAL_SEED + i)
            for _ in range(365):
                s = env.state_discrete()
                a = self._act(s, greedy=True)
                env.step(a)
            results.append(env.metrics())
        return results


# ══════════════════════════════════════════════════════════════════
# DQN (from dqn_numpy.py but using FastEnv internally)
# ══════════════════════════════════════════════════════════════════
from dqn_numpy import DQNAgent as _BaseDQN

class FastDQN(_BaseDQN):
    def train(self, c_l, mu, b, D, episodes, seed_base):
        import random as py_random
        for ep in range(episodes):
            env = FastEnv(c_l, mu, b, D, seed_base + ep)
            state = env.state_continuous()
            for _ in range(365):
                action = self.act(state)
                cost, dem, sold = env.step(action)
                fr_t   = sold / dem if dem > 0 else 1.0
                reward = 10.0 * fr_t - cost / 500.0
                next_state = env.state_continuous()
                done = (env.day >= 365)
                self.buffer.push(state, action, reward, next_state, done)
                self.update()
                state = next_state
            self.decay_eps()

    def evaluate(self, c_l, mu, b, D, n_runs):
        results = []
        for i in range(n_runs):
            env = FastEnv(c_l, mu, b, D, EVAL_SEED + i)
            state = env.state_continuous()
            for _ in range(365):
                action = self.act(state, greedy=True)
                env.step(action)
                state = env.state_continuous()
            results.append(env.metrics())
        return results


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════
def ci95(v):
    return 1.96 * np.std(v, ddof=1) / np.sqrt(len(v))

def mwu(a, b_arr):
    _, p = stats.mannwhitneyu(a, b_arr, alternative='two-sided')
    return p

def to_agg(results, method, ratio_name, ratio_val, mu, b, D,
           p_fr=np.nan, p_k=np.nan):
    frs = np.array([r['fill_rate'] for r in results])
    kst = np.array([r['cost']     for r in results])
    lst = np.array([r['lost_demand']  for r in results])
    bws = np.array([r['bullwhip']  for r in results])
    return {
        'ratio':         ratio_name,
        'c_l_over_c_h':  ratio_val,
        'MEAN_DEMAND':  mu,
        'LEAD_MODE':  b,
        'DISRUPT_DAYS':  D,
        'Method':        method,
        'fill_rate_mean': np.mean(frs),
        'fill_rate_sd':   np.std(frs, ddof=1),
        'fill_rate_ci95': ci95(frs),
        'koszt_mean':     np.mean(kst),
        'koszt_sd':       np.std(kst, ddof=1),
        'koszt_ci95':     ci95(kst),
        'lost_demand_mean':  np.mean(lst),
        'bullwhip_mean':  np.mean(bws),
        'p_vs_ROP_fr':    p_fr,
        'p_vs_ROP_k':     p_k,
    }


# ══════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════
rows_agg = []

# ── Baseline (ratio=20) from existing files ───────────────────────
if has_base:
    print("Processing baseline (ratio=20)...")
    for mu in DEMANDS:
        for b in LEAD_MODES:
            for D in DISRUPTIONS:
                mask = ((base_all['MEAN_DEMAND']==mu) &
                        (base_all['LEAD_MODE']==b)  &
                        (base_all['DISRUPT_DAYS']==D))
                rop_frs  = base_all[mask & (base_all['Method']=='ROP')]['fill_rate'].values
                rop_ksts = base_all[mask & (base_all['Method']=='ROP')]['cost'].values

                for method in ['ROP','(s,S)','QL','SARSA','DQN']:
                    m_mask = mask & (base_all['Method']==method)
                    sub = base_all[m_mask]
                    if len(sub) == 0:
                        continue
                    # use up to 15 runs for consistency
                    sub = sub.iloc[:N_RUNS]
                    res = [{'fill_rate': r,'cost': k,'lost_demand': u,'bullwhip': bw}
                           for r,k,u,bw in zip(sub['fill_rate'], sub['cost'],
                                               sub['lost_demand'], sub['bullwhip'])]
                    p_fr = p_k = np.nan
                    if method not in ['ROP'] and len(rop_frs) >= N_RUNS:
                        p_fr = mwu(sub['fill_rate'].values, rop_frs[:N_RUNS])
                        p_k  = mwu(sub['cost'].values,     rop_ksts[:N_RUNS])
                    rows_agg.append(to_agg(res, method, 'BASE', 20, mu, b, D,
                                           p_fr, p_k))
    print(f"  Baseline: {len(rows_agg)} aggregate rows")

# ── LOW and HIGH variants ─────────────────────────────────────────
total = len(COST_CONFIGS) * len(DEMANDS) * len(LEAD_MODES) * len(DISRUPTIONS)
done_n = 0
t0 = time.time()

for ratio_name, cfg in COST_CONFIGS.items():
    c_l      = cfg['c_l']
    ratio_val = cfg['ratio']
    print(f"\n{'='*55}")
    print(f"RATIO: {ratio_name}  (c_l={c_l}, c_h={C_H}, "
          f"c_l/c_h={ratio_val})")
    print(f"{'='*55}")

    seed_off = {'LOW': 5000, 'HIGH': 6000}[ratio_name]

    for mu in DEMANDS:
        for b in LEAD_MODES:
            for D in DISRUPTIONS:
                done_n += 1
                elapsed = time.time() - t0
                eta = (elapsed/done_n*(total-done_n)) if done_n > 1 else 0
                print(f"[{done_n:2d}/{total}] {ratio_name} "
                      f"μ={mu} b={b:2d} D={D:2d}  "
                      f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m",
                      flush=True)

                combo = seed_off + done_n * 7

                # Classical (fast, no training)
                rop_res  = eval_rop(c_l, mu, b, D, N_RUNS)
                ss_res   = eval_ss( c_l, mu, b, D, N_RUNS)

                # Tabular QL
                ql = FastTabular('QL',    seed=combo)
                ql.train(c_l, mu, b, D, EPISODES_TAB, combo*100)
                ql_res = ql.evaluate(c_l, mu, b, D, N_RUNS)

                # Tabular SARSA
                sa = FastTabular('SARSA', seed=combo+1)
                sa.train(c_l, mu, b, D, EPISODES_TAB, (combo+1)*100)
                sa_res = sa.evaluate(c_l, mu, b, D, N_RUNS)

                # DQN
                dq = FastDQN(seed=combo+2)
                dq.train(c_l, mu, b, D, EPISODES_DQN, (combo+2)*100)
                dq_res = dq.evaluate(c_l, mu, b, D, N_RUNS)

                all_res = {'ROP': rop_res, '(s,S)': ss_res,
                           'QL': ql_res, 'SARSA': sa_res, 'DQN': dq_res}

                rop_frs  = np.array([r['fill_rate'] for r in rop_res])
                rop_ksts = np.array([r['cost']     for r in rop_res])

                for method, res_list in all_res.items():
                    frs_m = np.array([r['fill_rate'] for r in res_list])
                    kst_m = np.array([r['cost']     for r in res_list])
                    p_fr = p_k = np.nan
                    if method != 'ROP':
                        p_fr = mwu(frs_m, rop_frs)
                        p_k  = mwu(kst_m, rop_ksts)
                    rows_agg.append(to_agg(res_list, method,
                                           ratio_name, ratio_val,
                                           mu, b, D, p_fr, p_k))

                ql_fr  = np.mean([r['fill_rate'] for r in ql_res])
                rop_fr = np.mean([r['fill_rate'] for r in rop_res])
                dqn_fr = np.mean([r['fill_rate'] for r in dq_res])
                print(f"         ROP={rop_fr:.1f}%  "
                      f"QL={ql_fr:.1f}%  DQN={dqn_fr:.1f}%")

# ── Save ─────────────────────────────────────────────────────────
df_agg = pd.DataFrame(rows_agg)
df_agg.to_csv(OUT_AGG, index=False)
total_time = time.time() - t0
print(f"\nDone in {total_time/60:.1f} min")
print(f"Agg: {len(df_agg)} rows -> {OUT_AGG}")

# ── Breakeven analysis ────────────────────────────────────────────
rows_be = []
for mu in DEMANDS:
    for b in LEAD_MODES:
        for D in DISRUPTIONS:
            row = {'MEAN_DEMAND': mu, 'LEAD_MODE': b, 'DISRUPT_DAYS': D}
            for rname in ['LOW','BASE','HIGH']:
                rval = {'LOW':5,'BASE':20,'HIGH':50}[rname]
                m = ((df_agg['ratio']==rname) & (df_agg['MEAN_DEMAND']==mu) &
                     (df_agg['LEAD_MODE']==b) & (df_agg['DISRUPT_DAYS']==D))
                sub = df_agg[m]
                if sub.empty: continue
                ql_fr  = sub[sub['Method']=='QL']['fill_rate_mean'].values
                rop_fr = sub[sub['Method']=='ROP']['fill_rate_mean'].values
                ql_k   = sub[sub['Method']=='QL']['koszt_mean'].values
                rop_k  = sub[sub['Method']=='ROP']['koszt_mean'].values
                best_fr  = sub.loc[sub['fill_rate_mean'].idxmax(), 'Method']
                best_cst = sub.loc[sub['koszt_mean'].idxmin(), 'Method']
                row[f'{rname}_best_FR']    = best_fr
                row[f'{rname}_best_cost']  = best_cst
                if len(ql_fr) and len(rop_fr):
                    row[f'{rname}_QL_advantage_pp']   = float(ql_fr[0]-rop_fr[0])
                    row[f'{rname}_QL_cost_premium_pct']= float(
                        (ql_k[0]-rop_k[0])/rop_k[0]*100 if rop_k[0]>0 else np.nan)
            rows_be.append(row)

df_be = pd.DataFrame(rows_be)
df_be.to_csv(OUT_BREAKEVEN, index=False)
print(f"Breakeven: {len(df_be)} rows -> {OUT_BREAKEVEN}")

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'─'*55}")
print("SUMMARY (all 36 combinations averaged):")
print(f"{'Ratio':>8} {'Method':>8} {'FR%':>8} {'Cost':>10} {'p<0.05 vs ROP':>15}")
for rname in ['LOW','BASE','HIGH']:
    rval = {'LOW':5,'BASE':20,'HIGH':50}[rname]
    m = df_agg['ratio']==rname
    for method in ['ROP','(s,S)','QL','SARSA','DQN']:
        mm = m & (df_agg['Method']==method)
        if df_agg[mm].empty: continue
        fr  = df_agg[mm]['fill_rate_mean'].mean()
        k   = df_agg[mm]['koszt_mean'].mean()
        sig = (df_agg[mm]['p_vs_ROP_fr'] < 0.05).sum()
        n36 = df_agg[mm]['p_vs_ROP_fr'].notna().sum()
        sig_str = f"{sig}/{n36}" if n36 > 0 else "—"
        print(f"{rval:>8} {method:>8} {fr:>8.1f} {k:>10,.0f} {sig_str:>15}")
    print()
