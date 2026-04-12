"""
dqn_numpy.py
============
DQN agent implemented in pure NumPy (no PyTorch/TensorFlow required).

Architecture : 3 -> 64 -> 64 -> 6
Training     : Mnih et al. (2015) – replay buffer + target network + Adam
State space  : continuous [stock_norm, in_transit_norm, disruption_flag]
Action space : {0, 50, 100, 150, 200, 250} units  (identical to QL/SARSA)
Reward       : identical to QL/SARSA agents
"""

import numpy as np
import random
from collections import deque


# ══════════════════════════════════════════════════════════════════
# Supply Chain Environment
# ══════════════════════════════════════════════════════════════════
class SupplyChainEnv:
    """Three-echelon supply chain with stochastic lead times and disruptions.
    State is continuous (3-dim float), identical dynamics to QL/SARSA version.
    """

    ACTIONS = [0, 50, 100, 150, 200, 250]

    def __init__(self, mean_demand=20, lead_mode=7, disruption_days=30,
                 disruption_start=None, seed=None):
        self.rng              = np.random.default_rng(seed)
        self.mean_demand      = mean_demand
        self.lead_mode        = lead_mode
        self.disruption_days  = disruption_days
        self.disruption_start = (disruption_start
                                 if disruption_start is not None
                                 else int(self.rng.integers(60, 201)))
        self.capacity = 600
        self.c_hold   = 0.50
        self.c_order  = 100.0
        self.c_lost   = 10.0
        self.reset()

    def reset(self):
        self.stock       = 200
        self.in_transit  = []   # list of (arrival_day, qty)
        self.day         = 0
        self.total_sold  = 0
        self.total_dem   = 0
        self.total_lost  = 0
        self.total_cost  = 0.0
        self.orders_list = []
        return self._state()

    def _state(self):
        """Continuous state vector: [stock/cap, in_transit/cap, disruption_flag]"""
        in_tr    = sum(q for _, q in self.in_transit)
        flag     = 1.0 if (self.disruption_start <= self.day <
                           self.disruption_start + self.disruption_days) else 0.0
        return np.array([self.stock / self.capacity,
                         in_tr     / self.capacity,
                         flag], dtype=np.float32)

    def _lead_time(self):
        a, b, c = 2, self.lead_mode, 14
        return int(np.ceil(self.rng.triangular(a, b, c)))

    def step(self, action_idx):
        qty       = self.ACTIONS[action_idx]
        disrupted = (self.disruption_start <= self.day <
                     self.disruption_start + self.disruption_days)

        self.orders_list.append(qty)

        if qty > 0 and not disrupted:
            self.in_transit.append((self.day + self._lead_time(), qty))

        received, still = [], []
        for (arr, q) in self.in_transit:
            (received if arr <= self.day else still).append((arr, q))
        self.in_transit = still
        self.stock = min(self.stock + sum(q for _, q in received), self.capacity)

        demand          = int(self.rng.poisson(self.mean_demand))
        self.total_dem += demand
        sold            = min(demand, self.stock)
        lost            = demand - sold
        self.stock     -= sold
        self.total_sold += sold
        self.total_lost += lost

        cost = (self.c_hold * self.stock +
                (self.c_order if qty > 0 else 0) +
                self.c_lost * lost)
        self.total_cost += cost

        fr_t   = sold / demand if demand > 0 else 1.0
        reward = 10.0 * fr_t - cost / 500.0

        self.day += 1
        return self._state(), reward, self.day >= 365

    def metrics(self):
        fr  = (self.total_sold / self.total_dem * 100
               if self.total_dem else 0.0)
        bei = (np.std(self.orders_list) / np.sqrt(self.mean_demand)
               if self.orders_list else 0.0)
        return {'fill_rate': fr,
                'cost':     self.total_cost,
                'lost_demand':  self.total_lost,
                'bullwhip':  bei}


# ══════════════════════════════════════════════════════════════════
# NumPy MLP  (2 hidden layers, ReLU, Adam optimiser)
# ══════════════════════════════════════════════════════════════════
class NumpyMLP:
    """Fully-connected network: in_dim -> hidden -> hidden -> out_dim.
    Activations: ReLU. Optimiser: Adam (Kingma & Ba, 2015).
    """

    def __init__(self, in_dim=3, hidden=64, out_dim=6, lr=1e-3, seed=0):
        rng    = np.random.default_rng(seed)
        s1, s2 = np.sqrt(2.0 / in_dim), np.sqrt(2.0 / hidden)

        self.W1 = rng.normal(0, s1, (hidden, in_dim )).astype(np.float32)
        self.b1 = np.zeros(hidden,  dtype=np.float32)
        self.W2 = rng.normal(0, s2, (hidden, hidden )).astype(np.float32)
        self.b2 = np.zeros(hidden,  dtype=np.float32)
        self.W3 = rng.normal(0, s2, (out_dim, hidden)).astype(np.float32)
        self.b3 = np.zeros(out_dim, dtype=np.float32)

        self.lr  = lr
        self.t   = 0
        self.b1_ = 0.9; self.b2_ = 0.999; self.eps_ = 1e-8
        names    = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        self.m   = {n: np.zeros_like(getattr(self, n)) for n in names}
        self.v   = {n: np.zeros_like(getattr(self, n)) for n in names}

    # ── activation helpers ────────────────────────────────────────
    @staticmethod
    def _relu(x):  return np.maximum(0.0, x)
    @staticmethod
    def _drelu(x): return (x > 0).astype(np.float32)

    # ── forward pass ─────────────────────────────────────────────
    def forward(self, X):
        """X: (batch, in_dim)  ->  Q: (batch, out_dim)"""
        self.X  = X
        self.z1 = X  @ self.W1.T + self.b1;  self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2.T + self.b2; self.a2 = self._relu(self.z2)
        self.z3 = self.a2 @ self.W3.T + self.b3
        return self.z3

    # ── backward pass + Adam update ───────────────────────────────
    def backward(self, dL_dout):
        """dL_dout: (batch, out_dim)"""
        B = self.X.shape[0]
        dW3 = dL_dout.T @ self.a2 / B;  db3 = dL_dout.mean(0)
        da2 = dL_dout @ self.W3
        dz2 = da2 * self._drelu(self.z2)
        dW2 = dz2.T @ self.a1 / B;      db2 = dz2.mean(0)
        da1 = dz2 @ self.W2
        dz1 = da1 * self._drelu(self.z1)
        dW1 = dz1.T @ self.X / B;       db1 = dz1.mean(0)

        for name, grad in [('W1',dW1),('b1',db1),
                            ('W2',dW2),('b2',db2),
                            ('W3',dW3),('b3',db3)]:
            self._adam(name, grad)

    def _adam(self, name, grad):
        self.t += 1
        self.m[name] = self.b1_ * self.m[name] + (1 - self.b1_) * grad
        self.v[name] = self.b2_ * self.v[name] + (1 - self.b2_) * grad**2
        m_h = self.m[name] / (1 - self.b1_**self.t)
        v_h = self.v[name] / (1 - self.b2_**self.t)
        setattr(self, name,
                getattr(self, name) - self.lr * m_h / (np.sqrt(v_h) + self.eps_))

    # ── inference ─────────────────────────────────────────────────
    def predict(self, x):
        """Single state (3,) -> Q-values (6,)"""
        a1 = self._relu(x @ self.W1.T + self.b1)
        a2 = self._relu(a1 @ self.W2.T + self.b2)
        return (a2 @ self.W3.T + self.b3).ravel()

    def copy_weights_from(self, other):
        for attr in ['W1','b1','W2','b2','W3','b3']:
            setattr(self, attr, getattr(other, attr).copy())


# ══════════════════════════════════════════════════════════════════
# Replay Buffer
# ══════════════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s.copy(), int(a), float(r), s2.copy(), float(done)))

    def sample(self, batch_size):
        batch       = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.int64),
                np.array(r,  dtype=np.float32),
                np.array(s2, dtype=np.float32),
                np.array(d,  dtype=np.float32))

    def __len__(self): return len(self.buf)


# ══════════════════════════════════════════════════════════════════
# DQN Agent
# ══════════════════════════════════════════════════════════════════
class DQNAgent:
    """Standard DQN (Mnih et al., 2015):
       - Experience replay buffer
       - Target network with periodic hard update
       - ε-greedy exploration with exponential decay
    """

    def __init__(self, seed=42, lr=1e-3, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.994,
                 buffer_size=10_000, batch_size=64, target_update=10):
        random.seed(seed)
        np.random.seed(seed)

        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self.eps           = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.n_actions     = 6
        self.steps         = 0

        self.policy = NumpyMLP(lr=lr, seed=seed)
        self.target = NumpyMLP(lr=lr, seed=seed)
        self.target.copy_weights_from(self.policy)
        self.buffer = ReplayBuffer(buffer_size)

    def act(self, state, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.policy.predict(state)))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        s, a, r, s2, done = self.buffer.sample(self.batch_size)

        Q_all  = self.policy.forward(s)
        Q_next = self.target.forward(s2).max(axis=1)
        Q_tgt  = r + self.gamma * Q_next * (1 - done)

        # MSE gradient only for the taken action
        dL = np.zeros_like(Q_all)
        for i in range(self.batch_size):
            dL[i, a[i]] = 2.0 * (Q_all[i, a[i]] - Q_tgt[i]) / self.batch_size

        self.policy.backward(dL)

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target.copy_weights_from(self.policy)

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def train(self, mean_demand, lead_mode, disruption_days,
              episodes=600, train_seed_base=4000):
        for ep in range(episodes):
            env   = SupplyChainEnv(mean_demand, lead_mode, disruption_days,
                                   seed=train_seed_base + ep)
            state = env.reset()
            done  = False
            while not done:
                action              = self.act(state)
                next_s, rew, done   = env.step(action)
                self.buffer.push(state, action, rew, next_s, done)
                self.update()
                state = next_s
            self.decay_eps()

    def evaluate(self, mean_demand, lead_mode, disruption_days,
                 n_runs=30, eval_seed_base=1000):
        results = []
        for i in range(n_runs):
            env   = SupplyChainEnv(mean_demand, lead_mode, disruption_days,
                                   seed=eval_seed_base + i)
            state = env.reset()
            done  = False
            while not done:
                action        = self.act(state, greedy=True)
                state, _, done = env.step(action)
            results.append(env.metrics())
        return results
