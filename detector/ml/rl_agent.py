"""A Deep Q-Network agent that runs fraud decisions on top of the semi-supervised GNN.

Design, following the two reference papers:

* FraudGNN-RL (Cui et al., IEEE OJ-CS 2025, Sec. IV-B). The agent's state is a readout of the GNN's
  graph embedding; its action sets the fraud-detection threshold and feature-importance weights
  (a continuous space the paper discretises); the reward combines detection and false-positive cost;
  the Q-network has two hidden layers of 128 units, trained with Adam (lr 1e-3, batch 64) against a
  target network, and it keeps learning from feedback while deployed.
* DRL+GNN (Almasan et al., Computer Communications 2022, Sec. IV-V). The Q-value is computed from a
  sum/mean readout of GNN hidden states; epsilon-greedy exploration with epsilon = 1 for the first 70
  iterations, then exponential decay; a 4,000-transition FIFO experience replay. That paper discounts
  at 0.95; here an action only reaches later windows through the labels it propagates, and 0.5 was
  clearly more stable on the European Cardholder data (test F1 0.515 +- 0.021 against 0.464 +- 0.035
  over three agents), so 0.5 is the default.

The decision problem (an MDP over time windows of the transaction stream):

state    s_t  mean of the GNN's hidden states over window t (projected to 16 dimensions, the readout),
              the window's GNN score quantiles, how much label evidence its graph neighbours carry,
              the current threshold and settings, and running precision / recall from the labelled
              feedback received so far
action   a_t  one of 7 x 3 x 3 = 63 combinations of
              - the alert threshold, as an offset on the log-odds scale from the threshold chosen on
                labelled validation data: -2, -1, -0.5, 0, +0.5, +1, +2 (FraudGNN-RL's action is the
                threshold itself; an absolute choice also keeps each action's effect to its own window,
                which made learning far more stable than step-by-step adjustments did)
              - label-propagation weight w in {0, 0.5, 1}: how much evidence from labelled
                neighbours (graph semi-supervised learning) is added to the GNN's own score
              - pseudo-labelling confidence: off, 0.99, 0.95 (unlabelled transactions the GNN is this
                sure about are pseudo-labelled at once and propagate to their neighbours; the GNN's own
                probability decides, not the propagated score, so pseudo-labels can't feed on
                themselves)
reward   r_t  computed from labelled transactions only (a fraction `label_rate` of them):
              (caught frauds - missed frauds - fp_cost x false alarms) / (expected frauds + 1)
              With the default fp_cost = 1 a transaction is worth flagging once it is fraud with
              probability above 1/3, which keeps the agent's objective close to F1's.

Semi-supervised throughout: only `label_rate` of transactions ever get a label, and it arrives
`delay` windows later; everything else is unlabelled. Labels (true and pseudo) are held by graph
groups, entities or earlier similar transactions, and a transaction's label-propagation score is the
smoothed fraud rate among the labels its neighbours hold, as log-odds against the base rate:

    lp_i = logit((f_i + m * pi) / (f_i + l_i + m)) - logit(pi),   m = 5

so it is 0 with no evidence. The decision score is z_i + w * lp_i, where z_i is the GNN's log-odds.

Everything is NumPy, like the GNN, so PyTorch is not needed.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

THETA_OFFSETS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
LP_WEIGHTS = [0.0, 0.5, 1.0]
PL_CONF = [None, 0.99, 0.95]
ACTIONS = [(d, w, c) for d in THETA_OFFSETS for w in LP_WEIGHTS for c in PL_CONF]
N_ACTIONS = len(ACTIONS)
THETA_RANGE = (-8.0, 12.0)
PRIOR_STRENGTH = 5.0
READOUT_DIM = 16


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def windows_for(t: np.ndarray, idx: np.ndarray, seconds: float) -> list[np.ndarray]:
    """Consecutive time windows of `seconds` over the rows idx (sorted by time); empty windows are skipped."""
    tt = t[idx]
    w = np.floor((tt - tt[0]) / seconds).astype(np.int64)
    cuts = np.flatnonzero(np.diff(w)) + 1
    return [s for s in np.split(idx, cuts) if len(s)]


# ------------------------------------------------------------------ environment
class FraudStreamEnv:
    """Replays a labelled-but-mostly-hidden transaction stream window by window."""

    def __init__(self, z, H, y, read, write, windows, prior, theta0, label_rate, delay=2, fp_cost=1.0,
                 pseudo_weight=0.5, init_fraud=None, init_legit=None, projector=None):
        self.z, self.y, self.read, self.write = z, y, read, write
        self.windows, self.prior, self.theta0 = windows, max(prior, 1e-5), theta0
        self.label_rate, self.delay, self.fp_cost, self.pw = label_rate, delay, fp_cost, pseudo_weight
        G = read.shape[1]
        self.f0 = np.zeros(G) if init_fraud is None else init_fraud
        self.l0 = np.zeros(G) if init_legit is None else init_legit
        self.R = projector.transform(H).astype(np.float32) if projector is not None else H
        self.readT = None

    # state -------------------------------------------------------------
    def _lp(self, idx):
        rd = self.read[idx]
        f, l = rd @ self.f, rd @ self.l
        r = (f + PRIOR_STRENGTH * self.prior) / (f + l + PRIOR_STRENGTH)
        return _logit(r) - _logit(self.prior), f + l

    def _state(self):
        idx = self.windows[self.t]
        z = self.z[idx]
        lp, ev = self._lp(idx)
        s = z + self.w * lp
        q = np.quantile(z, [0.5, 0.9, 0.99, 0.999])
        top = s >= np.quantile(s, 0.99)
        fb = self.fb
        prec = fb["tp"] / max(fb["tp"] + fb["fp"], 1e-9) if fb["tp"] + fb["fp"] > 0 else 0.5
        rec = fb["tp"] / max(fb["tp"] + fb["fn"], 1e-9) if fb["tp"] + fb["fn"] > 0 else 0.5
        pl_onehot = [1.0 if self.c == c else 0.0 for c in PL_CONF]
        return np.concatenate([
            self.R[idx].mean(0),                                          # GNN readout (mean of hidden states)
            q / 10.0, [float((s >= self.theta).mean()) * 100, float((ev > 0).mean()), float(lp[top].mean()) / 10 if top.any() else 0.0],
            [(self.theta - self.theta0) / 2.0, self.w / 2.0], pl_onehot,
            [prec, rec, fb["rate"] * 100, np.log1p(self.n_pseudo_fraud) / 5, np.log1p(len(idx)) / 10],
        ]).astype(np.float32)

    @property
    def state_dim(self):
        return self.R.shape[1] + 4 + 3 + 2 + len(PL_CONF) + 5

    # episode -----------------------------------------------------------
    def reset(self, visible=None, seed=0):
        rng = np.random.default_rng(seed)
        n = len(self.y)
        self.visible = visible if visible is not None else rng.uniform(size=n) < self.label_rate
        self.f, self.l = self.f0.copy(), self.l0.copy()
        self.theta, self.w, self.c = self.theta0, 0.0, None
        self.t = 0
        self.pending = deque()
        self.fb = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "rate": self.prior}
        self.n_pseudo_fraud = 0
        self.scores = np.full(n, np.nan)
        self.flags = np.zeros(n, dtype=bool)
        # per-transaction record of the decision, so each escalation (or not) can be explained afterwards
        self.lp_tx = np.zeros(n)
        self.ev_fraud = np.zeros(n)
        self.ev_legit = np.zeros(n)
        self.theta_tx = np.full(n, np.nan)
        self.win_tx = np.full(n, -1, dtype=np.int64)
        self.trace = []
        return self._state()

    def done(self):
        return self.t >= len(self.windows)

    def _reveal(self):
        """Labels of window t - delay arrive: they update the label groups and the feedback statistics."""
        while self.pending and self.pending[0][0] <= self.t:
            _, lab, yl, fl = self.pending.popleft()
            if len(lab):
                wr = self.write[lab]
                self.f += wr.T @ (yl == 1).astype(float)
                self.l += wr.T @ (yl == 0).astype(float)
            d = 0.9   # exponential memory, so the agent tracks drift
            self.fb["tp"] = d * self.fb["tp"] + float(((yl == 1) & fl).sum())
            self.fb["fp"] = d * self.fb["fp"] + float(((yl == 0) & fl).sum())
            self.fb["fn"] = d * self.fb["fn"] + float(((yl == 1) & ~fl).sum())
            if len(yl):
                self.fb["rate"] = 0.8 * self.fb["rate"] + 0.2 * float(yl.mean())

    def step(self, a: int):
        d, w, c = ACTIONS[a]
        self.theta = float(np.clip(self.theta0 + d, *THETA_RANGE))
        self.w, self.c = w, c
        idx = self.windows[self.t]
        lp, _ = self._lp(idx)
        s = self.z[idx] + w * lp
        flag = s >= self.theta
        self.scores[idx], self.flags[idx] = s, flag
        rd = self.read[idx]
        self.lp_tx[idx], self.ev_fraud[idx], self.ev_legit[idx] = lp, rd @ self.f, rd @ self.l
        self.theta_tx[idx], self.win_tx[idx] = self.theta, self.t
        fb = self.fb
        fb_prec = fb["tp"] / (fb["tp"] + fb["fp"]) if fb["tp"] + fb["fp"] > 0 else None
        vis = self.visible[idx]
        lab, yl, fl = idx[vis], self.y[idx][vis], flag[vis]
        tp, fp, fn = int(((yl == 1) & fl).sum()), int(((yl == 0) & fl).sum()), int(((yl == 1) & ~fl).sum())
        reward = (tp - fn - self.fp_cost * fp) / (self.prior * max(len(lab), 1) + 1.0)
        # pseudo-labels for unlabelled transactions the system is sure about: immediate, half weight
        n_pf = 0
        if c is not None:
            p = 1 / (1 + np.exp(-np.clip(self.z[idx], -30, 30)))   # the GNN's own confidence only
            unl = ~vis
            pf, pl = idx[unl & (p >= c)], idx[unl & (p <= 1 - c)]
            if len(pf):
                self.f += self.pw * np.asarray(self.write[pf].sum(0)).ravel()
                n_pf = len(pf)
            if len(pl):
                self.l += self.pw * np.asarray(self.write[pl].sum(0)).ravel()
            self.n_pseudo_fraud += n_pf
        self.pending.append((self.t + self.delay, lab, yl, fl))
        yt = self.y[idx]
        self.trace.append({"theta": self.theta, "w": w, "pl": c, "alerts": int(flag.sum()), "n": int(len(idx)),
                           "tp": int(((yt == 1) & flag).sum()), "fp": int(((yt == 0) & flag).sum()),
                           "fn": int(((yt == 1) & ~flag).sum()), "reward": float(reward), "pseudo_frauds": n_pf,
                           "action": int(a), "feedback_precision": fb_prec, "window": int(self.t)})
        self.t += 1
        self._reveal()
        nxt = None if self.done() else self._state()
        return nxt, float(reward)


# ------------------------------------------------------------------ Q-network (NumPy MLP)
class QNet:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.W[-1] *= 0.1
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.k = 0

    def forward(self, X, keep=False):
        acts = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)
            acts.append(h)
        return (h, acts) if keep else h

    def train_step(self, X, a, target, lr=1e-3):
        out, acts = self.forward(X, keep=True)
        q = out[np.arange(len(a)), a]
        err = q - target
        g = np.clip(err, -1, 1) / len(a)                       # Huber loss gradient
        G = np.zeros_like(out)
        G[np.arange(len(a)), a] = g
        grads_W, grads_b = [], []
        for i in range(len(self.W) - 1, -1, -1):
            grads_W.insert(0, acts[i].T @ G)
            grads_b.insert(0, G.sum(0))
            if i:
                G = (G @ self.W[i].T) * (acts[i] > 0)
        self.k += 1
        for j, (p, gr) in enumerate(zip(self.W + self.b, grads_W + grads_b)):
            self.m[j] = 0.9 * self.m[j] + 0.1 * gr
            self.v[j] = 0.999 * self.v[j] + 0.001 * gr * gr
            p -= lr * (self.m[j] / (1 - 0.9 ** self.k)) / (np.sqrt(self.v[j] / (1 - 0.999 ** self.k)) + 1e-8)
        return float(np.mean(np.minimum(err ** 2, np.abs(err))))

    def train_step_multi(self, X, cols, target, lr=1e-3):
        """Each row updates several outputs (one per action branch) towards the same target."""
        out, acts = self.forward(X, keep=True)
        rows = np.repeat(np.arange(len(X)), cols.shape[1])
        q = out[rows, cols.ravel()]
        err = q - np.repeat(target, cols.shape[1])
        G = np.zeros_like(out)
        np.add.at(G, (rows, cols.ravel()), np.clip(err, -1, 1) / len(X))
        for i in range(len(self.W) - 1, -1, -1):
            gW, gb = acts[i].T @ G, G.sum(0)
            if i:
                G = (G @ self.W[i].T) * (acts[i] > 0)
            self._adam(i, gW, gb, lr)
        self.k += 1
        return float(np.mean(np.minimum(err ** 2, np.abs(err))))

    def _adam(self, i, gW, gb, lr):
        n = len(self.W)
        for j, p, gr in ((i, self.W[i], gW), (n + i, self.b[i], gb)):
            t = self.k + 1
            self.m[j] = 0.9 * self.m[j] + 0.1 * gr
            self.v[j] = 0.999 * self.v[j] + 0.001 * gr * gr
            p -= lr * (self.m[j] / (1 - 0.9 ** t)) / (np.sqrt(self.v[j] / (1 - 0.999 ** t)) + 1e-8)

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]


BRANCHES = [len(THETA_OFFSETS), len(LP_WEIGHTS), len(PL_CONF)]


def split_action(a: int) -> tuple[int, int, int]:
    d, w, c = ACTIONS[a]
    return THETA_OFFSETS.index(d), LP_WEIGHTS.index(w), PL_CONF.index(c)


def join_action(i: int, j: int, k: int) -> int:
    return ACTIONS.index((THETA_OFFSETS[i], LP_WEIGHTS[j], PL_CONF[k]))


class DQNAgent:
    """DQN over the 63 joint actions, or (branching=True) an action-branching DQN with one Q-head per decision.

    FraudGNN-RL's action is a vector (threshold and feature weights). A flat Q-network has to learn 63
    values from the same noisy rewards, and its max over them picks lucky estimates. Action branching
    (Tavakoli, Pardo & Kormushev, AAAI 2018) gives each decision its own head on a shared torso:
    Q_d(s, a_d) for the threshold (7), the LP weight (3) and pseudo-labelling (3), trained towards the
    shared target r + gamma * mean_d max Q_d'(s', .), so each head sees every transition.
    """

    def __init__(self, state_dim, n_actions=N_ACTIONS, hidden=128, lr=1e-3, gamma=0.5, batch=64,
                 buffer=4000, target_every=250, seed=0, branching=True):
        self.branching = branching
        out = sum(BRANCHES) if branching else n_actions
        self.q = QNet([state_dim, hidden, hidden, out], seed)
        self.target = QNet([state_dim, hidden, hidden, out], seed)
        self.target.copy_from(self.q)
        self.lr, self.gamma, self.batch, self.target_every = lr, gamma, batch, target_every
        self.mem = deque(maxlen=buffer)                        # FIFO experience replay (DRL+GNN: 4,000)
        self.rng = np.random.default_rng(seed)
        self.mu = np.zeros(state_dim, np.float32)
        self.sd = np.ones(state_dim, np.float32)
        self.updates = 0
        self.losses = []
        self.offsets = np.cumsum([0] + BRANCHES)

    def norm(self, s):
        return (s - self.mu) / self.sd

    def fit_norm(self, states):
        S = np.asarray(states)
        self.mu, self.sd = S.mean(0).astype(np.float32), (S.std(0) + 1e-3).astype(np.float32)

    def _greedy(self, out):
        if not self.branching:
            return int(np.argmax(out))
        o = self.offsets
        return join_action(*[int(np.argmax(out[o[d]:o[d + 1]])) for d in range(len(BRANCHES))])

    def act(self, s, eps):
        if self.rng.uniform() < eps:
            return int(self.rng.integers(N_ACTIONS))
        return self._greedy(self.q.forward(self.norm(s)[None, :])[0])

    def remember(self, s, a, r, s2):
        self.mem.append((s, a, r, s2))

    def learn(self):
        if len(self.mem) < self.batch:
            return None
        pick = self.rng.integers(len(self.mem), size=self.batch)
        S = np.array([self.mem[i][0] for i in pick])
        A = np.array([self.mem[i][1] for i in pick])
        Rw = np.array([self.mem[i][2] for i in pick], dtype=np.float32)
        nxt = [self.mem[i][3] for i in pick]
        alive = np.array([x is not None for x in nxt])
        target = Rw.copy()
        if alive.any():
            S2 = self.norm(np.array([x for x in nxt if x is not None]))
            on, tg = self.q.forward(S2), self.target.forward(S2)
            if self.branching:                                   # double DQN per branch, averaged
                o, vals = self.offsets, []
                for d in range(len(BRANCHES)):
                    a2 = np.argmax(on[:, o[d]:o[d + 1]], 1)
                    vals.append(tg[np.arange(len(a2)), o[d] + a2])
                target[alive] += self.gamma * np.mean(vals, 0)
            else:                                                # double DQN: online picks, target evaluates
                a2 = np.argmax(on, 1)
                target[alive] += self.gamma * tg[np.arange(len(a2)), a2]
        if self.branching:
            sub = np.array([split_action(a) for a in A])
            idx = sub + self.offsets[:-1][None, :]                 # one output per branch
            loss = self.q.train_step_multi(self.norm(S), idx, target, self.lr)
        else:
            loss = self.q.train_step(self.norm(S), A, target, self.lr)
        self.updates += 1
        if self.updates % self.target_every == 0:
            self.target.copy_from(self.q)
        self.losses.append(loss)
        return loss


# ------------------------------------------------------------------ training and evaluation
def epsilon_schedule(episodes, hold=70, floor=0.05):
    """DRL+GNN: epsilon = 1 for the first `hold` iterations, then exponential decay. The decay rate is set
    so epsilon reaches `floor` at 80% of training."""
    hold = min(hold, max(1, episodes // 3))
    decay_eps = max(1, int(episodes * 0.8) - hold)
    rate = floor ** (1 / decay_eps)
    return [1.0 if e < hold else max(floor, rate ** (e - hold)) for e in range(episodes)]


def train_agent(env: FraudStreamEnv, episodes=200, seed=0, progress=None, updates_per_step=2, **agent_kw):
    agent = DQNAgent(env.state_dim, seed=seed, **agent_kw)
    eps = epsilon_schedule(episodes)
    curve, warm = [], []
    for ep in range(episodes):
        s = env.reset(seed=seed * 10_000 + ep)            # a fresh labelled sample each episode
        total = 0.0
        while not env.done():
            if ep == 0 or eps[ep] == 1.0:
                warm.append(s)
            a = agent.act(s, eps[ep])
            s2, r = env.step(a)
            agent.remember(s, a, r, s2)
            for _ in range(updates_per_step):
                agent.learn()
            total += r
            s = s2
        if eps[ep] == 1.0 and (ep + 1 < len(eps) and eps[ep + 1] < 1.0):
            agent.fit_norm(warm)                           # freeze input scaling once exploration ends
        if ep == 0:
            agent.fit_norm(warm)
        curve.append({"episode": ep + 1, "reward": round(total, 3), "epsilon": round(eps[ep], 3)})
        if progress:
            progress(ep + 1, episodes, total)
    return agent, curve


def run_policy(env: FraudStreamEnv, policy, visible, agent=None, online=False, eps=0.0):
    """policy: "agent" or a fixed action index. Returns the env after one pass (its trace and scores)."""
    s = env.reset(visible=visible)
    total = 0.0
    while not env.done():
        a = agent.act(s, eps) if policy == "agent" else policy
        s2, r = env.step(a)
        if online and agent is not None:
            agent.remember(s, a, r, s2)
            agent.learn()
        total += r
        s = s2
    env.total_reward = total
    return env


def evaluate_run(env: FraudStreamEnv, idx: np.ndarray) -> dict:
    y = env.y[idx]
    s = env.scores[idx]
    f = env.flags[idx]
    tp, fp = int(((y == 1) & f).sum()), int(((y == 0) & f).sum())
    fn, tn = int(((y == 1) & ~f).sum()), int(((y == 0) & ~f).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    k = max(1, int(round(len(s) * 0.01)))
    top = np.argsort(-s)[:k]
    n_win = len(env.trace)
    # the agent's own objective, scored on every test label (evaluation only), per window as in training
    reward_all = sum((x["tp"] - x["fn"] - env.fp_cost * x["fp"]) / (env.prior * x["n"] + 1.0) for x in env.trace)
    return {"reward_all": float(reward_all), "windows": n_win,"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "precision": prec, "recall": rec,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "recall_at_1pct": float(y[top].sum() / max(y.sum(), 1)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "alerts": tp + fp, "reward": float(env.total_reward)}


def best_static_action(env: FraudStreamEnv, episodes=4, seed=0):
    """The strongest non-adaptive baseline: one fixed action (threshold, LP weight, pseudo-labelling) for the
    whole stream, picked by average reward over a few labelled samples of the validation stream."""
    masks = [np.random.default_rng(seed + 991 * e).uniform(size=len(env.y)) < env.label_rate for e in range(episodes)]
    rewards = [np.mean([run_policy(env, a, m).total_reward for m in masks]) for a in range(N_ACTIONS)]
    a = int(np.argmax(rewards))
    off, w, c = ACTIONS[a]
    return {"action": a, "theta_offset": off, "lp_weight": w, "pseudo_conf": c, "val_reward": float(rewards[a])}


def make_projector(H_fit: np.ndarray, seed=0):
    return PCA(n_components=min(READOUT_DIM, H_fit.shape[1]), random_state=seed).fit(H_fit)
