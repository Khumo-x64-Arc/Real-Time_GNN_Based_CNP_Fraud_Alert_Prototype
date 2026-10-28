"""FSMOTE: Fuzzy-SMOTE oversampling of the fraud (minority) class.

Based on Xu, Wu, Zheng, Niu & Yang (2017), "Fuzzy-synthetic minority
oversampling technique: Oversampling based on fuzzy set theory",
International Journal of Distributed Sensor Networks 13(4),
doi:10.1177/1550147717703116.

Plain SMOTE gives every fraud the same number of synthetic neighbours. Most
frauds are easy, though; the costly mistakes are frauds that look like
legitimate transactions. FSMOTE aims the synthetic data at those:

1. Fuzzy membership. With the legitimate centroid V_p and the fraud centroid
   V_q (fuzziness m = 2), each training fraud x gets
       mu_q(x) = 1 / sum_k ( |x - V_q|^2 / |x - V_k|^2 )^(1/(m-1)),   k in {p, q}
   mu_q near 1 = a clear fraud; at or below 0.5 = closer to the legitimate class.
2. Fuzzy region. Frauds with mu_q <= `region` are oversampled (the paper
   tunes this range and finds the counts settle above ~0.7, our default). If
   fewer than two frauds fall inside, every fraud is used.
3. Allocation. The paper gives fraud j  floor((1 - mu_j) / mu_j * IR)
   synthetic samples, so lower membership gets more. Here the same
   (1 - mu) / mu weights split a fixed budget instead, so the training set
   reaches the requested fraud : legitimate `ratio` (1.0 = balanced). Pass
   ratio=None for the paper's raw counts.
4. Interpolation. For each synthetic sample pick one of the fraud's k = 5
   nearest fraud neighbours and a uniform lambda in [0, 1]:
       x_new = x + lambda * (x_neighbour - x)
   (the paper prints the difference the other way round, which would
   extrapolate away from the neighbour; its text describes interpolation).

Two practical additions:
- Distances and centroids use standardised features, otherwise the amount
  column would decide every membership on its own.
- One-hot groups (a whole categorical column) and 0/1 flags are not averaged:
  each synthetic sample takes them from whichever parent lambda is closer to,
  as SMOTE-NC does, so "country = NG and ZA at 0.4 each" never appears.
  Whole-number columns (transaction counts, hour of day) are rounded, so no
  synthetic fraud has "2.6 transactions in the past hour" — a giveaway a
  flexible model would otherwise learn instead of fraud itself.

`plan()` returns (seed row, neighbour row, lambda) triplets. The GNN applies
them in its own input space (each synthetic node mixes its parents' features
*and* graph neighbourhoods, GraphSMOTE style). `synthesize()` still turns a
plan into plain feature rows, for anyone testing the GNN against a tabular
baseline outside the app.

Only training rows are ever oversampled. Validation (threshold tuning) and
test (every reported number) stay real.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class FSMOTEPlan:
    seed: np.ndarray            # row index (into the full X) of the fraud each sample grows from
    nbr: np.ndarray             # row index of the fraud neighbour it moves towards
    lam: np.ndarray             # interpolation weight in [0, 1]
    stats: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.seed)

    @classmethod
    def empty(cls, stats=None):
        z = np.zeros(0, dtype=np.int64)
        return cls(z, z.copy(), np.zeros(0), stats or {})


class FuzzySMOTE:
    def __init__(self, ratio: float | None = 1.0, k: int = 5, m: float = 2.0, region: float = 0.7, seed: int = 42):
        if m <= 1:
            raise ValueError("Fuzziness m must be greater than 1.")
        self.ratio, self.k, self.m, self.region, self.seed = ratio, k, m, region, seed

    # ------------------------------------------------------------------ memberships
    def memberships(self, Z_min: np.ndarray, Z_maj: np.ndarray) -> np.ndarray:
        """Fuzzy membership of each minority row to the minority class (FCM formula, two fixed centroids)."""
        vq, vp = Z_min.mean(0), Z_maj.mean(0)
        dq = ((Z_min - vq) ** 2).sum(1)
        dp = ((Z_min - vp) ** 2).sum(1)
        e = 1.0 / (self.m - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(dp > 0, (dq / dp) ** e, np.inf)
        mu = 1.0 / (1.0 + r)            # = 1 / [ (dq/dq)^e + (dq/dp)^e ]
        mu[dq == 0] = 1.0               # sitting exactly on the fraud centroid
        return np.clip(mu, 1e-6, 1.0)

    # ------------------------------------------------------------------ plan
    def plan(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray) -> FSMOTEPlan:
        """Decide which synthetic frauds to create from the rows `idx` of X (training rows only)."""
        rng = np.random.default_rng(self.seed)
        idx = np.asarray(idx)
        yi = y[idx]
        min_rows, maj_rows = idx[yi == 1], idx[yi == 0]
        n_min, n_maj = len(min_rows), len(maj_rows)
        stats = {"method": "FSMOTE (Fuzzy-SMOTE, Xu et al. 2017)", "k": self.k, "m": self.m, "region": self.region,
                 "ratio": self.ratio, "minority_before": n_min, "majority": n_maj}
        if n_min < 2 or n_maj == 0:
            return FSMOTEPlan.empty({**stats, "synthetic": 0, "note": "Too few frauds to oversample."})

        mu_x, sd_x = X[idx].mean(0), X[idx].std(0) + 1e-9
        Z_min = (X[min_rows] - mu_x) / sd_x
        # the majority centroid is all that's needed, so compute it without materialising the big block
        Z_maj_mean = ((X[maj_rows].mean(0) - mu_x) / sd_x)[None, :]
        mu = self.memberships(Z_min, Z_maj_mean)

        in_region = mu <= self.region
        if in_region.sum() < 2:
            in_region[:] = True
        w = np.where(in_region, (1 - mu) / mu, 0.0)
        if w.sum() <= 0:
            w = np.ones(n_min)

        ir = n_maj / n_min
        if self.ratio is None:   # the paper's per-example counts
            counts = np.floor(w * np.floor(ir)).astype(int)
        else:
            target = max(0, int(round(self.ratio * n_maj)) - n_min)
            share = w / w.sum() * target
            counts = np.floor(share).astype(int)
            rest = target - counts.sum()    # largest remainder, so the total is exact
            if rest > 0:
                counts[np.argsort(-(share - counts))[:rest]] += 1
        total = int(counts.sum())
        stats.update({"synthetic": total, "minority_after": n_min + total, "imbalance_before": ir,
                      "imbalance_after": n_maj / max(1, n_min + total), "in_region": int(in_region.sum()),
                      "membership_hist": np.histogram(mu, np.linspace(0, 1, 11))[0].tolist(),
                      "membership_mean": float(mu.mean())})
        if total == 0:
            return FSMOTEPlan.empty(stats)

        k = min(self.k, n_min - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Z_min)
        neigh = nn.kneighbors(Z_min, return_distance=False)[:, 1:]  # drop the point itself
        seed_local = np.repeat(np.arange(n_min), counts)
        nbr_local = neigh[seed_local, rng.integers(0, k, size=total)]
        stats["used_as_seed"] = int((counts > 0).sum())
        return FSMOTEPlan(min_rows[seed_local], min_rows[nbr_local], rng.random(total), stats)

    # ------------------------------------------------------------------ feature-space samples
    @staticmethod
    def snap_groups(columns: list[str], categorical: dict, X_train: np.ndarray) -> list[list[int]]:
        """Column groups to copy from one parent instead of averaging: one-hot groups and 0/1 columns."""
        groups, used = [], set()
        for c, levels in categorical.items():
            g = [columns.index(f"{c}={v}") for v in levels]
            groups.append(g)
            used.update(g)
        for j in range(X_train.shape[1]):
            if j in used:
                continue
            vals = np.unique(X_train[:, j])
            if len(vals) <= 2 and set(vals.tolist()) <= {0.0, 1.0}:
                groups.append([j])
        return groups

    @staticmethod
    def integer_columns(X_train: np.ndarray, groups: list[list[int]]) -> list[int]:
        snapped = {j for g in groups for j in g}
        return [j for j in range(X_train.shape[1]) if j not in snapped
                and np.all(np.equal(np.mod(X_train[:, j], 1), 0))]

    @staticmethod
    def synthesize(X: np.ndarray, plan: FSMOTEPlan, groups: list[list[int]] | None = None,
                   integer_cols: list[int] | None = None) -> np.ndarray:
        if not len(plan):
            return np.zeros((0, X.shape[1]))
        a, b = X[plan.seed], X[plan.nbr]
        out = a + plan.lam[:, None] * (b - a)
        if integer_cols:
            out[:, integer_cols] = np.round(out[:, integer_cols])
        if groups:
            from_b = plan.lam >= 0.5
            for g in groups:
                out[np.ix_(from_b, g)] = b[np.ix_(from_b, g)]
                out[np.ix_(~from_b, g)] = a[np.ix_(~from_b, g)]
        return out
