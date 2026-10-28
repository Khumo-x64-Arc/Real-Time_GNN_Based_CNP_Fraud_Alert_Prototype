"""A two-layer GraphSAGE-style GNN on a heterogeneous transaction graph.

Graph: every transaction is a node; it links to the entities it touches
(account, card, device, IP, merchant, terminal). Messages flow
transaction -> entity (mean) -> transaction (mean), with separate weights per
entity type, so a transaction "sees" what other transactions on the same
device / IP / account look like. That is what lets it catch fraud rings and
bursts that per-row models miss.

Written with NumPy + SciPy sparse (hand-derived gradients, Adam) so it runs
anywhere without PyTorch. It is inductive: a brand-new transaction is scored
by averaging over the stored per-entity embeddings, which are then updated.

Oversampling (FSMOTE). Synthetic fraud nodes can't simply be added to the
graph: they would send messages into real transactions' neighbourhoods. As in
GraphSMOTE (Zhao et al., 2021), they are interpolated in representation space
instead. For seed fraud i, neighbour fraud j and weight lambda, the synthetic
node's layer-1 input is (1-lambda) z_i + lambda z_j (its own features *and* its
aggregated neighbourhood), and its layer-2 neighbourhood message is the same
mix of i's and j's. Gradients flow back through both parents, and synthetic
nodes only ever receive messages, so real nodes are untouched.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score


def build_incidence(keys: list[str], index_map: dict | None = None):
    """keys[i] = entity key of transaction i (None = no entity). Returns (B, index_map)."""
    index_map = {} if index_map is None else index_map
    rows, cols = [], []
    for i, k in enumerate(keys):
        if k is None:
            continue
        j = index_map.setdefault(k, len(index_map))
        rows.append(i)
        cols.append(j)
    B = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(keys), max(len(index_map), 1)))
    return B, index_map


class _Agg:
    """Mean aggregation P = rownorm(B) @ colnorm(B).T, applied without materialising n x n."""

    def __init__(self, B: sp.csr_matrix):
        deg_tx = np.asarray(B.sum(1)).ravel()
        deg_ent = np.asarray(B.sum(0)).ravel()
        self.R = sp.diags(1 / np.maximum(deg_tx, 1)) @ B            # n x m
        self.C = (B @ sp.diags(1 / np.maximum(deg_ent, 1))).tocsr()  # n x m
        self.has = (deg_tx > 0).astype(float)[:, None]

    def fwd(self, H):   # P @ H ; rows without entity fall back to themselves
        return self.R @ (self.C.T @ H) + (1 - self.has) * H

    def bwd(self, G):   # P.T @ G
        return self.C @ (self.R.T @ G) + (1 - self.has) * G


class _AdjAgg:
    """Mean over a node's listed neighbours *and itself*: P = D^-1 (A + I) for a sparse adjacency A.

    Used for similarity graphs (e.g. the k most similar earlier transactions within an hour, as in
    FraudGNN-RL for the anonymised European cardholder data). If A only points to earlier rows, every
    aggregation is causal, so full-graph scores equal streaming scores.
    """

    def __init__(self, A: sp.csr_matrix):
        n = A.shape[0]
        A = (sp.csr_matrix(A) != 0).astype(float)
        A = A + sp.identity(n, format="csr")
        self.P = (sp.diags(1 / np.asarray(A.sum(1)).ravel()) @ A).tocsr()
        self.PT = self.P.T.tocsr()

    def fwd(self, H):
        return self.P @ H

    def bwd(self, G):
        return self.PT @ G


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


class TransactionGNN:
    def __init__(self, hidden=64, epochs=300, lr=0.01, weight_decay=1e-4, patience=40, seed=7, dtype=np.float32):
        self.hidden, self.epochs, self.lr = hidden, epochs, lr
        self.wd, self.patience, self.seed, self.dtype = weight_decay, patience, seed, dtype
        self.history: list[dict] = []

    # ---------------------------------------------------------------- training
    def fit(self, X, entity_keys: dict[str, list], y, train_idx, val_idx, synth=None, progress=None,
            sample_weight=None, syn_weight=None, relations: dict | None = None, init=None):
        """synth: an FSMOTEPlan whose seed/nbr are node indices of training frauds (optional).

        relations: optional {name: n x n sparse adjacency} for similarity graphs, alongside (or instead of)
        the entity links in entity_keys. Each relation gets its own layer-2 weights, like an entity type.

        sample_weight (one per train_idx) and syn_weight (one per synthetic node) scale each node's share
        of the loss on top of the class weighting, e.g. to give a small, weakly labelled domain a set share.

        init: a trained TransactionGNN with the same entity types, relations and feature count to start
        from (fine-tuning for transfer learning). Its weights and its feature scaling are kept as the
        starting point, so what it learned on the source data is the prior the new labels adjust.
        """
        rng = np.random.default_rng(self.seed)
        relations = relations or {}
        self.types = list(entity_keys)
        self.relation_types = list(relations)
        tr_sd = X[train_idx].std(0)
        new_feats = np.zeros(X.shape[1], bool)
        if init is not None:
            if list(init.types) != self.types or list(getattr(init, "relation_types", [])) != list(relations) \
                    or init.W1.shape[0] != X.shape[1] * (len(self.types) + len(relations) + 1) or init.hidden != self.hidden:
                raise ValueError("init must be a GNN with the same entity types, relations, features and hidden size")
            self.mu, self.sd = init.mu.copy(), init.sd.copy()
            # A feature that was constant in the source (a field that file doesn't have) has no learned weights
            # and a near-zero spread: scaled by the source it would blow up. It is scaled on the new data and
            # its weights start at 0, so it enters the model only as far as the new labels support it.
            new_feats = ~init.live_features() & (tr_sd > 1e-8)
            self.mu[new_feats] = X[train_idx][:, new_feats].mean(0)
            self.sd[new_feats] = tr_sd[new_feats] + 1e-6
        else:
            self.mu, self.sd = X[train_idx].mean(0), tr_sd + 1e-6
        self.live = tr_sd > 1e-8
        self.seen_types = [t for t in self.types if any(k is not None for k in entity_keys[t])]
        Xs = self._scale(X)
        aggs = [_Agg(build_incidence(entity_keys[t])[0]) for t in self.types] + \
               [_AdjAgg(relations[r]) for r in self.relation_types]
        K, d, h = len(aggs), X.shape[1], self.hidden

        Z0 = np.hstack([Xs.astype(self.dtype)] + [a.fwd(Xs).astype(self.dtype) for a in aggs])  # fixed layer-1 input
        W1 = rng.normal(0, np.sqrt(2 / Z0.shape[1]), (Z0.shape[1], h))
        b1 = np.zeros(h)
        V = rng.normal(0, np.sqrt(1 / h), (K + 1, h))       # layer-2: self + one per type
        c = np.zeros(1)
        if init is not None:
            W1, b1, V, c = init.W1.copy(), init.b1.copy(), init.V.copy(), init.c.copy()
            # An entity type the source never had (every key missing) fell back to the transaction itself, so
            # its weights were trained as a copy of the self weights. Fold them into the self weights (the
            # source model's function is unchanged) and start the relation at 0, to be learned from new labels.
            for k, t in enumerate(self.types):
                if not init.saw_type(t) and t in self.seen_types:
                    blk = slice((k + 1) * d, (k + 2) * d)
                    W1[:d] += W1[blk]
                    W1[blk] = 0.0
                    V[0] += V[k + 1]
                    V[k + 1] = 0.0
            if new_feats.any():   # Z0 = [x, agg_1(x), ..., agg_K(x)]: the feature's row in every block
                rows = np.concatenate([np.flatnonzero(new_feats) + k * d for k in range(K + 1)])
                W1[rows] = 0.0
        params = [np.array(prm, dtype=self.dtype) for prm in (W1, b1, V, c)]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]

        ytr = y[train_idx].astype(self.dtype)
        n_syn = len(synth) if synth is not None else 0
        if n_syn:   # M mixes the two parents of every synthetic node: (1 - lambda) * seed + lambda * neighbour
            n = X.shape[0]
            M = sp.csr_matrix((np.concatenate([1 - synth.lam, synth.lam]),
                               (np.concatenate([np.arange(n_syn)] * 2), np.concatenate([synth.seed, synth.nbr]))),
                              shape=(n_syn, n))
        pos = max(ytr.sum() + n_syn, 1)
        neg = len(ytr) - ytr.sum()
        pw = min(max(neg / pos, 1.0), 50.0)   # whatever imbalance FSMOTE left (1.0 once balanced)
        wts = np.where(ytr == 1, pw, 1.0)
        ws = np.full(n_syn, pw)
        if sample_weight is not None:
            wts = wts * np.asarray(sample_weight, dtype=float)
        if syn_weight is not None and n_syn:
            ws = ws * np.asarray(syn_weight, dtype=float)
        tot = wts.sum() + ws.sum()
        wts, ws = (wts / tot).astype(self.dtype), ws / tot
        self.pos_weight = float(pw)

        best, best_params, bad = -1, None, 0
        ctx = {"Z0": Z0, "aggs": aggs, "train_idx": train_idx, "ytr": ytr, "wts": wts,
               "M": M.astype(self.dtype) if n_syn else None, "ws": ws.astype(self.dtype)}
        if init is not None and y[val_idx].sum():
            # fine-tuning: the source weights themselves are a candidate, so a few noisy target labels
            # can't make the result worse (on validation) than not fine-tuning at all
            _, _, prob0 = self._loss_grad(params, ctx)
            best = average_precision_score(y[val_idx], prob0[val_idx])
            best_params = [p.copy() for p in params]
            self.history.append({"epoch": 0, "val_ap": float(best)})
        for ep in range(1, self.epochs + 1):
            _, grads, prob = self._loss_grad(params, ctx)
            for i, (p, gr) in enumerate(zip(params, grads)):  # Adam
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                p -= self.lr * (m[i] / (1 - 0.9 ** ep)) / (np.sqrt(v[i] / (1 - 0.999 ** ep)) + 1e-8)

            if ep % 5 == 0:
                ap = average_precision_score(y[val_idx], prob[val_idx]) if y[val_idx].sum() else 0
                self.history.append({"epoch": ep, "val_ap": float(ap)})
                if progress:
                    progress(ep, float(ap))
                if ap > best:
                    best, bad = ap, 0
                    best_params = [p.copy() for p in params]
                else:
                    bad += 5
                    if bad >= self.patience:
                        break
        self.W1, self.b1, self.V, self.c = best_params or params
        self.best_val_ap = best
        self._Z0, self._aggs = Z0, aggs
        return self

    def saw_type(self, t) -> bool:
        """Whether training data had links of entity type t (older models: assume yes)."""
        seen = getattr(self, "seen_types", None)
        return True if seen is None else t in seen

    def _known_keys(self, entity_keys: dict) -> dict:
        """Links of a type the model never saw in training are dropped: it only learned that type as
        'no neighbours' (the transaction itself), so real neighbours would be fed to weights never trained on them."""
        return {t: (v if self.saw_type(t) else [None] * len(v)) for t, v in entity_keys.items()}

    def live_features(self) -> np.ndarray:
        """Features that varied in the training data (older models: inferred from the stored spread)."""
        live = getattr(self, "live", None)
        return np.asarray(live, bool) if live is not None else self.sd > 1e-4

    def _scale(self, X):
        """Standardise; a feature that was constant in training is held at 0 (its mean), whatever value it
        takes in new data, because the model never learned what it means."""
        Xs = (X - self.mu) / self.sd
        dead = ~self.live_features()
        if dead.any():
            Xs = np.array(Xs, copy=True)
            Xs[..., dead] = 0.0
        return Xs

    def _loss_grad(self, params, ctx):
        """Weighted BCE over real training nodes and synthetic frauds, with exact gradients.

        Layer 2 is linear in H1, so P_k (H1 V_k) is computed as P_k applied to the
        *vector* H1 @ V_k: no n x hidden neighbourhood matrix is ever built. A
        synthetic node's pre-activation is M @ A1 (M's rows sum to 1), so it needs
        no input copy either. Memory stays ~4 n x hidden floats at any size.
        """
        W1, b1, V, c = params
        Z0, aggs, train_idx, ytr, wts = ctx["Z0"], ctx["aggs"], ctx["train_idx"], ctx["ytr"], ctx["wts"]
        M, ws = ctx["M"], ctx["ws"]
        K = len(aggs)
        A1 = Z0 @ W1 + b1
        H1 = np.maximum(A1, 0)
        HV = H1 @ V.T                                         # n x (K+1): self term + one per entity type
        nb = [aggs[k].fwd(HV[:, k + 1:k + 2])[:, 0] for k in range(K)]
        prob = _sigmoid(HV[:, 0] + sum(nb) + c[0])
        pt = prob[train_idx]
        loss = -float((wts * (ytr * np.log(pt + 1e-12) + (1 - ytr) * np.log(1 - pt + 1e-12))).sum())
        g = np.zeros_like(prob)
        g[train_idx] = (pt - ytr) * wts
        g_nb = g.copy()                                       # gradient reaching the neighbourhood terms
        if M is not None:   # synthetic frauds (label 1): own path + the parents' mixed neighbourhood terms
            As = M @ A1
            Hs = np.maximum(As, 0)
            ps = _sigmoid(Hs @ V[0] + M @ sum(nb) + c[0])
            loss -= float((ws * np.log(ps + 1e-12)).sum())
            gs = (ps - 1.0) * ws
            g_nb = g_nb + M.T @ gs
        U = np.column_stack([g] + [aggs[k].bwd(g_nb[:, None])[:, 0] for k in range(K)])   # n x (K+1)
        gV = (H1.T @ U).T
        gc = np.array([g.sum()])
        gH1 = U @ V
        if M is not None:
            gV[0] += Hs.T @ gs
            gc += gs.sum()
            gA1 = gH1 * (A1 > 0) + M.T @ (np.outer(gs, V[0]) * (As > 0))
        else:
            gA1 = gH1 * (A1 > 0)
        gW1 = Z0.T @ gA1 + self.wd * W1
        gb1 = gA1.sum(0)
        loss += 0.5 * self.wd * float((W1 * W1).sum() + (V * V).sum())
        return loss, [gW1, gb1, gV + self.wd * V, gc], prob

    def _aggs_for(self, X, entity_keys, relations):
        n = X.shape[0]
        relations = relations or {}
        aggs = [_Agg(build_incidence(entity_keys.get(t) or [None] * n)[0]) for t in self.types]
        for r in getattr(self, "relation_types", []):
            aggs.append(_AdjAgg(relations[r]) if r in relations else _AdjAgg(sp.csr_matrix((n, n))))
        return aggs

    def _forward(self, Z0, aggs):
        H1 = np.maximum(Z0 @ self.W1 + self.b1, 0)
        HV = H1 @ self.V.T
        logit = HV[:, 0] + sum(a.fwd(HV[:, k + 1:k + 2])[:, 0] for k, a in enumerate(aggs)) + self.c[0]
        return logit, H1

    def embed_new(self, X, entity_keys: dict[str, list], relations: dict | None = None):
        """(log-odds, layer-1 hidden states) for every transaction of a graph the model hasn't seen.

        Entity types the model wasn't trained on are ignored; a type or relation the new data lacks behaves
        like a transaction with no neighbours of that kind (pass all-None keys / leave a relation out to
        rescore without those links).
        """
        Xs = self._scale(X).astype(self.dtype)
        aggs = self._aggs_for(X, self._known_keys(entity_keys), relations)
        Z0 = np.hstack([Xs] + [a.fwd(Xs).astype(self.dtype) for a in aggs])
        return self._forward(Z0, aggs)

    def predict_new(self, X, entity_keys: dict[str, list], relations: dict | None = None):
        """Score every transaction of a graph the model has never seen (no labels, no training)."""
        return _sigmoid(self.embed_new(X, entity_keys, relations)[0])

    def embed_graph(self):
        """(log-odds, layer-1 hidden states) for every node of the graph used in fit."""
        return self._forward(self._Z0, self._aggs)

    def predict_graph(self):
        """Probabilities for every node of the graph used in fit (transductive eval)."""
        return _sigmoid(self.embed_graph()[0])

    # ------------------------------------------------------- live (inductive)
    def build_inference_state(self, X_hist, keys_hist: dict[str, list]):
        """Store per-entity sums of x and h1 over historical transactions."""
        Xs = self._scale(X_hist)
        keys_hist = self._known_keys(keys_hist)
        self.state = {}
        agg_x = []
        for t in self.types:
            B, idx = build_incidence(keys_hist[t])
            cnt = np.asarray(B.sum(0)).ravel()
            sx = B.T @ Xs
            self.state[t] = {"idx": idx, "cnt": list(cnt), "sx": [r for r in sx], "sh": None}
            agg_x.append(_Agg(B).fwd(Xs))
        H1 = np.maximum(np.hstack([Xs] + agg_x) @ self.W1 + self.b1, 0)
        for t in self.types:
            B, _ = build_incidence(keys_hist[t], dict(self.state[t]["idx"]))
            self.state[t]["sh"] = [r for r in (B.T @ H1)]
        self._Z0 = self._aggs = None  # drop training-graph caches before pickling

    def _score_parts(self, x, keys):
        xs = self._scale(x)
        z, js = [xs], []
        for t in self.types:
            st, k = self.state[t], (keys.get(t) if self.saw_type(t) else None)
            j = st["idx"].get(k) if k else None
            js.append(j)
            z.append((st["sx"][j] + xs) / (st["cnt"][j] + 1) if j is not None else xs)
        h1 = np.maximum(np.concatenate(z) @ self.W1 + self.b1, 0)
        z_out = h1 @ self.V[0] + self.c[0]
        for n, t in enumerate(self.types):
            st, j = self.state[t], js[n]
            z_out += ((st["sh"][j] + h1) / (st["cnt"][j] + 1) if j is not None else h1) @ self.V[n + 1]
        return float(z_out), h1, xs

    def _add(self, keys, xs, h1):
        for t in self.types:
            k = keys.get(t) if self.saw_type(t) else None
            if not k:
                continue
            st = self.state[t]
            j = st["idx"].get(k)
            if j is None:
                st["idx"][k] = j = len(st["cnt"])
                st["cnt"].append(0.0)
                st["sx"].append(np.zeros_like(xs))
                st["sh"].append(np.zeros_like(h1))
            st["cnt"][j] += 1
            st["sx"][j] = st["sx"][j] + xs
            st["sh"][j] = st["sh"][j] + h1

    def score_one(self, x: np.ndarray, keys: dict[str, str | None], update=True, as_logit=False) -> float:
        """Score one new transaction against the stored graph history, then (if update) add it.

        The neighbourhood means include the transaction itself, exactly as in training, where each
        entity's mean runs over all of its transactions including the one being scored. (Excluding it,
        as earlier versions did, skews live scores away from what the GNN learned: on the synthetic test
        file it cost 0.07 live PR-AUC.) Up to the staleness of earlier nodes' hidden states, this equals
        running the trained GNN on the graph of every transaction so far.

        as_logit=True returns the log-odds instead of the probability (used by the explainer, because a
        saturated probability hides how far the evidence is from the threshold).
        Entity links only: similarity relations have no streaming state (score those with embed_new,
        which is already causal when the adjacency only points to earlier transactions).
        """
        z_out, h1, xs = self._score_parts(x, keys)
        if update:
            self._add(keys, xs, h1)
        return z_out if as_logit else float(_sigmoid(z_out))

    def stream(self, X, entity_keys: dict[str, list], start: int, stop: int, progress=None):
        """Score rows start..stop one at a time against the history before each, adding each after scoring.
        Returns (log-odds, layer-1 hidden states) for those rows."""
        logits = np.empty(stop - start)
        H = np.empty((stop - start, self.hidden), dtype=self.dtype)
        cols = [entity_keys[t] for t in self.types]
        for i in range(start, stop):
            keys = {t: c[i] for t, c in zip(self.types, cols)}
            z, h1, xs = self._score_parts(X[i], keys)
            self._add(keys, xs, h1)
            logits[i - start], H[i - start] = z, h1
            if progress and (i - start) % 20_000 == 0 and i > start:
                progress(i - start, stop - start)
        return logits, H
