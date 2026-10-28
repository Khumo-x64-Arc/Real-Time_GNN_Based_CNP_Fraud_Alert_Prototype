"""Semi-supervised training of the GNN: few labels, graph propagation and self-training.

Real fraud labels are scarce and slow: a chargeback or an investigation confirms a small share of
transactions, weeks later. This module trains the GNN as if only a fraction `label_rate` of
transactions (training and validation alike) had a label, and uses the rest without labels:

1. The graph itself is semi-supervised (Kipf & Welling, 2017, cited by FraudGNN-RL): every node,
   labelled or not, is in the graph, so unlabelled transactions shape their neighbours' embeddings.
2. Self-training. After each round, unlabelled training transactions the GNN is most sure about
   become pseudo-labels for the next round: frauds with probability >= `fraud_conf`, at most half
   the number of frauds the labelled prior predicts among them, and a sample of the clearest
   legitimate ones. Pseudo-labels count half as much as real labels in the loss.
3. Model selection uses only the labelled validation transactions. The round with the best
   labelled-validation PR-AUC is kept, so the procedure never peeks at hidden labels.

For the report, the same GNN is also trained on 100% of labels (an upper bound) and pseudo-label
precision is measured against the hidden labels. Neither is used for any decision.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import average_precision_score

from .fsmote import FSMOTEPlan, FuzzySMOTE
from .gnn import TransactionGNN


class SemiSupError(Exception):
    pass


def visible_mask(n: int, rate: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(size=n) < rate


def _fit(g, y_eff, train_idx, val_idx, weights, fsmote_ratio, epochs, progress=None):
    if fsmote_ratio > 0 and y_eff[train_idx].sum() >= 2:
        plan = FuzzySMOTE(ratio=fsmote_ratio).plan(g.X, y_eff, train_idx)
    else:
        plan = FSMOTEPlan.empty({"method": "off", "synthetic": 0})
    gnn = TransactionGNN(epochs=epochs)
    gnn.fit(g.X, g.entity_keys, y_eff, train_idx, val_idx, synth=plan, progress=progress,
            sample_weight=weights, relations=g.relations)
    return gnn, plan


def train_semi_supervised(g, label_rate=0.2, rounds=2, seed=42, epochs=300, fsmote_ratio=1.0,
                          fraud_conf=0.9, legit_conf=0.02, pseudo_weight=0.5, full_reference=True,
                          progress=lambda m, p: None, pct=(15, 55)):
    tr, va, te = g.split()
    vis = visible_mask(g.n, label_rate, seed)
    lab_tr, lab_va = tr[vis[tr]], va[vis[va]]
    unl_tr = tr[~vis[tr]]
    if g.y[lab_tr].sum() < 3 or g.y[lab_va].sum() < 1:
        raise SemiSupError(f"With {label_rate:.0%} of labels visible there are only {int(g.y[lab_tr].sum())} labelled "
                           f"training frauds and {int(g.y[lab_va].sum())} labelled validation frauds. "
                           "Raise the label rate.")
    prior = float(g.y[lab_tr].mean())
    rng = np.random.default_rng(seed)
    lo, hi = pct
    steps = rounds + 1 + (1 if full_reference else 0)
    span = (hi - lo) / steps
    y_eff = g.y.copy()
    train_idx, weights = lab_tr, np.ones(len(lab_tr))
    history, models = [], []
    for r in range(rounds + 1):
        t0 = time.time()
        base = lo + span * r
        gnn, plan = _fit(g, y_eff, train_idx, lab_va, weights, fsmote_ratio, epochs,
                         progress=lambda ep, ap: progress(
                             f"Semi-supervised GNN, round {r + 1} of {rounds + 1}: epoch {ep}, labelled-validation "
                             f"PR-AUC {ap:.3f}", int(base + min(span * 0.9, ep / epochs * span))))
        logit, _ = gnn.embed_graph()
        gnn._Z0 = gnn._aggs = None   # the training-graph cache is big (n x features x link types); scoring rebuilds what it needs
        p = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
        val_ap = float(average_precision_score(g.y[lab_va], p[lab_va]))
        pseudo_now = train_idx[len(lab_tr):]
        entry = {"round": r, "train_labels": int(len(lab_tr)), "train_label_frauds": int(g.y[lab_tr].sum()),
                 "pseudo_frauds": int((y_eff[pseudo_now] == 1).sum()), "pseudo_legit": int((y_eff[pseudo_now] == 0).sum()),
                 "labelled_val_pr_auc": val_ap, "seconds": round(time.time() - t0, 1),
                 "synthetic_frauds": int(plan.stats.get("synthetic", 0))}
        if r:
            pseudo = train_idx[len(lab_tr):]
            pf = pseudo[y_eff[pseudo] == 1]
            entry["pseudo_fraud_precision"] = float(g.y[pf].mean()) if len(pf) else None   # report only
            entry["pseudo_legit_error"] = float(g.y[pseudo[y_eff[pseudo] == 0]].mean()) if len(pseudo) > len(pf) else None
        history.append(entry)
        models.append(gnn)
        if r == rounds:
            break
        # next round's pseudo-labels, from this round's predictions on unlabelled training transactions
        pu = p[unl_tr]
        budget = int(prior * len(unl_tr) * 0.5)
        cand = unl_tr[pu >= fraud_conf]
        pf = cand[np.argsort(-p[cand])][:budget]
        legit_pool = unl_tr[pu <= legit_conf]
        pl = rng.choice(legit_pool, size=min(len(legit_pool), 3 * len(lab_tr)), replace=False) if len(legit_pool) else legit_pool
        y_eff = g.y.copy()
        y_eff[pf], y_eff[pl] = 1, 0
        train_idx = np.concatenate([lab_tr, pf, pl])
        weights = np.concatenate([np.ones(len(lab_tr)), np.full(len(pf) + len(pl), pseudo_weight)])
    best = int(np.argmax([h["labelled_val_pr_auc"] for h in history]))
    reference = None
    if full_reference:
        base = lo + span * (rounds + 1)
        reference, _ = _fit(g, g.y, tr, va, None, fsmote_ratio, epochs,
                            progress=lambda ep, ap: progress(
                                f"Reference GNN with every label: epoch {ep}, validation PR-AUC {ap:.3f}",
                                int(base + min(span * 0.9, ep / epochs * span))))
        reference._Z0 = reference._aggs = None
    return {"visible": vis, "prior": prior, "history": history, "best_round": best,
            "supervised": models[0], "semi": models[best], "reference": reference}


def causal_scores(gnn: TransactionGNN, g, start: int, progress=None):
    """(log-odds, hidden states) for rows start..n, each scored with only earlier transactions as neighbours.

    Similarity graphs are causal by construction, so a full-graph pass is already a streaming pass. Entity
    graphs are streamed: the history before `start` is loaded, then each row is scored and added in turn.
    """
    if g.entity_keys:
        gnn.build_inference_state(g.X[:start], {t: v[:start] for t, v in g.entity_keys.items()})
        if g.relations:
            # entity streaming can't carry similarity links; score them causally in one pass and add their terms
            raise SemiSupError("Streaming entity + similarity graphs together isn't supported; use one or the other.")
        return gnn.stream(g.X, g.entity_keys, start, g.n, progress=progress)
    logit, H = gnn.embed_new(g.X, {}, g.relations)
    return logit[start:], H[start:]


def features_only_scores(gnn: TransactionGNN, g, start: int):
    """The same trained GNN with every link removed: each transaction read on its own (the "w/o GNN" ablation)."""
    logit, H = gnn.embed_new(g.X, {t: [None] * g.n for t in g.entity_keys}, {})
    return logit[start:], H[start:]
