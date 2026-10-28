"""Transfer learning + reinforcement learning onto a new, mostly unlabelled card dataset.

    sources (labelled)                      target (labels scarce and late)
    IEEE-CIS, European Cardholder,    -->   a South African card file (credit_card_clean.csv)
    Brazilian Card Holder (optional)

1. Every dataset is mapped onto the shared schema of transfer.py (account, card, device, merchant,
   country, balance) and described by behaviour that means the same in every file (amount against
   the account's history, velocity, new device / merchant / country, hour of day), standardised within
   each file. The European data has no account or device, so it contributes only amount and
   time-of-day behaviour; its account-history features are set to "unknown" (0).
2. Pre-training: one GNN per source set, trained on source labels only, early stopping on source
   validation data. No target label is used.
3. Target, three ways, each scored causally (each test transaction sees only earlier ones):
      zero-shot      the source GNN applied unchanged
      target-only    a GNN trained from scratch on the revealed target labels
      fine-tuned     the source GNN's weights as the starting point, trained further on the same labels
   at several label rates (5%, 20%, 100% of the target's training and validation periods), over
   several random label masks.
4. RL: the branching DQN of rl_agent.py sets, per time window of the target stream, the alert
   threshold offset, the label-propagation weight and the pseudo-labelling confidence, learning from
   the delayed 20% of target labels. It runs on top of the zero-shot and the fine-tuned GNN.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import rl_agent as R
from .fsmote import FuzzySMOTE
from .gnn import TransactionGNN, build_incidence
from .rl_data import GraphData, _groups
from .rl_pipeline import _best_logit_threshold, _score_report, _init_counts, _trace_summary
from .transfer import (ENTITY_TYPES, MODEL_FEATURES, _canonical, _split, entity_keys, load_brazil, load_ieee,
                       pseudo_label_brazil, shared_features)

HISTORY_FEATURES = ["f_prior", "f_amt_ratio", "f_amt_z", "f_log_gap", "f_tx_24h", "f_bal_ratio", "f_first",
                    "f_new_device", "f_new_merchant", "f_new_country", "f_foreign"]


# ------------------------------------------------------------------ loaders onto the shared schema
def load_ulb_canonical(path, exclude: pd.DataFrame | None = None) -> pd.DataFrame:
    """exclude: rows (Time, Amount) to drop, e.g. European transactions that were copied into the target file,
    so the source never carries a target transaction's label."""
    u = pd.read_csv(path, usecols=["Time", "Amount", "Class"]).sort_values("Time", kind="stable").reset_index(drop=True)
    if exclude is not None and len(exclude):
        key = set(zip(exclude["Time"].astype(float), exclude["Amount"].astype(float).round(2)))
        drop = np.array([(t, a) in key for t, a in zip(u["Time"].astype(float), u["Amount"].astype(float).round(2))])
        u = u[~drop].reset_index(drop=True)
    c = _canonical(len(u))
    c.update({"tx_id": np.arange(len(u)).astype(str), "t": u["Time"].astype(float).to_numpy(),
              "amount": u["Amount"].astype(float).to_numpy(), "label": u["Class"].astype(int).to_numpy()})
    return pd.DataFrame(c)


def load_sa(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(path, low_memory=False)
    d = d[d["Class"].notna()].copy()
    d["Class"] = d["Class"].astype(int)
    d = d.sort_values(["Time", "transaction_id"], kind="stable").reset_index(drop=True)

    def col(name):
        return [None if pd.isna(v) else str(v) for v in d[name].tolist()]

    c = _canonical(len(d))
    c.update({"tx_id": d["transaction_id"].astype(str).to_numpy(), "t": d["Time"].astype(float).to_numpy(),
              "amount": d["Amount"].astype(float).to_numpy(), "account": col("account_id"), "card": col("card_token"),
              "device": col("device_id"), "merchant": col("merchant_id"), "country": col("tx_country"),
              "balance": d["account_balance"].astype(float).to_numpy(), "label": d["Class"].to_numpy()})
    return pd.DataFrame(c), d


def features(c: pd.DataFrame, no_history=False) -> np.ndarray:
    f = shared_features(c)
    if no_history:   # no account key: "first transaction" etc. would be true of every row, a domain giveaway
        for k in HISTORY_FEATURES:
            f[k] = 0.0
    return f[MODEL_FEATURES].to_numpy(np.float32)


SA_CATS = {"channel": None, "entry_mode": None, "merchant_category": None, "device_type": None, "card_product": None}


def sa_rich_features(raw: pd.DataFrame, X_shared: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Shared features + the South African file's own fields (channel, 3-D Secure, merchant category ...).
    Only for the target-only reference: no source dataset has these fields."""
    parts, names = [X_shared], list(MODEL_FEATURES)
    for colname in SA_CATS:
        dm = pd.get_dummies(raw[colname].fillna("none"), prefix=colname, dtype=np.float32)
        parts.append(dm.to_numpy())
        names += list(dm.columns)
    extra = np.c_[np.log1p(raw["account_tenure_months"].to_numpy(float)),
                  np.log1p(raw["seconds_since_prev_account_tx"].fillna(-1).clip(lower=0).to_numpy(float)),
                  raw["seconds_since_prev_account_tx"].isna().to_numpy(float)]
    parts.append(extra.astype(np.float32))
    names += ["log_tenure", "log_secs_prev", "first_tx"]
    return np.hstack(parts).astype(np.float32), names


# ------------------------------------------------------------------ source pre-training
def pretrain(sources: dict, epochs=300, seed=7, progress=print, shared_relations=True) -> tuple[TransactionGNN, dict]:
    """sources: {name: (canonical df, X, y, share)}. Each source carries `share` of the loss.

    shared_relations: link only through entity types every source has. A type one source lacks falls back
    to the transaction itself in that source's rows, so its weights would be trained mostly as a copy of the
    self weights and then misread real links in a new file (e.g. IEEE-CIS has no merchant, Brazil has one)."""
    common = set(ENTITY_TYPES)
    for c_, *_ in sources.values():
        common &= {t for t in ENTITY_TYPES if pd.Series(c_[t]).notna().any()}
    Xs, ys, keys, tr_all, va_all, wts, syn_plans, offs = [], [], {t: [] for t in ENTITY_TYPES}, [], [], [], [], 0
    stats = {}
    shares = {k: v[3] for k, v in sources.items()}
    tot_share = sum(shares.values())
    for name, (c, X, y, share) in sources.items():
        n = len(y)
        tr, va, _ = _split(n)
        k = entity_keys(c, name)
        for t in ENTITY_TYPES:
            keys[t] += k[t] if (t in common or not shared_relations) else [None] * n
        Xs.append(X), ys.append(y)
        tr_all.append(tr + offs), va_all.append(va + offs)
        wts.append((share / tot_share) / len(tr) * np.ones(len(tr)))
        stats[name] = {"rows": n, "frauds": int(y.sum()), "share": share / tot_share}
        offs += n
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    train_idx, val_idx = np.concatenate(tr_all), np.concatenate(va_all)
    w = np.concatenate(wts)
    w = w / w.mean()
    fs = FuzzySMOTE(ratio=1.0)
    plans = [fs.plan(X, y, tr) for tr in tr_all]
    from .transfer import _merge_plans
    plan, syn_dom, fs_stats = _merge_plans([(p, str(i)) for i, p in enumerate(plans)])
    # synthetic frauds inherit their domain's weight
    dom_w = {str(i): float(wts[i][0] / np.concatenate(wts).mean()) for i in range(len(wts))}
    syn_w = np.array([dom_w[d] for d in syn_dom]) if len(syn_dom) else None
    t0 = time.time()
    gnn = TransactionGNN(epochs=epochs, lr=0.015, seed=seed).fit(
        X, keys, y, train_idx, val_idx, synth=plan if len(plan) else None, sample_weight=w, syn_weight=syn_w,
        progress=lambda ep, ap: progress(f"  epoch {ep}: source validation PR-AUC {ap:.3f}") if ep % 50 == 0 else None)
    p = gnn.predict_graph()
    z = np.log(np.clip(p, 1e-9, 1)) - np.log(np.clip(1 - p, 1e-9, 1))
    off = 0
    for name, (c, X_, y_, _) in sources.items():
        n = len(y_)
        _, va, te = _split(n)
        stats[name]["val_pr_auc"] = float(_score_report(y_[va], z[off + va], 0)["pr_auc"]) if y_[va].sum() else None
        stats[name]["test"] = _score_report(y_[te], z[off + te], _best_logit_threshold(y_[va], z[off + va]))
        off += n
    thr = _best_logit_threshold(y[val_idx], z[val_idx])
    gnn._Z0 = gnn._aggs = None
    return gnn, {"sources": stats, "threshold_logit": thr, "train_seconds": round(time.time() - t0, 1),
                 "relations": list(gnn.seen_types),
                 "best_val_pr_auc": float(gnn.best_val_ap)}


# ------------------------------------------------------------------ target
def target_graph(c: pd.DataFrame, X: np.ndarray, name="South African cards") -> GraphData:
    keys = entity_keys(c, "sa")
    read, write = _groups(keys, {}, len(c))
    return GraphData(name=name, kind="sa", X=X, y=c["label"].to_numpy(int), t=c["t"].to_numpy(float),
                     amount=c["amount"].to_numpy(float), feature_names=list(MODEL_FEATURES), entity_keys=keys,
                     read=read, write=write, tx_id=c["tx_id"].to_numpy())


def visible_mask(n, rate, seed):
    return np.random.default_rng(seed).random(n) < rate


def causal(gnn: TransactionGNN, g: GraphData, start: int):
    gnn.build_inference_state(g.X[:start], {t: v[:start] for t, v in g.entity_keys.items()})
    return gnn.stream(g.X, g.entity_keys, start, g.n)


FINETUNE_LRS = (0.005, 0.015)


def train_target(g: GraphData, vis, init=None, epochs=300, seed=7, lr=None, fsmote_ratio=1.0):
    """A GNN on the revealed target labels: from scratch (lr 0.015), or fine-tuned from `init`, in which case
    the learning rate is chosen on the revealed validation labels among FINETUNE_LRS unless given."""
    tr, va, _ = g.split()
    tr_l, va_l = tr[vis[tr]], va[vis[va]]
    if g.y[tr_l].sum() < 2 or g.y[va_l].sum() < 1:
        return None
    plan = FuzzySMOTE(ratio=fsmote_ratio).plan(g.X, g.y, tr_l) if fsmote_ratio > 0 else []
    lrs = [lr] if lr else (list(FINETUNE_LRS) if init is not None else [0.015])
    best = None
    for rate in lrs:
        gnn = TransactionGNN(epochs=epochs, lr=rate, seed=seed)
        gnn.fit(g.X, g.entity_keys, g.y, tr_l, va_l, synth=plan if len(plan) else None, init=init)
        gnn._Z0 = gnn._aggs = None
        if best is None or gnn.best_val_ap > best.best_val_ap:
            best = gnn
    return best


def report(g, z, vis, thr=None):
    """Test metrics. Threshold: the one given, else the best F1 on the revealed validation labels."""
    tr, va, te = g.split()
    if thr is None:
        va_l = va[vis[va]]
        thr = _best_logit_threshold(g.y[va_l], z[va_l]) if g.y[va_l].sum() else float(np.quantile(z[va], 0.99))
    return {**_score_report(g.y[te], z[te], thr), "threshold_logit": float(thr)}


def full_scores(gnn, g):
    tr, va, te = g.split()
    start = int(va[0])
    z, H = causal(gnn, g, start)
    zf = np.zeros(g.n)
    Hf = np.zeros((g.n, gnn.hidden), np.float32)
    zf[start:], Hf[start:] = z, H
    return zf, Hf


# ------------------------------------------------------------------ RL on the target stream
def rl_on_target(g: GraphData, z, H, theta0, vis, window_hours=2.0, delay=3, episodes=300, agent_seeds=3,
                 label_rate=0.2, seed=42, progress=print):
    tr, va, te = g.split()
    secs = window_hours * 3600
    w_val, w_test = R.windows_for(g.t, va, secs), R.windows_for(g.t, te, secs)
    prior = float(g.y[tr[vis[tr]]].mean()) if vis[tr].sum() else 0.01
    f_tr, l_tr = _init_counts(g, tr, vis)
    f_tv, l_tv = _init_counts(g, np.concatenate([tr, va]), vis)
    proj = R.make_projector(H[va])
    common = dict(y=g.y, read=g.read, write=g.write, prior=prior, theta0=theta0, label_rate=label_rate,
                  delay=delay, fp_cost=1.0, projector=proj)
    val_env = R.FraudStreamEnv(z, H, windows=w_val, init_fraud=f_tr, init_legit=l_tr, **common)
    test_env = R.FraudStreamEnv(z, H, windows=w_test, init_fraud=f_tv, init_legit=l_tv, **common)
    out = {"windows_val": len(w_val), "windows_test": len(w_test), "prior": prior, "theta0": theta0}
    static = R.best_static_action(val_env, seed=seed)
    out["static_action"] = list(R.ACTIONS[static["action"]])
    out["static_val_reward"] = static["val_reward"]
    pol = {"fixed": R.evaluate_run(R.run_policy(test_env, R.ACTIONS.index((0.0, 0.0, None)), vis), te),
           "static": R.evaluate_run(R.run_policy(test_env, static["action"], vis), te)}
    agents, online, curves, traces = [], [], [], {}
    for k in range(agent_seeds):
        ag, curve = R.train_agent(val_env, episodes=episodes, seed=seed + 100 * k)
        env_a = R.run_policy(test_env, "agent", vis, agent=ag)
        agents.append(R.evaluate_run(env_a, te))
        import copy
        env_o = R.run_policy(test_env, "agent", vis, agent=copy.deepcopy(ag), online=True, eps=0.02)
        online.append(R.evaluate_run(env_o, te))
        curves.append([c["reward"] for c in curve])
        if k == 0:
            traces["agent"] = _trace_summary(env_a)
            traces["online"] = _trace_summary(env_o)
        progress(f"    agent {k + 1}/{agent_seeds}: test F1 {agents[-1]['f1']:.3f}, online {online[-1]['f1']:.3f}")
    traces["fixed"] = _trace_summary(R.run_policy(test_env, R.ACTIONS.index((0.0, 0.0, None)), vis))
    traces["static"] = _trace_summary(R.run_policy(test_env, static["action"], vis))

    def agg(runs):
        keys = ["f1", "precision", "recall", "pr_auc", "roc_auc", "tp", "fp", "reward", "reward_all"]
        keys = [k for k in keys if k in runs[0]]
        return {**{k: float(np.mean([r[k] for r in runs])) for k in keys},
                "sd": {k: float(np.std([r[k] for r in runs])) for k in keys}}
    pol["agent"], pol["online"] = agg(agents), agg(online)
    usage = np.bincount(traces["agent"]["actions"], minlength=R.N_ACTIONS)
    out["action_usage"] = {
        "theta_offset": {str(d): int(sum(usage[i] for i, a in enumerate(R.ACTIONS) if a[0] == d)) for d in R.THETA_OFFSETS},
        "lp_weight": {str(w): int(sum(usage[i] for i, a in enumerate(R.ACTIONS) if a[1] == w)) for w in R.LP_WEIGHTS},
        "pseudo_conf": {str(c): int(sum(usage[i] for i, a in enumerate(R.ACTIONS) if a[2] == c)) for c in R.PL_CONF}}
    out["policies"] = pol
    out["training_curve"] = np.mean(curves, axis=0).round(4).tolist()
    out["traces"] = {k: {kk: v[kk] for kk in ("theta", "cum_tp", "cum_fp", "cum_reward_all", "w", "pl")} for k, v in traces.items()}
    return out
