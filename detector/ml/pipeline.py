"""Train the graph neural network on one dataset and test it.

This is a GNN-only test bench: no other model is trained, and every number
on the results pages describes the GNN.

Split (time-ordered, so the GNN is tested on the future):
    first 60%  -> train
    next 15%   -> validation (early stopping + picking the alert threshold)
    last 25%   -> test (every number on the results page)
If either later slice has fewer than 5 frauds, a stratified random split with
the same proportions is used instead and the results page says so.

The GNN is tested in two ways, because they answer different questions:

    "GNN"       full graph (transductive). The whole file is one graph and every
                transaction sees every other transaction sharing its account,
                card, device, IP or merchant, including later ones (their
                features, never their labels). This is how well the model can
                separate fraud when the graph is complete.
    "GNN live"  streaming (inductive). Transactions arrive one at a time in time
                order and each is scored against only the history before it,
                then joins the graph. This is exactly what the live monitor and
                the scoring API do, so its threshold is the one they alert on.
                Only measured on a time-ordered split.

Two further tests probe what the GNN has learned:

    graph ablation    the test set is rescored with one entity type's links
                      removed (and once with no links at all), using the same
                      trained weights. The drop in PR-AUC is how much the GNN
                      relies on that part of the graph.
    threshold sweep   precision, recall, F1 and alert volume at a range of
                      thresholds, for choosing an operating point.

Every row of the file is used. Class imbalance is handled by FSMOTE
(fsmote.py), which adds synthetic fraud nodes to the training slice only, in
GraphSMOTE style. Set FSMOTE_RATIO (default 1.0 = balanced, 0 = off, which
falls back to class weights only) to change it.
"""
from __future__ import annotations

import copy
import os
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split

from .behavior import SIGNALS, BehaviorState, parse_times
from .features import FeatureBuilder
from .fsmote import FSMOTEPlan, FuzzySMOTE
from .gnn import TransactionGNN
from .schema import ENTITY_ROLES, describe_problems, detect_schema
from .txgraph import build_graph, save_graph

PIPELINE_VERSION = "gnn-only-1"   # stored in results and the bundle; older bundles need a retrain


def fsmote_ratio() -> float:
    try:
        return max(0.0, float(os.environ.get("FSMOTE_RATIO", "1.0")))
    except ValueError:
        return 1.0


MODEL_NAMES = ["GNN", "GNN live"]   # full-graph and streaming evaluation of the same trained GNN
LIVE = "GNN live"
HIST_BINS = 20
SWEEP = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SIGNAL_LABELS = {
    "foreign_country": "Foreign country", "new_device": "New device", "new_ip": "New IP address",
    "amount_spike": "Amount spike", "velocity": "3+ transactions in an hour",
    "rapid_succession": "Under a minute since last transaction", "shared_device": "Device shared by 3+ accounts",
    "shared_ip": "IP shared by 3+ accounts", "away_from_home": "Away from home city",
    "night_time": "Between midnight and 5am", "drains_balance": "Half or more of the balance",
}
ENTITY_LABELS = {"account": "Account", "card": "Card", "device": "Device", "ip": "IP address",
                 "merchant": "Merchant", "terminal": "Terminal", "amount_bucket": "Amount bucket"}


class DatasetError(Exception):
    """Raised with a message that is shown to the user as-is."""


def _best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5


def _curve(x, y, n=120):
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).astype(int))
    return [[round(float(x[i]), 4), round(float(y[i]), 4)] for i in idx]


def _chart_data(y, p, thr):
    """Extra series for the results-page charts (kept out of ModelResult so no migration is needed)."""
    prec, rec, _ = precision_recall_curve(y, p)
    edges = np.linspace(0, 1, HIST_BINS + 1)
    return {
        "pr_curve": _curve(rec[::-1], prec[::-1]),
        "hist_fraud": np.histogram(np.clip(p[y == 1], 0, 1), edges)[0].tolist(),
        "hist_legit": np.histogram(np.clip(p[y == 0], 0, 1), edges)[0].tolist(),
        "threshold": float(thr),
    }


def _metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(thr),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "roc_curve": _curve(fpr, tpr),
    }


def _at(y, p, thr):
    """Operating-point numbers at one threshold (for the sweep and the ablation table)."""
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": float(thr), "flagged": int(pred.sum()), "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "alert_rate": float(pred.mean()) if len(pred) else 0.0}


def threshold_sweep(y, p, chosen):
    rows = [{**_at(y, p, t), "chosen": False} for t in SWEEP]
    rows.append({**_at(y, p, chosen), "chosen": True})
    rows.sort(key=lambda r: (r["threshold"], not r["chosen"]))
    return rows


def entity_keys_for(df: pd.DataFrame, schema: dict) -> dict[str, list]:
    keys = {}
    seen_cols = set()
    for role in ENTITY_ROLES:
        col = schema.get(role)
        if not col or col in seen_cols:
            continue
        seen_cols.add(col)
        vals = df[col].astype(object).where(df[col].notna(), None).tolist()
        keys[role] = [None if v is None or str(v).strip() in {"", "nan"} else f"{role}:{v}" for v in vals]
    return keys


def graph_keys_for(df: pd.DataFrame, schema: dict) -> dict[str, list]:
    """Entity keys the GNN links on. Without entity columns, transactions of the same amount bucket are
    linked so the GNN still has a graph; with no amount either, it has none and reads each row alone."""
    keys = entity_keys_for(df, schema)
    if keys:
        return keys
    if schema.get("amount"):
        amounts = pd.to_numeric(df[schema["amount"]], errors="coerce").fillna(0)
        return {"amount_bucket": [f"amt:{round(float(a), -1)}" for a in amounts]}
    return {"none": [None] * len(df)}


def stream_scores(gnn: TransactionGNN, X, keys: dict[str, list], start: int, stop: int,
                  progress=None) -> np.ndarray:
    """Score rows start..stop one at a time in order, each against only the history before it, and add each
    to the graph after scoring it. Exactly what LiveEngine does, minus the behaviour bookkeeping."""
    out = np.empty(stop - start)
    cols = [keys[t] for t in gnn.types]
    for i in range(start, stop):
        out[i - start] = gnn.score_one(X[i], {t: c[i] for t, c in zip(gnn.types, cols)}, update=True)
        if progress and (i - start) % 10_000 == 0 and i > start:
            progress(i - start, stop - start)
    return out


def graph_ablation(gnn: TransactionGNN, X, keys, y, test_idx, p_full, thr) -> list[dict]:
    """Rescore the full graph with one entity type's links removed, then with no links at all.

    The trained weights don't change. A transaction that loses its links behaves like one whose
    entity has no other transactions, so the drop shows how much the GNN leans on that relation.
    """
    yte = y[test_idx]
    base_ap, base_auc = float(average_precision_score(yte, p_full[test_idx])), float(roc_auc_score(yte, p_full[test_idx]))
    rows = [{"type": "", "label": "Nothing removed (full graph)", "pr_auc": base_ap, "roc_auc": base_auc,
             "d_pr_auc": 0.0, "d_roc_auc": 0.0, **_at(yte, p_full[test_idx], thr)}]
    n = X.shape[0]
    variants = [(t, [t]) for t in gnn.types] + ([("*", list(gnn.types))] if len(gnn.types) > 1 else [])
    for name, drop in variants:
        p = gnn.predict_new(X, {t: ([None] * n if t in drop else keys[t]) for t in gnn.types})[test_idx]
        ap, auc = float(average_precision_score(yte, p)), float(roc_auc_score(yte, p))
        label = "Every link removed (features only)" if name == "*" else f"{ENTITY_LABELS.get(name, name.title())} links removed"
        rows.append({"type": name, "label": label, "pr_auc": ap, "roc_auc": auc,
                     "d_pr_auc": ap - base_ap, "d_roc_auc": auc - base_auc, **_at(yte, p, thr)})
    return rows


def prepare(csv_path: str | Path, progress=lambda msg, pct: None) -> dict:
    """Read, profile and split a labelled file, and build its node features and graph keys. Shared by the full
    training run and by the with/without-oversampling comparison, so both see exactly the same data."""
    progress("Reading the file", 3)
    df = pd.read_csv(csv_path, low_memory=False)
    schema = detect_schema(df)
    problems = describe_problems(df, schema)
    if problems:
        raise DatasetError(" ".join(problems))

    label = schema["label"]
    df = df[pd.to_numeric(df[label], errors="coerce").isin([0, 1])].copy()
    secs, hours, is_dt = parse_times(df, schema)
    order = np.argsort(secs.to_numpy(), kind="stable")
    df, secs = df.iloc[order].reset_index(drop=True), secs.iloc[order].reset_index(drop=True)
    hours = hours.iloc[order].reset_index(drop=True) if hours is not None else None
    origin = pd.to_datetime(df[schema["time"]], errors="coerce").min() if is_dt else None
    y = df[label].astype(int).to_numpy()
    n = len(df)

    # ---- behavioural features, one transaction at a time in time order
    progress("Profiling account behaviour", 8)
    country_col = schema.get("country")
    default_country = df[country_col].mode().iloc[0] if country_col and df[country_col].notna().any() else None
    state = BehaviorState(schema, default_country)
    state.datetime_time = is_dt
    cut = int(n * 0.75)
    records = df.to_dict("records")
    feats, sigs, snapshot = [], [], None
    for i, row in enumerate(records):
        if i == cut:
            snapshot = copy.deepcopy(state)  # history as of the start of the test period
        if i and i % 20_000 == 0:
            progress(f"Profiling account behaviour ({i:,} of {n:,} rows)", 8 + int(14 * i / n))
        f, s = state.process(row, float(secs[i]), int(hours[i]) if hours is not None else None)
        feats.append(f)
        sigs.append([x["code"] for x in s])
    if snapshot is None:
        snapshot = copy.deepcopy(state)
    behavior = pd.DataFrame(feats)

    # ---- split
    tr, va = int(n * 0.60), cut
    idx = np.arange(n)
    train_idx, val_idx, test_idx = idx[:tr], idx[tr:va], idx[va:]
    split_kind = "time"
    if y[val_idx].sum() < 5 or y[test_idx].sum() < 5:
        split_kind = "stratified"
        train_idx, rest = train_test_split(idx, train_size=0.60, stratify=y, random_state=42)
        val_idx, test_idx = train_test_split(rest, train_size=0.375, stratify=y[rest], random_state=42)

    progress("Building node features", 23)
    fb = FeatureBuilder().fit(df.iloc[train_idx], schema)
    X = fb.transform(df, behavior)
    ytr, yva, yte = y[train_idx], y[val_idx], y[test_idx]

    progress("Building the transaction graph", 30)
    keys = graph_keys_for(df, schema)
    return {"df": df, "schema": schema, "origin": origin, "y": y, "n": n, "sigs": sigs, "snapshot": snapshot,
            "cut": cut, "tr": tr, "va": va, "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
            "split_kind": split_kind, "fb": fb, "X": X, "ytr": ytr, "yva": yva, "yte": yte, "keys": keys}


def train_and_evaluate(csv_path: str | Path, out_dir: str | Path, progress=lambda msg, pct: None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    P = prepare(csv_path, progress)
    (df, schema, origin, y, n, sigs, snapshot, cut, tr, va, train_idx, val_idx, test_idx, split_kind, fb, X, ytr, yva,
     yte, keys) = (P[k] for k in ("df", "schema", "origin", "y", "n", "sigs", "snapshot", "cut", "tr", "va",
                                  "train_idx", "val_idx", "test_idx", "split_kind", "fb", "X", "ytr", "yva", "yte",
                                  "keys"))

    # ---- FSMOTE: synthetic fraud nodes for the training slice (GraphSMOTE style, inside the GNN)
    ratio = fsmote_ratio()
    t0 = time.time()
    if ratio > 0:
        progress(f"Planning FSMOTE synthetic frauds ({int(ytr.sum()):,} real frauds in training)", 31)
        plan = FuzzySMOTE(ratio=ratio).plan(X, y, train_idx)
    else:
        plan = FSMOTEPlan.empty({"method": "off", "minority_before": int(ytr.sum()),
                                 "majority": int((ytr == 0).sum()), "synthetic": 0})
    plan.stats.update({"seconds": round(time.time() - t0, 1), "train_rows": int(len(train_idx)),
                       "train_rows_after": int(len(train_idx) + len(plan))})

    # ---- train
    t0 = time.time()
    gnn = TransactionGNN().fit(
        X, keys, y, train_idx, val_idx, synth=plan,
        progress=lambda ep, ap: progress(f"Training the GNN (epoch {ep}, validation PR-AUC {ap:.3f})",
                                         min(62, 32 + ep // 10)))
    train_seconds = time.time() - t0
    plan.stats["residual_weight"] = gnn.pos_weight

    # ---- test 1: full graph (transductive)
    progress("Testing the GNN on the full graph", 64)
    p_full = gnn.predict_graph()
    probs_val, probs_test = {"GNN": p_full[val_idx]}, {"GNN": p_full[test_idx]}
    thresholds = {"GNN": _best_threshold(yva, probs_val["GNN"])}

    # ---- test 2: which parts of the graph the GNN relies on
    progress("Graph ablation: rescoring the test set without each entity type", 68)
    ablation = graph_ablation(gnn, X, keys, y, test_idx, p_full, thresholds["GNN"])

    # ---- the whole transaction graph, laid out for the results page
    progress("Laying out the transaction graph", 74)
    split_code = np.zeros(n, dtype=np.uint8)
    split_code[val_idx], split_code[test_idx] = 1, 2
    graph_meta, graph_blob = build_graph(keys, y, split_code, p_full, thresholds["GNN"])
    tx_info = pd.DataFrame({
        "id": df[schema["tx_id"]].astype(str).to_numpy() if schema.get("tx_id") else np.arange(n).astype(str),
        "amount": pd.to_numeric(df[schema["amount"]], errors="coerce").to_numpy() if schema.get("amount") else np.nan,
        "time": df[schema["time"]].astype(str).to_numpy() if schema.get("time") else "",
    })
    save_graph(out_dir, graph_meta, graph_blob, tx_info)
    graph_summary = {k: v for k, v in graph_meta.items() if k not in ("names", "arrays", "bounds")}
    graph_summary["rings"] = [{k: v for k, v in r.items() if k not in ("tx", "ents")} for r in graph_meta["rings"][:10]]

    # ---- test 3: streaming (inductive), exactly as the live monitor scores
    live_seconds = None
    if split_kind == "time":
        t0 = time.time()
        progress("Streaming test: scoring validation transactions one at a time", 80)
        gnn.build_inference_state(X[:tr], {t: v[:tr] for t, v in keys.items()})
        p_live_val = stream_scores(gnn, X, keys, tr, va, progress=lambda d, tot: progress(
            f"Streaming test: validation {d:,} of {tot:,}", 80))
        live_state = copy.deepcopy(gnn.state)   # graph history as of the start of the test period
        progress("Streaming test: scoring test transactions one at a time", 85)
        p_live_test = stream_scores(gnn, X, keys, va, n, progress=lambda d, tot: progress(
            f"Streaming test: test {d:,} of {tot:,}", 85))
        gnn.state = live_state
        probs_val[LIVE], probs_test[LIVE] = p_live_val, p_live_test
        thresholds[LIVE] = _best_threshold(yva, p_live_val)
        live_seconds = time.time() - t0
    else:
        # random split: "the past" isn't defined, so only the full-graph test applies
        gnn.build_inference_state(X[:cut], {t: v[:cut] for t, v in keys.items()})
    live_threshold = thresholds.get(LIVE, thresholds["GNN"])

    # ---- evaluate
    progress("Scoring the test set", 90)
    names = [m for m in MODEL_NAMES if m in probs_test]
    seconds = {"GNN": round(train_seconds, 1), LIVE: round(live_seconds or 0, 1)}
    results = {m: {**_metrics(yte, probs_test[m], thresholds[m]), "train_seconds": seconds[m]} for m in names}
    sweep_model = LIVE if LIVE in probs_test else "GNN"
    charts = {
        "models": {m: _chart_data(yte, probs_test[m], thresholds[m]) for m in names},
        "training": {"GNN": {"unit": "epoch", "points": [[h["epoch"], round(h["val_ap"], 4)] for h in gnn.history]}},
        "hist_bins": HIST_BINS,
    }

    base_rate = float(yte.mean())
    sig_stats = []
    for code, (w, _) in SIGNALS.items():
        hit = np.array([code in sigs[i] for i in test_idx])
        if hit.sum():
            rate = float(yte[hit].mean())
            sig_stats.append({"code": code, "label": SIGNAL_LABELS.get(code, code), "count": int(hit.sum()),
                              "fraud_rate": rate, "lift": rate / base_rate if base_rate else 0.0,
                              "frauds_caught": int(yte[hit].sum())})
    sig_stats.sort(key=lambda s: -s["lift"])

    # ---- the same GNN with oversampling switched the other way, for the results page
    oversampling = None
    if os.environ.get("OVERSAMPLING_COMPARISON", "1") != "0":
        oversampling = oversampling_block({"fsmote": plan.stats, "models": results,
                                           "gnn": {"best_val_ap": float(gnn.best_val_ap),
                                                   "train_seconds": round(train_seconds, 1),
                                                   "stream_seconds": round(live_seconds, 1) if live_seconds else None,
                                                   "epochs_run": gnn.history[-1]["epoch"] if gnn.history else 0}},
                                          P, progress, 90, 96)

    # ---- bundle for the live engine
    progress("Saving the GNN for live monitoring", 97)
    replay = df.iloc[cut:]  # the whole test period
    # "typical transaction" used as the reference point by the explainer:
    # training medians for numbers and behaviour, the most common level for categories
    baseline = np.median(X[train_idx], axis=0)
    for c, levels in fb.categorical.items():
        for v in levels:
            baseline[fb.columns.index(f"{c}={v}")] = 0.0
        baseline[fb.columns.index(f"{c}={fb.cat_modes[c]}")] = 1.0
    legit = train_idx[ytr == 0]
    bundle = {
        "version": PIPELINE_VERSION, "schema": schema, "features": fb, "gnn": gnn,
        "gnn_types": list(keys), "threshold": live_threshold, "thresholds": thresholds,
        "behavior": snapshot, "origin": origin, "baseline": baseline,
        # background sample of legitimate training transactions for averaged occlusion
        "background": X[np.random.default_rng(0).choice(legit, size=min(12, len(legit)), replace=False)],
    }
    joblib.dump(bundle, out_dir / "bundle.joblib", compress=3)
    replay.to_pickle(out_dir / "replay.pkl")

    return {
        "version": PIPELINE_VERSION,
        "schema": {k: v for k, v in schema.items() if v and k != "label_problem"},
        "rows": n, "rows_used": n, "frauds": int(y.sum()), "fraud_rate": float(y.mean()),
        "split": split_kind, "sampled_note": None,
        "sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "test_frauds": int(yte.sum()), "n_features": len(fb.columns),
        "graph_entities": list(keys), "gnn_history": gnn.history,
        "gnn": {"epochs_run": gnn.history[-1]["epoch"] if gnn.history else 0,
                "best_epoch": max(gnn.history, key=lambda h: h["val_ap"])["epoch"] if gnn.history else 0,
                "best_val_ap": float(gnn.best_val_ap), "hidden": gnn.hidden, "lr": gnn.lr,
                "pos_weight": gnn.pos_weight, "train_seconds": round(train_seconds, 1),
                "stream_seconds": round(live_seconds, 1) if live_seconds is not None else None,
                "stream_rows": int(n - tr) if live_seconds is not None else 0},
        "live_threshold": live_threshold, "live_model": LIVE if LIVE in thresholds else "GNN",
        "ablation": ablation, "sweep": {"model": sweep_model,
                                        "rows": threshold_sweep(yte, probs_test[sweep_model], thresholds[sweep_model])},
        "charts": charts, "fsmote": plan.stats, "graph": graph_summary,
        "models": results, "signals": sig_stats, "replay_rows": len(replay), "oversampling": oversampling,
        "signal_counts": dict(Counter(c for i in test_idx for c in sigs[i])),
    }


# ------------------------------------------------------------------ with and without oversampling
def _plan_for(P: dict, ratio: float):
    ytr = P["ytr"]
    if ratio > 0:
        plan = FuzzySMOTE(ratio=ratio).plan(P["X"], P["y"], P["train_idx"])
    else:
        plan = FSMOTEPlan.empty({"method": "off", "minority_before": int(ytr.sum()),
                                 "majority": int((ytr == 0).sum()), "synthetic": 0})
    return plan


def train_variant(P: dict, ratio: float, progress=lambda msg, pct: None, lo=0, hi=100) -> dict:
    """Train the GNN on prepared data with FSMOTE at `ratio` (0 = off) and test it on the full graph and, for a
    time-ordered split, live. Thresholds are chosen on validation data exactly as in the main run."""
    X, keys, y = P["X"], P["keys"], P["y"]
    train_idx, val_idx, test_idx = P["train_idx"], P["val_idx"], P["test_idx"]
    yva, yte = P["yva"], P["yte"]
    span = hi - lo
    name = "with FSMOTE" if ratio > 0 else "without oversampling"
    progress(f"Comparison: training the GNN {name}", lo)
    plan = _plan_for(P, ratio)
    t0 = time.time()
    gnn = TransactionGNN().fit(X, keys, y, train_idx, val_idx, synth=plan,
                               progress=lambda ep, ap: progress(f"Comparison: training the GNN {name} (epoch {ep})",
                                                                lo + min(int(span * .5), ep // 10)))
    train_seconds = time.time() - t0
    p_full = gnn.predict_graph()
    thr = _best_threshold(yva, p_full[val_idx])
    models = {"GNN": {**_metrics(yte, p_full[test_idx], thr), "train_seconds": round(train_seconds, 1)}}
    stream_seconds = None
    if P["split_kind"] == "time":
        tr, va, n = P["tr"], P["va"], P["n"]
        t0 = time.time()
        progress(f"Comparison: streaming the GNN {name}", lo + int(span * .6))
        gnn.build_inference_state(X[:tr], {t: v[:tr] for t, v in keys.items()})
        p_val = stream_scores(gnn, X, keys, tr, va)
        p_test = stream_scores(gnn, X, keys, va, n)
        thr_live = _best_threshold(yva, p_val)
        stream_seconds = time.time() - t0
        models[LIVE] = {**_metrics(yte, p_test, thr_live), "train_seconds": round(stream_seconds, 1)}
    for m in models.values():
        m.pop("roc_curve", None)
    plan.stats.update({"train_rows": int(len(train_idx)), "train_rows_after": int(len(train_idx) + len(plan))})
    return {"ratio": float(ratio), "fsmote": plan.stats, "models": models,
            "best_val_ap": float(gnn.best_val_ap), "train_seconds": round(train_seconds, 1),
            "stream_seconds": round(stream_seconds, 1) if stream_seconds is not None else None,
            "epochs": gnn.history[-1]["epoch"] if gnn.history else 0}


def _variant_from_results(res: dict) -> dict:
    """The main run's own numbers, in the same shape as train_variant's."""
    fs = res.get("fsmote") or {}
    models = {k: {kk: vv for kk, vv in v.items() if kk != "roc_curve"} for k, v in (res.get("models") or {}).items()}
    g = res.get("gnn") or {}
    return {"ratio": 0.0 if fs.get("method") == "off" else float(fs.get("ratio", fsmote_ratio()) or 1.0),
            "fsmote": fs, "models": models, "best_val_ap": g.get("best_val_ap"),
            "train_seconds": g.get("train_seconds"), "stream_seconds": g.get("stream_seconds"),
            "epochs": g.get("epochs_run")}


def oversampling_block(res: dict, P: dict, progress=lambda msg, pct: None, lo=0, hi=100) -> dict:
    """{"main": "with"|"without", "with": variant, "without": variant}: the main run's GNN and its counterpart
    trained on the same data with oversampling switched the other way."""
    main = _variant_from_results(res)
    main_key = "with" if main["ratio"] > 0 else "without"
    other = train_variant(P, 0.0 if main_key == "with" else (fsmote_ratio() or 1.0), progress, lo, hi)
    out = {"main": main_key, main_key: main, ("without" if main_key == "with" else "with"): other}
    out["computed_at"] = time.strftime("%Y-%m-%d %H:%M")
    return out


def oversampling_comparison(csv_path: str | Path, res: dict, progress=lambda msg, pct: None) -> dict:
    """For a dataset trained before the comparison existed: prepare the file again and train only the
    counterpart GNN. Same data, split and seed as the main run, so its numbers line up with the scorecards."""
    P = prepare(csv_path, progress)
    return oversampling_block(res, P, progress, 35, 98)
