"""Transfer learning onto a labelled-but-small target: the synthetic South African card file.

    sources (pre-training)                            target (fine-tuning)
    IEEE-CIS            real labels, 590,540 rows      credit_card_clean.csv: 17,097 transactions over
    European Cardholder real labels, 284,807 rows        23 days, 180 frauds (1.05%), accounts, cards,
    Brazilian Card Holder rule-based pseudo-labels       devices, merchants, countries and balances
                                                         like a South African bank's card feed

The datasets share almost no columns, so everything goes through the shared schema of transfer.py:
behaviour that means the same in every file (amount against the account's own history, velocity,
new device / merchant / country for the account, amount against the balance, hour of day),
standardised within each file. The European file has no account, so its history features are 0.

Three ways to get a GNN for the target, compared on the same time-ordered split (60/15/25) with each
test transaction scored only from earlier ones:
    zero-shot     the source GNN applied unchanged (no target labels at all)
    target-only   a GNN trained from scratch on the revealed target labels
    fine-tuned    the source GNN's weights and feature scaling as the starting point, trained further
                  on the same target labels (lower learning rate; the source weights are themselves a
                  candidate for early stopping, so fine-tuning can't do worse on validation than zero-shot)
at several label rates (a random share of the target's training and validation periods is revealed),
each over several random label masks.

feature_similarity() measures which shared features look alike across the datasets (their
distributions) and, more to the point for transfer, which carry the same fraud signal (per-feature
ROC-AUC per dataset, and whether it points the same way as in the target).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from . import transfer_rl as T
from .transfer import CONTINUOUS, FEATURE_LABELS, MODEL_FEATURES

SOURCE_NAMES = {"ieee": "IEEE-CIS", "ulb": "European Cardholder", "br": "Brazilian Card Holder",
                "target": "Synthetic SA (target)"}
# loss shares when sources are combined; Brazil's weak pseudo-labels get a fixed, small share
SOURCE_SETS = {"ieee": {"ieee": 1.0}, "ulb": {"ulb": 1.0}, "br": {"br": 1.0},
               "ieee+br": {"ieee": 0.8, "br": 0.2}, "ieee+ulb+br": {"ieee": 0.4, "ulb": 0.4, "br": 0.2}}
RATES = (0.1, 0.2, 1.0)
MASKS = (1, 2, 3, 4, 5)


# ------------------------------------------------------------------ loading
def load_target(path):
    c, raw = T.load_sa(path)
    X = T.features(c)
    return c, raw, X


def load_sources(ieee_transaction, ieee_identity, ulb_path, brazil_path, target_raw=None, progress=print):
    """{name: (canonical df, X, y)}. European rows copied into the target file are dropped from the source."""
    from .transfer import load_brazil, load_ieee, pseudo_label_brazil
    out = {}
    if ieee_transaction:
        progress("Reading IEEE-CIS")
        ie = load_ieee(ieee_transaction, ieee_identity)
        out["ieee"] = (ie, T.features(ie), ie["label"].to_numpy(int))
    if ulb_path:
        progress("Reading the European Cardholder file")
        ul = T.load_ulb_canonical(ulb_path, exclude=None if target_raw is None else target_raw[["Time", "Amount"]])
        out["ulb"] = (ul, T.features(ul, no_history=True), ul["label"].to_numpy(int))
    if brazil_path:
        progress("Reading the Brazilian Card Holder file and applying the pseudo-label rules")
        br, br_raw = load_brazil(brazil_path)
        y_br, _ = pseudo_label_brazil(br, br_raw)
        br["label"] = y_br
        out["br"] = (br, T.features(br), y_br)
    return out


# ------------------------------------------------------------------ feature similarity
def _auc(y, x):
    if y.min() == y.max() or np.nanstd(x) == 0:
        return None
    return float(roc_auc_score(y, x))


def _wasserstein(a, b, n=2000, seed=0):
    """1-D earth mover's distance between two samples (both already standardised within their file)."""
    rng = np.random.default_rng(seed)
    a = np.sort(rng.choice(a, min(n, len(a)), replace=False)) if len(a) > n else np.sort(a)
    b = np.sort(rng.choice(b, min(n, len(b)), replace=False)) if len(b) > n else np.sort(b)
    q = np.linspace(0, 1, 201)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def feature_similarity(datasets: dict, features=MODEL_FEATURES, domain_sample=20000, seed=0) -> dict:
    """datasets: {name: (X, y)} with "target" among them. Returns per-feature statistics per dataset."""
    rng = np.random.default_rng(seed)
    tX, ty = datasets["target"]
    rows = []
    for j, f in enumerate(features):
        row = {"feature": f, "label": FEATURE_LABELS.get(f, f), "continuous": f in CONTINUOUS, "per_dataset": {}}
        for name, (X, y) in datasets.items():
            x = X[:, j].astype(float)
            fr, lg = x[y == 1], x[y == 0]
            avail = bool(x.std() > 0)   # a field the file doesn't have is constant (0) after mapping
            row["per_dataset"][name] = {
                "available": avail, "mean": float(x.mean()), "std": float(x.std()), "nonzero": float((x != 0).mean()),
                "fraud_mean": float(fr.mean()) if len(fr) else None, "legit_mean": float(lg.mean()),
                "auc": _auc(y, x),
                "distance_to_target": None if name == "target" or not avail
                else _wasserstein(x, tX[:, j].astype(float)),
            }
        t_auc = row["per_dataset"]["target"]["auc"]
        for name, d in row["per_dataset"].items():
            a = d["auc"]
            # "agrees": the feature separates fraud in the same direction as in the target (both AUCs on the
            # same side of 0.5, ignoring features that carry no signal (|AUC-0.5| < 0.02) in either)
            d["agrees_with_target"] = (None if a is None or t_auc is None or abs(a - .5) < .02 or abs(t_auc - .5) < .02
                                       else bool((a - .5) * (t_auc - .5) > 0))
        rows.append(row)

    # signature similarity: correlation between datasets of the per-feature (AUC - 0.5) vectors
    names = list(datasets)
    sig = {n: np.array([(r["per_dataset"][n]["auc"] or .5) - .5 for r in rows]) for n in names}
    corr = {a: {b: float(np.corrcoef(sig[a], sig[b])[0, 1]) if sig[a].std() and sig[b].std() else None
                for b in names} for a in names}

    # domain classifier: how easily can each source be told apart from the target? Only on the features both
    # files actually have, otherwise a missing field (always 0) gives the source away trivially
    domain = {}
    for name, (X, _) in datasets.items():
        if name == "target":
            continue
        cols = [j for j in range(len(features)) if X[:, j].std() > 0 and tX[:, j].std() > 0]
        a = X[rng.choice(len(X), min(domain_sample, len(X)), replace=False)][:, cols]
        b = tX[rng.choice(len(tX), min(domain_sample, len(tX)), replace=False)][:, cols]
        Z = np.vstack([a, b]).astype(float)
        d = np.r_[np.zeros(len(a)), np.ones(len(b))]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        p = cross_val_predict(clf, Z, d, cv=3, method="predict_proba")[:, 1]
        clf.fit(Z, d)
        top = np.argsort(-np.abs(clf.coef_[0]))[:5]
        domain[name] = {"auc": float(roc_auc_score(d, p)), "features_used": [features[j] for j in cols],
                        "most_different": [{"feature": features[cols[k]], "coef": float(clf.coef_[0][k])}
                                           for k in top]}
    return {"features": rows, "signature_corr": corr, "domain_classifier": domain, "datasets": names}


# ------------------------------------------------------------------ target experiments
def run_target(g, source_models: dict, source_info: dict, rates=RATES, masks=MASKS, progress=print,
               rich=None) -> tuple[dict, dict]:
    """Zero-shot, target-only and fine-tuned GNNs on the target graph g. rich: optional (GraphData) with the
    target's own extra fields, for a target-only reference the sources can't match. Returns (results, scores)."""
    tr, va, te = g.split()
    res = {"split": {"rows": [len(tr), len(va), len(te)], "frauds": [int(g.y[s].sum()) for s in (tr, va, te)]},
           "zero_shot": {}, "runs": []}
    scores = {}

    def r4(d):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}

    for s, m in source_models.items():
        z, _ = T.full_scores(m, g)
        scores[("zero", s)] = z
        k_va = max(1, int(round(len(va) * 0.01)))
        budget_thr = float(np.sort(z[va])[-k_va])     # flag 1% of the volume: needs no labels
        # the same source GNN with every link removed: what its per-transaction behaviour alone carries over
        zl, _ = m.embed_new(g.X, {t: [None] * g.n for t in g.entity_keys})
        res["zero_shot"][s] = {
            "carried_threshold": r4(T.report(g, z, np.ones(g.n, bool), thr=source_info[s]["threshold_logit"])),
            "alert_budget_1pct": r4(T.report(g, z, np.ones(g.n, bool), thr=budget_thr)),
            "no_links": r4(T.report(g, zl, np.ones(g.n, bool), thr=float(np.sort(zl[va])[-k_va]))),
            "relations": list(getattr(m, "seen_types", [])),
            "features_known": int(m.live_features().sum()),
        }
        progress(f"zero-shot {s}: PR-AUC {res['zero_shot'][s]['alert_budget_1pct']['pr_auc']:.3f}")

    for rate in rates:
        for mk in (masks if rate < 1 else masks[:1]):
            vis = T.visible_mask(g.n, rate, mk) if rate < 1 else np.ones(g.n, bool)
            lab = {"train_frauds": int(g.y[tr][vis[tr]].sum()), "val_frauds": int(g.y[va][vis[va]].sum())}
            models = {"target_only": (None, g)}
            if rich is not None:
                models["target_only_rich"] = (None, rich)
            models.update({f"finetune_{s}": (m, g) for s, m in source_models.items()})
            for name, (init, gg) in models.items():
                t0 = time.time()
                m = T.train_target(gg, vis, init=init)
                if m is None:
                    res["runs"].append({"rate": rate, "mask": mk, "model": name, "skipped": True, **lab})
                    continue
                z, _ = T.full_scores(m, gg)
                if mk == masks[0]:
                    scores[(name, rate)] = z
                res["runs"].append({"rate": rate, "mask": mk, "model": name, **lab,
                                    "epochs": int(m.history[-1]["epoch"]) if m.history else 0, "lr": m.lr,
                                    "kept_source_weights": bool(init is not None and m.history and
                                                                max(m.history, key=lambda h: h["val_ap"])["epoch"] == 0),
                                    "secs": round(time.time() - t0, 1), **r4(T.report(gg, z, vis))})
                progress(f"{rate:.0%} labels, mask {mk}, {name}: PR-AUC {res['runs'][-1]['pr_auc']:.3f}")
    return res, scores


def summarise(res: dict) -> list[dict]:
    """Mean and standard deviation over label masks, per label rate and model."""
    df = pd.DataFrame([r for r in res["runs"] if not r.get("skipped")])
    out = []
    for (rate, model), g in df.groupby(["rate", "model"], sort=False):
        row = {"rate": float(rate), "model": model, "n": int(len(g))}
        for k in ("roc_auc", "pr_auc", "f1", "precision", "recall", "recall_at_1pct", "tp", "fp"):
            row[k] = float(g[k].mean())
            row[k + "_sd"] = float(g[k].std(ddof=0))
        row["kept_source_weights"] = int(g["kept_source_weights"].sum())
        out.append(row)
    return out


# ------------------------------------------------------------------ the whole experiment (app + command)
class FinetuneError(Exception):
    """Shown to the user as-is."""


def _sets_for(available: set, wanted=None) -> dict:
    sets = {k: v for k, v in SOURCE_SETS.items() if set(v) <= available}
    if wanted:
        sets = {k: v for k, v in sets.items() if k in wanted}
    return sets


def run_finetune_experiment(target_path, sources: dict, out_dir, progress=lambda m, p: None, rates=RATES,
                            masks=MASKS, source_sets=None, pretrained_dir=None) -> dict:
    """sources: {"ieee_transaction", "ieee_identity", "ulb", "brazil"} paths (any may be missing).
    pretrained_dir: a folder with src_<set>.joblib / .json from an earlier run, reused instead of pre-training."""
    import json
    from pathlib import Path

    import joblib

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    progress("Reading the target file", 2)
    try:
        c, raw, X = load_target(target_path)
    except KeyError as e:
        raise FinetuneError(f"The target file is missing the column {e}. It needs the synthetic South African "
                            "schema (transaction_id, Time, Amount, account_id, card_token, device_id, merchant_id, "
                            "tx_country, account_balance, Class).")
    if c["label"].sum() < 10:
        raise FinetuneError("The target file has fewer than 10 frauds, too few to fine-tune and evaluate on.")
    g = T.target_graph(c, X)
    Xr, _ = T.sa_rich_features(raw, X)
    g_rich = T.target_graph(c, Xr)

    progress("Reading the source datasets", 5)
    src = load_sources(sources.get("ieee_transaction"), sources.get("ieee_identity"), sources.get("ulb"),
                       sources.get("brazil"), target_raw=raw, progress=lambda m: progress(m, 6))
    if not src:
        raise FinetuneError("No source dataset was given.")
    sets = _sets_for(set(src), source_sets)
    models, infos = {}, {}
    for i, (name, members) in enumerate(sets.items()):
        pct = 10 + int(40 * i / max(len(sets), 1))
        cached = Path(pretrained_dir) / f"src_{name}.joblib" if pretrained_dir else None
        if cached is not None and cached.exists():
            progress(f"Using the pre-trained {name} GNN", pct)
            models[name] = joblib.load(cached)
            infos[name] = json.load(open(cached.with_suffix(".json")))
        else:
            progress(f"Pre-training the GNN on {name}", pct)
            models[name], infos[name] = T.pretrain({m: (*src[m], w) for m, w in members.items()},
                                                   progress=lambda m, n=name, p=pct: progress(f"Pre-training {n}:{m}", p))
            joblib.dump(models[name], out_dir / f"src_{name}.joblib")
            json.dump(infos[name], open(out_dir / f"src_{name}.json", "w"), default=float)

    progress("Comparing features across the datasets", 52)
    ds = {k: (v[1], v[2]) for k, v in src.items()}
    ds["target"] = (X, g.y)
    sim = feature_similarity(ds)

    progress("Zero-shot, target-only and fine-tuned GNNs on the target", 55)
    n_jobs = sum(len(masks) if r < 1 else 1 for r in rates)
    done = [0]

    def prog(m):
        done[0] += 1
        progress(m, min(95, 55 + int(40 * done[0] / max(1, n_jobs * (len(models) + 2) + len(models)))))
    res, scores = run_target(g, models, infos, rates=rates, masks=masks, rich=g_rich, progress=prog)
    summary = summarise(res)

    # the deployable model: the source whose fine-tuned GNN did best on validation with all target labels
    progress("Fine-tuning the final model on all target labels", 96)
    vis = np.ones(g.n, bool)
    best_name, best_m, best_ap = None, None, -1
    for name, m in models.items():
        ft = T.train_target(g, vis, init=m)
        if ft is not None and ft.best_val_ap > best_ap:
            best_name, best_m, best_ap = name, ft, ft.best_val_ap
    z_best, _ = T.full_scores(best_m, g)
    tr, va, te = g.split()
    from .rl_pipeline import _best_logit_threshold
    thr = _best_logit_threshold(g.y[va], z_best[va])
    joblib.dump({"gnn": best_m, "threshold_logit": thr, "source": best_name, "features": MODEL_FEATURES},
                out_dir / "finetuned.joblib", compress=3)
    pd.DataFrame({"transaction_id": c["tx_id"], "split": np.select([np.isin(np.arange(g.n), te),
                                                                    np.isin(np.arange(g.n), va)], ["test", "val"], "train"),
                  "gnn_logit": z_best, "label": g.y}).to_csv(out_dir / "scores.csv", index=False, float_format="%.6g")
    progress("Done", 100)
    return {"target": {"rows": int(g.n), "frauds": int(g.y.sum()), "days": float((g.t[-1] - g.t[0]) / 86400),
                       "split": res["split"], "entities": {t: int(sum(v is not None for v in g.entity_keys[t]))
                                                          for t in g.entity_keys}},
            "sources": {k: {"name": SOURCE_NAMES[k], "rows": int(len(v[2])), "frauds": int(v[2].sum())}
                        for k, v in src.items()},
            "source_sets": {k: {"members": v, "info": infos[k]} for k, v in sets.items()},
            "similarity": sim, "zero_shot": res["zero_shot"], "runs": res["runs"], "summary": summary,
            "final": {"source": best_name, "val_pr_auc": float(best_ap), "threshold_logit": float(thr),
                      "test": T.report(g, z_best, vis, thr=thr)},
            "rates": list(rates), "masks": list(masks), "seconds": round(time.time() - t_start, 1)}
