"""Automated fraud detection on a new card file: a transferred GNN, the DQN decision agent, and case grouping.

    source GNNs (IEEE-CIS, European, Brazil)  -->  base GNN for the target  -->  RL agent per time window
                                                   (zero-shot / fine-tuned /     sets the threshold, the
                                                    target-only, 20% labels)     graph-evidence weight and
                                                                                 pseudo-labelling
                                                                                        |
            every transaction gets a decision, a reason, and (if escalated) a case  <---+

1. Base GNNs. With 20% of the target's labels revealed (the same mask as the transfer experiment):
     zero-shot    the source GNN with the best validation PR-AUC on the revealed labels, unchanged; its
                  starting threshold flags 1% of validation volume (no labels needed: score scales don't
                  carry between datasets, so the source's own threshold is meaningless here)
     fine-tuned   each source GNN fine-tuned on the revealed labels; the one with the best validation PR-AUC
     target-only  a GNN trained from scratch on the revealed labels
2. The agent (rl_agent.py) is trained on the validation period of each base, replaying it with fresh 20%
   label samples and labels arriving `delay` windows late, and is compared on the test period with the
   GNN at a fixed threshold and with the best single setting found on validation data.
3. Automated deployment: the base whose agent earned the most validation reward is deployed on the test
   stream, still learning online from the delayed labels. Every test transaction gets one of four outcomes:
     block_case   escalated, and linked to other escalations in a case (an account-takeover burst or a
                  ring across accounts): block the card and open a case
     escalate     escalated on its own: to an analyst
     held_back    not escalated although suspicious (the GNN alone would have alerted, or the score was
                  within 1 log-odds of the threshold): logged with the reason the agent held it back
     approve      cleared
4. Cases are grown as escalations arrive, by linking each escalated transaction to earlier ones that share
   its card, device or IP address, its account within 24 hours, or a merchant within 72 hours (merchant
   links only between different accounts: the cash-out point of a coordinated campaign). A case spanning
   two or more accounts is reported as a ring.
"""
from __future__ import annotations

import copy
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from . import rl_agent as R
from . import transfer_rl as T
from .rl_pipeline import _best_logit_threshold, _init_counts, _trace_summary
from .transfer_target import SOURCE_NAMES, load_target

DECISION_LABELS = {"block_case": "Blocked: case opened", "escalate": "Escalated to analyst",
                   "held_back": "Not escalated (held back)", "approve": "Approved"}
LINK_LABELS = {"account": "same account within 24 h", "card": "same card", "device": "same device",
               "ip": "same IP address", "merchant": "same merchant within 72 h (different accounts)"}
ACCOUNT_LINK_SECS = 24 * 3600
MERCHANT_LINK_SECS = 72 * 3600


# ------------------------------------------------------------------ base GNNs
def base_gnns(g, vis, source_models: dict, progress=print, fsmote_ratio=1.0) -> dict:
    """{name: {"z", "H", "theta0", "source", "val_pr_auc"}} for zero-shot, fine-tuned and target-only.
    With no source models, only the target-only GNN (no transfer learning)."""
    from sklearn.metrics import average_precision_score
    tr, va, te = g.split()
    va_l = va[vis[va]]
    out = {}

    def val_ap(z):
        return float(average_precision_score(g.y[va_l], z[va_l])) if g.y[va_l].sum() else 0.0

    if source_models:   # without source GNNs, only the target-only base (no transfer)
        zs = {}
        for s, m in source_models.items():
            z, H = T.full_scores(m, g)
            zs[s] = (val_ap(z), z, H)
        s_best = max(zs, key=lambda s: zs[s][0])
        ap, z, H = zs[s_best]
        k = max(1, int(round(len(va) * 0.01)))
        out["zero_shot"] = {"z": z, "H": H, "theta0": float(np.sort(z[va])[-k]), "source": s_best, "val_pr_auc": ap,
                            "candidates": {s: round(v[0], 4) for s, v in zs.items()}}
        progress(f"Zero-shot base: {SOURCE_NAMES.get(s_best, s_best)} (validation PR-AUC {ap:.3f})")

        best = None
        for s, m in source_models.items():
            ft = T.train_target(g, vis, init=m, fsmote_ratio=fsmote_ratio)
            if ft is not None and (best is None or ft.best_val_ap > best[1].best_val_ap):
                best = (s, ft)
        s_ft, m_ft = best
        z, H = T.full_scores(m_ft, g)
        out["fine_tuned"] = {"z": z, "H": H, "theta0": _best_logit_threshold(g.y[va_l], z[va_l]), "source": s_ft,
                             "val_pr_auc": val_ap(z), "model": m_ft}
        progress(f"Fine-tuned base: from {SOURCE_NAMES.get(s_ft, s_ft)} (validation PR-AUC {out['fine_tuned']['val_pr_auc']:.3f})")

    m_to = T.train_target(g, vis, fsmote_ratio=fsmote_ratio)
    z, H = T.full_scores(m_to, g)
    out["target_only"] = {"z": z, "H": H, "theta0": _best_logit_threshold(g.y[va_l], z[va_l]), "source": None,
                          "val_pr_auc": val_ap(z), "model": m_to}
    progress(f"Target-only base: validation PR-AUC {out['target_only']['val_pr_auc']:.3f}")
    return out


# ------------------------------------------------------------------ agent per base
def make_envs(g, z, H, theta0, vis, window_hours, delay, label_rate=0.2):
    tr, va, te = g.split()
    secs = window_hours * 3600
    w_val, w_test = R.windows_for(g.t, va, secs), R.windows_for(g.t, te, secs)
    prior = float(g.y[tr[vis[tr]]].mean()) if vis[tr].sum() else 0.01
    f_tr, l_tr = _init_counts(g, tr, vis)
    f_tv, l_tv = _init_counts(g, np.concatenate([tr, va]), vis)
    proj = R.make_projector(H[va])
    common = dict(y=g.y, read=g.read, write=g.write, prior=prior, theta0=theta0, label_rate=label_rate,
                  delay=delay, fp_cost=1.0, projector=proj)
    return (R.FraudStreamEnv(z, H, windows=w_val, init_fraud=f_tr, init_legit=l_tr, **common),
            R.FraudStreamEnv(z, H, windows=w_test, init_fraud=f_tv, init_legit=l_tv, **common))


def agent_experiment(g, base: dict, vis, window_hours=2.0, delay=3, episodes=300, agent_seeds=3, seed=42,
                     progress=print) -> dict:
    tr, va, te = g.split()
    val_env, test_env = make_envs(g, base["z"], base["H"], base["theta0"], vis, window_hours, delay)
    static = R.best_static_action(val_env, seed=seed)
    fixed_a = R.ACTIONS.index((0.0, 0.0, None))
    pol = {"fixed": R.evaluate_run(R.run_policy(test_env, fixed_a, vis), te),
           "static": R.evaluate_run(R.run_policy(test_env, static["action"], vis), te)}
    masks = [np.random.default_rng(seed + 991 * e).uniform(size=g.n) < 0.2 for e in range(4)]
    agents, online, curves, val_rewards = [], [], [], []
    for k in range(agent_seeds):
        ag, curve = R.train_agent(val_env, episodes=episodes, seed=seed + 100 * k)
        val_rewards.append(float(np.mean([R.run_policy(val_env, "agent", m, agent=ag).total_reward for m in masks])))
        agents.append(ag)
        pol.setdefault("_agent", []).append(R.evaluate_run(R.run_policy(test_env, "agent", vis, agent=ag), te))
        pol.setdefault("_online", []).append(R.evaluate_run(
            R.run_policy(test_env, "agent", vis, agent=copy.deepcopy(ag), online=True, eps=0.02), te))
        curves.append([c["reward"] for c in curve])
        progress(f"agent {k + 1}/{agent_seeds}: validation reward {val_rewards[-1]:.2f}, "
                 f"test F1 {pol['_agent'][-1]['f1']:.3f}")

    def agg(runs):
        keys = [k for k in ("f1", "precision", "recall", "pr_auc", "roc_auc", "recall_at_1pct", "tp", "fp",
                            "reward", "reward_all") if k in runs[0]]
        return {**{k: float(np.mean([r[k] for r in runs])) for k in keys},
                "sd": {k: float(np.std([r[k] for r in runs])) for k in keys}}
    pol["agent"], pol["online"] = agg(pol.pop("_agent")), agg(pol.pop("_online"))
    best_k = int(np.argmax(val_rewards))
    return {"policies": pol, "static_action": list(R.ACTIONS[static["action"]]),
            "static_val_reward": static["val_reward"], "agent_val_reward": float(np.mean(val_rewards)),
            "agent_val_reward_sd": float(np.std(val_rewards)), "training_curve": np.mean(curves, axis=0).round(4).tolist(),
            "windows_val": len(val_env.windows), "windows_test": len(test_env.windows), "prior": val_env.prior,
            "_agent": agents[best_k], "_test_env": test_env}


# ------------------------------------------------------------------ cases and rings
def _key(v):
    """An entity key, or None when missing (pandas often stores a missing value as NaN, which is truthy and,
    as the same NaN object, would match itself: every transaction without a device would share one)."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    v = str(v).strip()
    return v or None


class _DSU:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def group_cases(rows: pd.DataFrame) -> tuple[dict, dict]:
    """rows: escalated transactions in time order with columns i, t, account, card, device, ip, merchant.
    Returns ({row i: case root}, {case root: list of (i, j, link kind)})."""
    dsu = _DSU()
    links = defaultdict(list)
    last_by = {k: {} for k in ("account", "card", "device", "ip", "merchant")}
    for r in rows.itertuples(index=False):
        dsu.find(r.i)
        for kind in ("card", "device", "ip"):
            key = _key(getattr(r, kind))
            if key and key in last_by[kind]:
                j = last_by[kind][key]
                dsu.union(r.i, j)
                links[r.i].append((j, kind))
        acct, merch = _key(r.account), _key(r.merchant)
        a = last_by["account"].get(acct) if acct else None
        if a is not None and r.t - a[1] <= ACCOUNT_LINK_SECS:
            dsu.union(r.i, a[0])
            links[r.i].append((a[0], "account"))
        m = last_by["merchant"].get(merch) if merch else None
        if m is not None and r.t - m[1] <= MERCHANT_LINK_SECS and m[2] != acct:
            dsu.union(r.i, m[0])
            links[r.i].append((m[0], "merchant"))
        for kind in ("card", "device", "ip"):
            key = _key(getattr(r, kind))
            if key:
                last_by[kind][key] = r.i
        if acct:
            last_by["account"][acct] = (r.i, r.t)
        if merch:
            last_by["merchant"][merch] = (r.i, r.t, acct)
    root = {i: dsu.find(i) for i in rows["i"]}
    by_case = defaultdict(list)
    for i, js in links.items():
        for j, kind in js:
            by_case[root[i]].append((int(j), int(i), kind))
    return root, by_case


# ------------------------------------------------------------------ the decision log
def _fmt_offset(d):
    return f"{d:+g}"


def decision_log(g, c: pd.DataFrame, raw: pd.DataFrame, env, theta0: float, vis) -> tuple[pd.DataFrame, list, list]:
    tr, va, te = g.split()
    z, s, th = env.z[te], env.scores[te], env.theta_tx[te]
    lp, fe, le = env.lp_tx[te], env.ev_fraud[te], env.ev_legit[te]
    flag = env.flags[te]
    win = env.win_tx[te]
    trace = env.trace
    w_of = np.array([trace[k]["w"] for k in win])
    fbp = [trace[k]["feedback_precision"] for k in win]

    def col(name):
        if name not in raw:
            return [None] * len(te)
        return np.array([_key(v) for v in raw[name].iloc[te].tolist()], dtype=object)

    df = pd.DataFrame({
        "i": te, "transaction_id": c["tx_id"].to_numpy()[te], "t": g.t[te], "timestamp": col("timestamp"),
        "window": win, "account": col("account_id"), "card": col("card_token"), "device": col("device_id"),
        "ip": col("ip_address"), "merchant": col("merchant_id"), "merchant_name": col("merchant_name"),
        "merchant_category": col("merchant_category"), "channel": col("channel"), "country": col("tx_country"),
        "amount": g.amount[te], "gnn_logit": z, "gnn_prob": 1 / (1 + np.exp(-np.clip(z, -30, 30))),
        "graph_evidence": lp, "evidence_weight": w_of, "linked_fraud_labels": fe, "linked_legit_labels": le,
        "decision_score": s, "threshold": th, "threshold_offset": th - theta0, "escalated": flag,
        "label_revealed": vis[te], "is_fraud": g.y[te]})

    # reasons
    reasons, decisions = [], []
    for k, r in enumerate(df.itertuples(index=False)):
        rs = []
        off = r.threshold_offset
        ev = r.evidence_weight * r.graph_evidence
        if r.escalated:
            rs.append(f"score {r.decision_score:.2f} ≥ threshold {r.threshold:.2f}")
            if r.gnn_logit < theta0:
                if ev > 0.05:
                    rs.append(f"graph evidence raised it by {ev:.2f} ({r.linked_fraud_labels:.0f} fraud labels on linked "
                              "account, card or device)")
                if off < 0:
                    rs.append(f"agent lowered the threshold by {-off:g} for this window")
            decisions.append("escalate")
        else:
            gnn_alone = r.gnn_logit >= theta0
            near = r.decision_score >= r.threshold - 1.0
            if gnn_alone or near:
                if gnn_alone and off > 0 and r.gnn_logit + max(ev, 0) < r.threshold:
                    p = fbp[k]
                    rs.append(f"agent raised the threshold by {off:g} for this window"
                              + (f" (recent alert precision {p:.0%})" if p is not None else ""))
                if ev < -0.05:
                    rs.append(f"graph evidence lowered it by {-ev:.2f} ({r.linked_legit_labels:.0f} legitimate labels on "
                              "linked account, card or device)")
                if not rs or not gnn_alone:
                    rs.append(f"score {r.decision_score:.2f} is {r.threshold - r.decision_score:.2f} below the threshold")
                if gnn_alone:
                    rs.insert(0, "the GNN alone would have alerted")
                decisions.append("held_back")
            else:
                decisions.append("approve")
        reasons.append("; ".join(rs))
    df["decision"], df["reason"] = decisions, reasons

    # cases, grown from escalations in time order
    esc = df[df["escalated"]].sort_values("t")
    root, links = group_cases(esc[["i", "t", "account", "card", "device", "ip", "merchant"]])
    df["case_id"] = df["i"].map(root)
    sizes = df.groupby("case_id")["i"].transform("size")
    df.loc[df["escalated"] & (sizes >= 2), "decision"] = "block_case"
    df.loc[sizes.isna() | (sizes < 2), "case_id"] = np.nan

    cases = []
    by_root = df[df["case_id"].notna()].groupby("case_id")
    for k, (cid, m) in enumerate(sorted(by_root, key=lambda x: x[1]["t"].min())):
        kinds = sorted({kind for _, _, kind in links.get(cid, [])})
        n_acc = m["account"].nunique()
        cases.append({"case": f"C{k + 1:03d}", "root": int(cid), "type": "ring" if n_acc >= 2 else "account burst",
                      "transactions": int(len(m)), "accounts": int(n_acc), "frauds": int(m["is_fraud"].sum()),
                      "start": str(m["timestamp"].iloc[0]), "end": str(m["timestamp"].iloc[-1]),
                      "hours": round(float(m["t"].max() - m["t"].min()) / 3600, 1),
                      "amount": round(float(m["amount"].sum()), 2), "links": [LINK_LABELS[x] for x in kinds],
                      "merchants": m["merchant_name"].dropna().value_counts().head(4).index.tolist(),
                      "account_list": m["account"].unique().tolist()[:12],
                      "transaction_ids": m["transaction_id"].tolist()[:40]})
    name = {c_["root"]: c_["case"] for c_ in cases}
    df["case"] = df["case_id"].map(name)
    df["case_type"] = df["case_id"].map({c_["root"]: c_["type"] for c_ in cases})

    # not escalated but tied to a ring: the same device or IP address as a ring member, or the same account
    # within an hour of one (a card belongs to one account, so "same card" would pull in the account's whole history)
    ring_rows = df[df["case_type"] == "ring"]
    keys = {kind: {_key(k): v for k, v in zip(ring_rows[kind], ring_rows["case"]) if _key(k)} for kind in ("device", "ip")}
    acc_t = defaultdict(list)
    for r in ring_rows.itertuples(index=False):
        acc_t[r.account].append((r.t, r.case))
    linked_case, linked_why = [None] * len(df), [None] * len(df)
    for k, r in enumerate(df.itertuples(index=False)):
        if r.escalated:
            continue
        for kind in ("device", "ip"):
            v = _key(getattr(r, kind))
            if v and v in keys[kind]:
                linked_case[k], linked_why[k] = keys[kind][v], LINK_LABELS[kind]
                break
        else:
            for t0, case in acc_t.get(r.account, []):
                if abs(r.t - t0) <= 3600:
                    linked_case[k], linked_why[k] = case, "same account within 1 h"
                    break
    df["ring_linked"], df["ring_link"] = linked_case, linked_why
    return df.drop(columns=["case_id"]), cases, links


def _known_rings(g, raw, ring_id, vis, df) -> list[dict] | None:
    """Score the agent against the rings planted by fraud_rings.inject_rings (None when the file has none)."""
    from .fraud_rings import score_known_rings
    idx = np.flatnonzero(ring_id != "")
    if not len(idx):
        return None
    tr, va, te = g.split()
    period = np.full(g.n, "test", dtype=object)
    period[tr], period[va] = "train", "validation"
    rows = pd.DataFrame({"ring": ring_id[idx], "period": period[idx], "account": raw["account_id"].iloc[idx].to_numpy(),
                         "pattern": raw["fraud_pattern"].iloc[idx].to_numpy() if "fraud_pattern" in raw else "",
                         "revealed": vis[idx], "is_fraud": g.y[idx]})
    # a ring belongs to the period of its first transaction
    first = rows.groupby("ring")["period"].transform("first")
    rows["period"] = first
    return score_known_rings(df, rows)


def funnel(df: pd.DataFrame) -> list[dict]:
    out = []
    for d in ("block_case", "escalate", "held_back", "approve"):
        m = df[df["decision"] == d]
        out.append({"decision": d, "label": DECISION_LABELS[d], "n": int(len(m)), "frauds": int(m["is_fraud"].sum()),
                    "share": float(len(m) / max(len(df), 1))})
    return out


def _json_safe(o):
    """NaN and infinities become None and NumPy scalars become Python ones, so the results store as JSON."""
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# ------------------------------------------------------------------ the whole run
def run_automation(target_path, source_models: dict, out_dir, progress=lambda m, p: None, label_rate=0.2,
                   mask_seed=1, window_hours=2.0, delay=3, episodes=300, agent_seeds=3, fsmote_ratio=1.0) -> dict:
    from pathlib import Path

    import joblib

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    progress("Reading the target file", 2)
    c, raw, X = load_target(target_path)
    g = T.target_graph(c, X)
    tr, va, te = g.split()
    vis = T.visible_mask(g.n, label_rate, mask_seed)

    progress("Building the base GNNs from the source models", 5)
    bases = base_gnns(g, vis, source_models, progress=lambda m: progress(m, 15), fsmote_ratio=fsmote_ratio)

    res = {}
    for k, (name, b) in enumerate(bases.items()):
        pct = 25 + 20 * k
        progress(f"Training the agent on the {name.replace('_', '-')} GNN", pct)
        res[name] = agent_experiment(g, b, vis, window_hours=window_hours, delay=delay, episodes=episodes,
                                     agent_seeds=agent_seeds, progress=lambda m, p=pct: progress(f"{name}: {m}", p))

    deployed = max(res, key=lambda k: res[k]["agent_val_reward"])
    progress(f"Automated run on the test period with the {deployed.replace('_', '-')} GNN", 88)
    env = R.run_policy(res[deployed]["_test_env"], "agent", vis, agent=copy.deepcopy(res[deployed]["_agent"]),
                       online=True, eps=0.0)
    metrics = R.evaluate_run(env, te)
    df, cases, _ = decision_log(g, c, raw, env, bases[deployed]["theta0"], vis)
    known_rings = None
    if "fraud_ring_id" in raw:   # ground truth from fraud_rings.inject_rings: evaluation only, never a feature
        ring_id = raw["fraud_ring_id"].fillna("").astype(str).to_numpy()
        df["known_ring"] = ring_id[te]
        known_rings = _known_rings(g, raw, ring_id, vis, df)
    adversarial = None
    if "adversarial_group" in raw:   # planted by adversarial.build_adversarial_file: evaluation only
        from .adversarial import adversarial_summary
        df["adversarial_group"] = raw["adversarial_group"].fillna("").astype(str).to_numpy()[te]
        adversarial = adversarial_summary(df, bases[deployed]["theta0"],
                                          [x - bases[deployed]["theta0"] for x in _trace_summary(env)["theta"]])
    df.to_csv(out_dir / "decisions.csv", index=False, float_format="%.5g")
    trace = _trace_summary(env)
    win_t = [str(raw["timestamp"].iloc[w[0]]) if "timestamp" in raw else "" for w in env.windows]

    rings = [x for x in cases if x["type"] == "ring"]
    held = df[df["decision"] == "held_back"].sort_values("gnn_logit", ascending=False)
    linked = df[df["ring_linked"].notna()]
    fr = df[df["is_fraud"] == 1]
    summary = {
        "funnel": funnel(df), "metrics": metrics, "cases": cases,
        "case_stats": {"cases": len(cases), "rings": len(rings), "bursts": len(cases) - len(rings),
                       "frauds_in_cases": int(df[df["case"].notna()]["is_fraud"].sum()),
                       "tx_in_cases": int(df["case"].notna().sum()),
                       "frauds_in_rings": int(df[df["case_type"] == "ring"]["is_fraud"].sum()),
                       "tx_in_rings": int((df["case_type"] == "ring").sum()),
                       "ring_linked": int(len(linked)), "ring_linked_frauds": int(linked["is_fraud"].sum())},
        "held_back": held.head(250)[["transaction_id", "timestamp", "account", "merchant_name", "amount", "gnn_prob",
                                     "decision_score", "threshold", "reason", "ring_linked", "is_fraud",
                                     "label_revealed"]].to_dict("records"),
        "held_back_stats": {"n": int(len(held)), "frauds": int(held["is_fraud"].sum()),
                            "gnn_alone": int(held["reason"].str.startswith("the GNN alone").sum()),
                            "gnn_alone_frauds": int(held[held["reason"].str.startswith("the GNN alone")]["is_fraud"].sum()),
                            "by_threshold": int(held["reason"].str.contains("raised the threshold").sum()),
                            "by_evidence": int(held["reason"].str.contains("graph evidence lowered").sum())},
        "ring_linked": linked.head(150)[["transaction_id", "timestamp", "account", "merchant_name", "amount", "gnn_prob",
                                          "decision", "ring_linked", "ring_link", "is_fraud"]].to_dict("records"),
        "missed_frauds": {"total": int(len(fr)), "escalated": int(fr["escalated"].sum()),
                          "held_back": int((fr["decision"] == "held_back").sum()),
                          "approved": int((fr["decision"] == "approve").sum())},
        "known_rings": known_rings, "adversarial": adversarial,
        "timeline": {"t": win_t, "theta_offset": [round(x - bases[deployed]["theta0"], 3) for x in trace["theta"]],
                     "w": trace["w"], "pl": trace["pl"],
                     "alerts": [x["alerts"] for x in env.trace], "tp": [x["tp"] for x in env.trace],
                     "fp": [x["fp"] for x in env.trace], "fn": [x["fn"] for x in env.trace]},
    }
    joblib.dump({"agent": res[deployed]["_agent"], "base": deployed, "theta0": bases[deployed]["theta0"],
                 "gnn": bases[deployed].get("model")}, out_dir / "automation.joblib", compress=3)
    progress("Done", 100)
    return _json_safe({
        "target": {"rows": int(g.n), "frauds": int(g.y.sum()), "test_rows": int(len(te)), "test_frauds": int(g.y[te].sum()),
                   "revealed_val_frauds": int(g.y[va][vis[va]].sum()), "revealed_train_frauds": int(g.y[tr][vis[tr]].sum())},
        "params": {"label_rate": label_rate, "window_hours": window_hours, "delay_windows": delay, "fsmote_ratio": fsmote_ratio,
                   "episodes": episodes, "agent_seeds": agent_seeds},
        "bases": {k: {"source": bases[k]["source"], "theta0": bases[k]["theta0"], "val_pr_auc": bases[k]["val_pr_auc"],
                      **({"candidates": bases[k]["candidates"]} if "candidates" in bases[k] else {}),
                      **{kk: vv for kk, vv in v.items() if not kk.startswith("_")}} for k, v in res.items()},
        "deployed": deployed, "automation": summary, "seconds": round(time.time() - t0, 1)})
