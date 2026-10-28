"""One semi-supervised GNN + DQN experiment, end to end.

    load data -> semi-supervised GNN (few labels + self-training) -> causal GNN scores and embeddings
    -> train the DQN on the validation stream -> run every policy on the test stream -> report

Policies compared on the same test stream, the same hidden labels and the same label delay:

    GNN, fixed threshold            the threshold chosen on labelled validation data, never changed
                                    (FraudGNN-RL's "w/o RL" ablation)
    GNN + LP, best fixed setting    the best single (threshold, label-propagation weight, pseudo-label)
                                    setting found on the validation stream, held fixed
    RL agent (DQN)                  the trained agent, acting greedily
    RL agent, learning online       the same agent, still learning from labelled feedback on the test
                                    stream (FraudGNN-RL's deployment loop)
    RL agent without the GNN        the GNN with every link removed (features only) under its own
                                    trained agent (FraudGNN-RL's "w/o GNN" ablation)

Every metric is computed on all test transactions with their true labels. The agent itself only ever
sees the labelled fraction, late.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from . import rl_agent as R
from .pipeline import fsmote_ratio as default_fsmote_ratio
from .rl_data import load
from .semisup import causal_scores, features_only_scores, train_semi_supervised

DEFAULTS = {
    "ulb": {"window_minutes": 10, "delay": 3, "episodes": 300},    # two days: 10-minute windows, labels 30 min late
    "ieee": {"window_minutes": 360, "delay": 4, "episodes": 600},  # six months: 6-hour windows, labels a day late
}
POLICY_ORDER = ["fixed", "static", "agent", "online", "no_gnn"]
POLICY_LABELS = {"fixed": "GNN, fixed threshold", "static": "GNN + LP, best fixed setting",
                 "agent": "RL agent (DQN)", "online": "RL agent, learning online", "no_gnn": "RL agent without the GNN"}


def _sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def _best_logit_threshold(y, z):
    prec, rec, thr = precision_recall_curve(y, z)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[int(np.nanargmax(f1[:-1]))]) if len(thr) else 0.0


def _score_report(y, z, thr):
    p = z >= thr
    tp, fp, fn = int((p & (y == 1)).sum()), int((p & (y == 0)).sum()), int((~p & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    k = max(1, int(round(len(z) * 0.01)))
    return {"roc_auc": float(roc_auc_score(y, z)), "pr_auc": float(average_precision_score(y, z)),
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "precision": prec, "recall": rec,
            "recall_at_1pct": float(y[np.argsort(-z)[:k]].sum() / max(y.sum(), 1)), "tp": tp, "fp": fp, "fn": fn}


def _full(n, start, a, fill=0.0):
    out = np.full((n,) + a.shape[1:], fill, dtype=np.float32 if a.ndim > 1 else float)
    out[start:] = a
    return out


def _init_counts(g, idx, visible):
    lab = idx[visible[idx]]
    wr = g.write[lab]
    return wr.T @ (g.y[lab] == 1).astype(float), wr.T @ (g.y[lab] == 0).astype(float)


def _trace_summary(env):
    tr = env.trace
    cum_tp = np.cumsum([x["tp"] for x in tr]).tolist()
    cum_fp = np.cumsum([x["fp"] for x in tr]).tolist()
    cum_r = np.cumsum([x["reward"] for x in tr]).round(3).tolist()
    cum_all = np.cumsum([(x["tp"] - x["fn"] - env.fp_cost * x["fp"]) / (env.prior * x["n"] + 1.0) for x in tr]).round(3).tolist()
    return {"theta": [round(x["theta"], 3) for x in tr], "w": [x["w"] for x in tr],
            "pl": [x["pl"] for x in tr], "alerts": [x["alerts"] for x in tr], "n": [x["n"] for x in tr],
            "cum_tp": cum_tp, "cum_fp": cum_fp, "cum_reward": cum_r, "cum_reward_all": cum_all,
            "pseudo_frauds": int(sum(x["pseudo_frauds"] for x in tr)), "actions": [x["action"] for x in tr]}


def prepare(kind: str, path, label_rate=0.2, rounds=2, epochs=300, max_rows=None, knn_k=5, knn_min_sim=0.9,
            seed=42, full_reference=True, fsmote=None, progress=lambda m, p: None) -> dict:
    """Stage 1: data, the semi-supervised GNN and its causal scores. Cacheable (see run_experiment)."""
    t_start = time.time()
    fsmote = default_fsmote_ratio() if fsmote is None else fsmote
    progress("Loading the dataset", 2)
    kw = {"max_rows": max_rows, "progress": progress}
    if kind == "ulb":
        kw.update(k=knn_k, min_sim=knn_min_sim)
    g = load(kind, path, **kw)
    tr, va, te = g.split()
    timings = {"load": round(time.time() - t_start, 1)}

    # ---- semi-supervised GNN
    t0 = time.time()
    semi = train_semi_supervised(g, label_rate=label_rate, rounds=rounds, seed=seed, epochs=epochs,
                                 fsmote_ratio=fsmote, full_reference=full_reference, progress=progress, pct=(12, 60))
    vis = semi["visible"]
    timings["gnn"] = round(time.time() - t0, 1)

    # ---- causal (streaming) scores and embeddings for the validation and test periods
    t0 = time.time()
    start = int(va[0])
    lab_va = va[vis[va]]
    models = {"supervised": semi["supervised"], "semi": semi["semi"]}
    if semi["reference"] is not None:
        models["reference"] = semi["reference"]
    scored = {}
    for i, (name, m) in enumerate(models.items()):
        progress(f"Scoring the validation and test periods one transaction at a time ({name} GNN)", 61 + i * 3)
        z, H = causal_scores(m, g, start)
        scored[name] = (_full(g.n, start, z), _full(g.n, start, H) if name == "semi" else None)
    zf, Hf = features_only_scores(models["semi"], g, start)
    scored["features"] = (_full(g.n, start, zf), _full(g.n, start, Hf))
    timings["scoring"] = round(time.time() - t0, 1)

    semi_report = {}
    for name, (z, _) in scored.items():
        thr_idx = va if name == "reference" else lab_va      # the reference GNN may use every label
        thr = _best_logit_threshold(g.y[thr_idx], z[thr_idx])
        semi_report[name] = {**_score_report(g.y[te], z[te], thr), "threshold_logit": thr}
    return {"g": g, "semi": {k: v for k, v in semi.items() if k not in ("supervised", "semi", "reference")},
            "semi_gnn": models["semi"], "scored": scored, "semi_report": semi_report, "timings": timings,
            "params": {"kind": kind, "label_rate": label_rate, "rounds": rounds, "epochs": epochs, "max_rows": max_rows,
                       "knn_k": knn_k, "knn_min_sim": knn_min_sim, "seed": seed, "fsmote_ratio": fsmote}}


def run_experiment(kind: str, path, out_dir, label_rate=0.2, rounds=2, episodes=None, window_minutes=None,
                   delay=None, fp_cost=1.0, agent_seeds=3, epochs=300, max_rows=None, knn_k=5, knn_min_sim=0.9, seed=42,
                   full_reference=True, fsmote=None, progress=lambda m, p: None, stage=None) -> dict:
    """stage: the output of prepare() for the same data and settings, to rerun only the RL part."""
    t_start = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    window_minutes = window_minutes or DEFAULTS[kind]["window_minutes"]
    delay = DEFAULTS[kind]["delay"] if delay is None else delay
    episodes = episodes or DEFAULTS[kind]["episodes"]
    if stage is None:
        stage = prepare(kind, path, label_rate, rounds, epochs, max_rows, knn_k, knn_min_sim, seed, full_reference,
                        fsmote, progress)
    g, semi, scored, semi_report = stage["g"], stage["semi"], stage["scored"], stage["semi_report"]
    fsmote = stage["params"]["fsmote_ratio"]
    timings = dict(stage["timings"])
    tr, va, te = g.split()
    vis = semi["visible"]

    # ---- the RL environment: windows of the validation (training) and test streams
    secs = window_minutes * 60.0
    w_val, w_test = R.windows_for(g.t, va, secs), R.windows_for(g.t, te, secs)
    prior = float(g.y[tr[vis[tr]]].mean())
    f_tr, l_tr = _init_counts(g, tr, vis)
    f_tv, l_tv = _init_counts(g, np.concatenate([tr, va]), vis)

    def envs(z, H, theta0):
        proj = R.make_projector(H[va])
        common = dict(y=g.y, read=g.read, write=g.write, prior=prior, theta0=theta0, label_rate=label_rate,
                      delay=delay, fp_cost=fp_cost, projector=proj)
        return (R.FraudStreamEnv(z, H, windows=w_val, init_fraud=f_tr, init_legit=l_tr, **common),
                R.FraudStreamEnv(z, H, windows=w_test, init_fraud=f_tv, init_legit=l_tv, **common))

    z_semi, H_semi = scored["semi"]
    theta0 = semi_report["semi"]["threshold_logit"]
    val_env, test_env = envs(z_semi, H_semi, theta0)

    t0 = time.time()
    progress("Finding the best fixed setting on the validation stream", 70)
    static = R.best_static_action(val_env, seed=seed)

    z_f, H_f = scored["features"]
    theta_f = semi_report["features"]["threshold_logit"]
    val_env_f, test_env_f = envs(z_f, H_f, theta_f)

    # ---- evaluate on the test stream, with the dataset's real (hidden) label mask
    policies, traces, per_seed = {}, {}, {"agent": [], "online": [], "no_gnn": []}

    def record(key, env):
        policies[key] = {"label": POLICY_LABELS[key], **R.evaluate_run(env, te)}
        traces[key] = _trace_summary(env)

    record("fixed", R.run_policy(test_env, R.ACTIONS.index((0.0, 0.0, None)), vis))
    record("static", R.run_policy(test_env, static["action"], vis))
    curves, curves_f, agent = [], [], None
    span = 22 / max(agent_seeds, 1)
    for k in range(agent_seeds):
        sd = seed + 100 * k
        base = 72 + span * k
        ag, curve = R.train_agent(val_env, episodes=episodes, seed=sd, progress=lambda e, E, r: progress(
            f"Training the DQN agent (seed {k + 1} of {agent_seeds}): episode {e} of {E}, reward {r:.1f}",
            int(base + span * 0.7 * e / E)))
        progress(f"Training the agent without the GNN (seed {k + 1} of {agent_seeds})", int(base + span * 0.75))
        ag_f, curve_f = R.train_agent(val_env_f, episodes=episodes, seed=sd + 1)
        runs = {"agent": R.run_policy(test_env, "agent", vis, agent=ag),
                }
        per_seed["agent"].append(R.evaluate_run(runs["agent"], te))
        if k == 0:
            traces["agent"] = _trace_summary(runs["agent"])
        online = copy.deepcopy(ag)
        env_o = R.run_policy(test_env, "agent", vis, agent=online, online=True, eps=0.02)
        per_seed["online"].append(R.evaluate_run(env_o, te))
        if k == 0:
            traces["online"] = _trace_summary(env_o)
        env_f = R.run_policy(test_env_f, "agent", vis, agent=ag_f)
        per_seed["no_gnn"].append(R.evaluate_run(env_f, te))
        if k == 0:
            traces["no_gnn"] = _trace_summary(env_f)
            agent, agent_f = ag, ag_f
        curves.append(curve)
        curves_f.append(curve_f)
    for key, runs in per_seed.items():
        mean = {m: float(np.mean([r[m] for r in runs])) for m in runs[0]}
        sdv = {m: float(np.std([r[m] for r in runs])) for m in runs[0]}
        policies[key] = {"label": POLICY_LABELS[key], **mean, "sd": sdv, "seeds": len(runs),
                         "per_seed": [{m: r[m] for m in ("f1", "pr_auc", "recall_at_1pct", "reward", "reward_all", "tp", "fp")}
                                      for r in runs]}
    curve = [{"episode": i + 1, "reward": float(np.mean([c[i]["reward"] for c in curves])),
              "epsilon": curves[0][i]["epsilon"]} for i in range(episodes)]
    curve_f = [{"episode": i + 1, "reward": float(np.mean([c[i]["reward"] for c in curves_f])),
                "epsilon": curves_f[0][i]["epsilon"]} for i in range(episodes)]
    timings["agent"] = round(time.time() - t0, 1)

    usage = np.bincount(traces["agent"]["actions"], minlength=R.N_ACTIONS)
    action_usage = {
        "theta_offset": {str(d): int(sum(usage[i] for i, a in enumerate(R.ACTIONS) if a[0] == d)) for d in R.THETA_OFFSETS},
        "lp_weight": {str(w): int(sum(usage[i] for i, a in enumerate(R.ACTIONS) if a[1] == w)) for w in R.LP_WEIGHTS},
        "pseudo_conf": {str(c): int(sum(usage[i] for i, a in enumerate(R.ACTIONS) if a[2] == c)) for c in R.PL_CONF},
    }
    joblib.dump({"agent": agent, "agent_no_gnn": agent_f, "semi_gnn": stage["semi_gnn"]},
                out_dir / "rl_bundle.joblib", compress=3)
    timings["total"] = round(sum(v for k, v in timings.items() if k != "agent") + time.time() - t_start, 1)
    return {
        "dataset": {"kind": kind, "name": g.name, "rows": int(g.n), "frauds": int(g.y.sum()), "fraud_rate": float(g.y.mean()),
                    "features": len(g.feature_names), "graph": g.graph_stats,
                    "sizes": {"train": len(tr), "val": len(va), "test": len(te)},
                    "test_frauds": int(g.y[te].sum()), "span_days": float((g.t[-1] - g.t[0]) / 86400)},
        "params": {"label_rate": label_rate, "rounds": rounds, "episodes": episodes, "window_minutes": window_minutes,
                   "delay_windows": delay, "fp_cost": fp_cost, "agent_seeds": agent_seeds, "epochs": epochs, "max_rows": max_rows, "seed": seed,
                   "knn_k": knn_k if kind == "ulb" else None, "knn_min_sim": knn_min_sim if kind == "ulb" else None,
                   "fsmote_ratio": fsmote, "windows_val": len(w_val), "windows_test": len(w_test), "prior": prior},
        "labels": {"visible_train": int(vis[tr].sum()), "visible_train_frauds": int(g.y[tr][vis[tr]].sum()),
                   "visible_val": int(vis[va].sum()), "visible_val_frauds": int(g.y[va][vis[va]].sum()),
                   "visible_test": int(vis[te].sum()), "visible_test_frauds": int(g.y[te][vis[te]].sum())},
        "semi_supervised": {"history": semi["history"], "best_round": semi["best_round"], "report": semi_report},
        "static": static, "policies": policies, "traces": traces,
        "training": {"agent": curve, "no_gnn": curve_f, "static_val_reward": static["val_reward"],
                     "losses": [round(float(x), 5) for x in agent.losses[::max(1, len(agent.losses) // 300)]]},
        "action_usage": action_usage, "timings": timings,
    }
