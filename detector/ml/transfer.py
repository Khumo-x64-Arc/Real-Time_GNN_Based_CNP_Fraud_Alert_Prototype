"""Cross-dataset (transfer) GNN: learn fraud on labelled sources, score an unlabelled file.

Sources
    IEEE-CIS (train_transaction + train_identity)   real isFraud labels
    Brazilian Card Holder (MiBolsillo.csv)          no labels -> rule-based pseudo-labels (weak)
Target
    any transaction CSV with no labels (e.g. the generated synthetic set)

The three files share almost no columns (IEEE-CIS's V/C/D/M features are anonymised and
exist nowhere else), so everything is first mapped onto one shared schema:

    tx_id, t (seconds), has_time, amount, account, card, device, merchant, country, balance

and described only by features that mean the same thing in every dataset: amount against
the account's own history, velocity, time since the last transaction, new device / merchant /
country for the account, how many accounts share the device, amount against the balance,
hour of day. Columns a dataset doesn't have become 0 plus a "has_" flag. Continuous features
are standardised within each dataset (its own mean and spread), so a R$50 coffee in Brazil and
a $50 purchase in the IEEE data both read as "a typical amount for this file". This per-domain
normalisation is what lets one model move between currencies and markets.

Graph: one disjoint graph per dataset (entity keys prefixed with the dataset name, so no
cross-dataset links), the same four entity types everywhere (account, card, device, merchant).
The GNN learns from IEEE-CIS labels and Brazilian pseudo-labels together; Brazil's share of the
loss is fixed (default 20%) so 4,955 weak labels neither vanish next to 590,540 real ones nor
dominate. Early stopping and the alert threshold use IEEE-CIS validation data (real labels) only.
FSMOTE oversamples each dataset's training frauds separately.

Applying to an unlabelled file builds that file's own graph and runs the trained weights over
it once (no training, no labels). A label column in the file, if any, is ignored; an answer key
can be supplied separately, purely to evaluate the scores afterwards.

Scores rank well across datasets but their scale does not carry over: the threshold tuned on
IEEE-CIS flags almost nothing in a new file. Without labels there is nothing to tune a new one on,
so a run flags a fixed share of the file instead (an alert budget, default 1%), the way a review
team works to its capacity. The carried-over threshold is still reported.
"""
from __future__ import annotations

import math
import time
from collections import Counter, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from .fsmote import FSMOTEPlan, FuzzySMOTE
from .gnn import TransactionGNN
from .schema import detect_schema
from .txgraph import build_graph, save_graph

ENTITY_TYPES = ["account", "card", "device", "merchant"]
CONTINUOUS = ["f_log_amt", "f_prior", "f_amt_ratio", "f_amt_z", "f_log_gap", "f_tx_24h", "f_dev_accounts",
              "f_bal_ratio"]
BINARY = ["f_first", "f_new_device", "f_new_merchant", "f_new_country", "f_foreign", "f_night",
          "f_hour_sin", "f_hour_cos", "f_has_device", "f_has_merchant", "f_has_country", "f_has_balance",
          "f_has_time"]
FEATURES = CONTINUOUS + BINARY   # everything computed (the raw values also explain alerts)
# What the model actually sees. Left out on purpose:
#  - the "has_" flags: each source lacks different fields, so they identify the dataset and let the
#    model learn "which file is this" instead of "is this fraud" (the target is the only file with
#    all of them set, so it would be extrapolating);
#  - accounts per device: an IEEE-CIS "device" is a browser/OS configuration shared by thousands of
#    legitimate users, while elsewhere it's a physical phone, so the signal points opposite ways.
MODEL_FEATURES = [f for f in FEATURES if not f.startswith("f_has_") and f != "f_dev_accounts"]
FEATURE_LABELS = {
    "f_log_amt": "Amount (log, within the file)", "f_prior": "Earlier transactions on the account",
    "f_amt_ratio": "Amount vs the account's average", "f_amt_z": "Amount z-score vs the account",
    "f_log_gap": "Time since the account's last transaction", "f_tx_24h": "Account transactions in 24 h",
    "f_dev_accounts": "Accounts seen on the device", "f_bal_ratio": "Amount as a share of the balance",
    "f_first": "First transaction on the account", "f_new_device": "Device new to the account",
    "f_new_merchant": "Merchant new to the account", "f_new_country": "Country new to the account",
    "f_foreign": "Outside the account's usual country", "f_night": "Between midnight and 5am",
    "f_hour_sin": "Hour of day (sine)", "f_hour_cos": "Hour of day (cosine)",
    "f_has_device": "Device known", "f_has_merchant": "Merchant known", "f_has_country": "Country known",
    "f_has_balance": "Balance known", "f_has_time": "Time of day known",
}
BRAZIL_COUNTRY_FIX = {"SP": "BR", "OH": "US"}   # state codes typed into the country column
SPLIT = (0.60, 0.75)


class TransferError(Exception):
    """Shown to the user as-is."""


# ------------------------------------------------------------------ loading: every source -> one schema
def _join(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """'a|b|c' per row; rows where every part is missing become None."""
    parts = [df[c] for c in cols]
    allna = np.logical_and.reduce([p.isna().to_numpy() for p in parts])
    s = parts[0].map(str)
    for p in parts[1:]:
        s = s + "|" + p.map(str)
    return s.astype(object).where(~allna, None)   # object dtype keeps None (a string column would turn it into NaN)


def _canonical(n: int) -> dict:
    return {"tx_id": None, "t": None, "has_time": np.ones(n, bool), "amount": None,
            **{k: [None] * n for k in ENTITY_TYPES}, "country": [None] * n, "balance": np.full(n, np.nan),
            "label": np.full(n, -1)}


def load_ieee(trans_path, ident_path=None) -> pd.DataFrame:
    cols = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "card1", "card2", "card3", "card5",
            "addr1", "P_emaildomain"]
    t = pd.read_csv(trans_path, usecols=cols)
    if ident_path:
        idc = ["TransactionID", "DeviceInfo", "id_30", "id_31", "id_33", "id_19", "id_20"]
        t = t.merge(pd.read_csv(ident_path, usecols=idc), on="TransactionID", how="left")
    else:
        for c in ["DeviceInfo", "id_30", "id_31", "id_33", "id_19", "id_20"]:
            t[c] = np.nan
    t = t.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    c = _canonical(len(t))
    c.update({
        "tx_id": t["TransactionID"].astype(str).to_numpy(), "t": t["TransactionDT"].astype(float).to_numpy(),
        "amount": t["TransactionAmt"].astype(float).to_numpy(),
        # the account ("uid") most IEEE-CIS work uses: card + billing address + email domain
        "account": _join(t, ["card1", "addr1", "P_emaildomain"]).tolist(),
        "card": _join(t, ["card1", "card2", "card3", "card5"]).tolist(),
        # a device fingerprint: DeviceInfo alone is too coarse ("Windows" covers 47,722 transactions);
        # a fingerprint missing most of its parts is a category, not a device, so it counts as unknown
        "device": [d if isinstance(d, str) and d.count("nan") < 3 else None
                   for d in _join(t, ["DeviceInfo", "id_30", "id_31", "id_33", "id_19", "id_20"]).tolist()],
        "label": t["isFraud"].astype(int).to_numpy(),
    })
    return pd.DataFrame(c)


def _br_number(s: pd.Series) -> pd.Series:
    """'1.784,13' -> 1784.13 (Brazilian thousands dot, decimal comma)."""
    return pd.to_numeric(s.astype(str).str.strip().str.replace(".", "", regex=False)
                         .str.replace(",", ".", regex=False), errors="coerce")


def load_brazil(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, sep=";", encoding="latin-1", dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    need = {"id", "data", "valor", "limite_disp", "grupo_estabelecimento", "cidade_estabelecimento",
            "pais_estabelecimento"}
    if not need <= set(raw.columns):
        raise TransferError(f"The Brazilian file is missing columns: {', '.join(sorted(need - set(raw.columns)))}.")
    for col in raw.columns:
        raw[col] = raw[col].str.strip()
    raw["_date"] = pd.to_datetime(raw["data"], format="%d.%m.%Y", errors="coerce")
    raw["_row"] = np.arange(len(raw))
    raw = raw.dropna(subset=["_date"]).sort_values(["_date", "_row"], kind="stable").reset_index(drop=True)
    # the id column was rounded by a spreadsheet (4,53E+11), so customers are rebuilt from their profile
    prof = ["id", "safra_abertura", "cidade", "idade", "sexo", "limite_total"]
    prof = [p for p in prof if p in raw.columns]
    cust = _join(raw, prof)
    cust_id = "C" + pd.Series(pd.factorize(cust)[0], index=raw.index).astype(str).str.zfill(3)
    country = raw["pais_estabelecimento"].str.upper().replace(BRAZIL_COUNTRY_FIX)
    c = _canonical(len(raw))
    secs = (raw["_date"] - raw["_date"].min()).dt.total_seconds().to_numpy()
    c.update({
        "tx_id": ("BR" + raw["_row"].astype(str)).to_numpy(), "t": secs, "has_time": np.zeros(len(raw), bool),
        "amount": _br_number(raw["valor"]).fillna(0).to_numpy(),
        "account": cust_id.tolist(),
        "merchant": _join(raw, ["grupo_estabelecimento", "cidade_estabelecimento"]).tolist(),
        "country": country.where(country.notna(), None).tolist(),
        "balance": _br_number(raw["limite_disp"]).to_numpy(),
    })
    raw["_customer"] = cust_id
    return pd.DataFrame(c), raw


def load_generic(path_or_df) -> tuple[pd.DataFrame, dict]:
    """Any transaction CSV, columns found by name (detector/ml/schema.py). Labels are never read."""
    df = pd.read_csv(path_or_df, low_memory=False) if not isinstance(path_or_df, pd.DataFrame) else path_or_df
    sch = detect_schema(df)
    info = {"columns": {k: v for k, v in sch.items() if v and k not in {"label_problem"}},
            "ignored_label": sch.get("label")}
    if not sch.get("amount"):
        raise TransferError("No amount column was found (expected a column such as amount or TransactionAmt).")
    t_col = sch.get("time")
    has_time = False
    if t_col:
        dt = pd.to_datetime(df[t_col], errors="coerce")
        if dt.notna().mean() > 0.9:
            secs = (dt - dt.min()).dt.total_seconds()
            has_time = bool((dt.dt.hour.fillna(0) + dt.dt.minute.fillna(0)).ne(0).any())
        else:
            secs = pd.to_numeric(df[t_col], errors="coerce")
            has_time = True
        secs = secs.fillna(secs.median() if secs.notna().any() else 0)
    else:
        secs = pd.Series(np.arange(len(df), dtype=float))
    df = df.assign(_t=secs.to_numpy()).sort_values("_t", kind="stable").reset_index(drop=True)

    def col(role):
        name = sch.get(role)
        if not name:
            return [None] * len(df)
        s = df[name]
        return s.astype(object).where(s.notna(), None).map(lambda v: None if v is None else str(v)).tolist()

    c = _canonical(len(df))
    c.update({
        "tx_id": (df[sch["tx_id"]].astype(str) if sch.get("tx_id") else pd.Series(np.arange(len(df))).astype(str)).to_numpy(),
        "t": df["_t"].astype(float).to_numpy(), "has_time": np.full(len(df), has_time),
        "amount": pd.to_numeric(df[sch["amount"]], errors="coerce").fillna(0).to_numpy(),
        "account": col("account"), "card": col("card"), "device": col("device"), "merchant": col("merchant"),
        "country": col("country"),
        "balance": pd.to_numeric(df[sch["balance"]], errors="coerce").to_numpy() if sch.get("balance")
        else np.full(len(df), np.nan),
    })
    out = pd.DataFrame(c)
    out["time_text"] = df[t_col].astype(str).to_numpy() if t_col else ""   # shown as the file wrote it
    return out, info


# ------------------------------------------------------------------ features (the same meaning in every dataset)
def shared_features(c: pd.DataFrame) -> pd.DataFrame:
    """One pass in time order; each row only sees the account's past. Standardised within the dataset."""
    n = len(c)
    out = {f: [0.0] * n for f in FEATURES}   # plain lists: far cheaper per element than numpy in a Python loop
    acc: dict = {}
    dev_acc: dict = {}
    amounts = c["amount"].astype(float).tolist()
    ts = c["t"].astype(float).tolist()
    bal = c["balance"].astype(float).tolist()
    has_time = c["has_time"].astype(bool).tolist()
    accounts, devices, merchants, countries = (_clean(c[k].tolist()) for k in ("account", "device", "merchant", "country"))
    for i in range(n):
        a, d, m, ctry, amt, t = accounts[i], devices[i], merchants[i], countries[i], amounts[i], ts[i]
        st = acc.get(a) if a is not None else None
        if st is None:
            st = {"n": 0, "s": 0.0, "ss": 0.0, "last": None, "dev": set(), "mer": set(), "ctry": Counter(),
                  "recent": deque()}
            if a is not None:
                acc[a] = st
        k = st["n"]
        out["f_log_amt"][i] = math.log1p(max(amt, 0))
        out["f_prior"][i] = math.log1p(k)
        out["f_first"][i] = 1.0 if k == 0 else 0.0
        if k:
            mean = st["s"] / k
            var = max(st["ss"] / k - mean * mean, 0)
            out["f_amt_ratio"][i] = min(5.0, max(-5.0, math.log((max(amt, 0) + 1) / (max(mean, 0) + 1))))
            out["f_amt_z"][i] = min(10.0, max(-10.0, (amt - mean) / math.sqrt(var))) if k >= 2 and var > 0 else 0.0
            out["f_log_gap"][i] = math.log1p(max(t - st["last"], 0))
        rq = st["recent"]
        while rq and t - rq[0] > 86400:
            rq.popleft()
        out["f_tx_24h"][i] = math.log1p(len(rq))
        if d is not None:
            out["f_has_device"][i] = 1
            out["f_new_device"][i] = 1.0 if k and d not in st["dev"] else 0.0
            pool = dev_acc.setdefault(d, set())
            out["f_dev_accounts"][i] = math.log1p(len(pool) - (a in pool))   # O(1); a set difference copied the pool
            if a is not None:
                pool.add(a)
        if m is not None:
            out["f_has_merchant"][i] = 1
            out["f_new_merchant"][i] = 1.0 if k and m not in st["mer"] else 0.0
        if ctry is not None:
            out["f_has_country"][i] = 1
            out["f_new_country"][i] = 1.0 if k and ctry not in st["ctry"] else 0.0
            if st["ctry"]:
                out["f_foreign"][i] = 1.0 if ctry != st["ctry"].most_common(1)[0][0] else 0.0
        if bal[i] == bal[i] and bal[i] > 0:   # not NaN
            out["f_has_balance"][i] = 1
            out["f_bal_ratio"][i] = min(5.0, max(0.0, amt / bal[i]))
        if has_time[i]:
            out["f_has_time"][i] = 1
            hour = (t / 3600.0) % 24
            out["f_hour_sin"][i], out["f_hour_cos"][i] = math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)
            out["f_night"][i] = 1.0 if hour < 5 else 0.0
        st["n"] += 1
        st["s"] += amt
        st["ss"] += amt * amt
        st["last"] = t
        rq.append(t)
        if d is not None:
            st["dev"].add(d)
        if m is not None:
            st["mer"].add(m)
        if ctry is not None:
            st["ctry"][ctry] += 1
    f = pd.DataFrame(out)
    raw = f.copy()
    for col in CONTINUOUS:   # standardised within this dataset: the step that lets the model travel
        mu, sd = f[col].mean(), f[col].std()
        f[col] = (f[col] - mu) / sd if sd and sd > 0 else 0.0
    f.attrs["raw"] = raw
    return f


def _clean(values) -> list:
    """pandas stores a missing key as NaN (a float) even when None was put in; both mean "no entity"."""
    return [None if v is None or (isinstance(v, float) and v != v) else v for v in values]


def entity_keys(c: pd.DataFrame, prefix: str) -> dict[str, list]:
    return {t: [None if v is None else f"{prefix}|{t}:{v}" for v in _clean(c[t].tolist())] for t in ENTITY_TYPES}


# ------------------------------------------------------------------ pseudo-labels for the Brazilian data
BRAZIL_RULES = [
    ("foreign", 2, "Merchant outside Brazil"),
    ("cnp", 1, "Mail or telephone order (card not present)"),
    ("spike", 1, "At least 5× the customer's median amount and R$300 or more"),
    ("drain", 1, "Half or more of the available limit"),
    ("burst", 1, "4 or more transactions by the customer that day"),
]
BRAZIL_CUTOFF = 3


def pseudo_label_brazil(c: pd.DataFrame, raw: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Rule points per transaction; 3 or more = pseudo-fraud. Weak, partly circular labels (see the README)."""
    amt = c["amount"].to_numpy(float)
    hits = {}
    hits["foreign"] = np.array([x is not None and x != "BR" for x in c["country"]])
    hits["cnp"] = raw["grupo_estabelecimento"].str.upper().str.contains("M.O.T.O", regex=False).to_numpy()
    med = raw.assign(_a=amt).groupby("_customer")["_a"].transform("median").to_numpy()
    hits["spike"] = (amt >= 5 * med) & (amt >= 300)
    balance = c["balance"].to_numpy(float)
    hits["drain"] = np.where(balance > 0, amt >= 0.5 * balance, False)
    per_day = raw.groupby(["_customer", "_date"])["_row"].transform("count").to_numpy()
    hits["burst"] = per_day >= 4
    points = sum(w * hits[k].astype(int) for k, w, _ in BRAZIL_RULES)
    y = (points >= BRAZIL_CUTOFF).astype(int)
    stats = {"cutoff": BRAZIL_CUTOFF, "positives": int(y.sum()), "rows": int(len(y)), "rate": float(y.mean()),
             "rules": [{"key": k, "points": w, "label": lab, "fired": int(hits[k].sum()),
                        "in_positives": int((hits[k] & (y == 1)).sum())} for k, w, lab in BRAZIL_RULES]}
    return y, stats


# ------------------------------------------------------------------ metrics helpers
def _best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5


def _thin(x, y, n=120):
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).astype(int))
    return [[round(float(x[i]), 4), round(float(y[i]), 4)] for i in idx]


def evaluate(y, p, thr) -> dict:
    """Metrics against labels (real, pseudo or an answer key). Needs both classes present."""
    y = np.asarray(y).astype(int)
    out = {"n": int(len(y)), "positives": int(y.sum())}
    if y.min() == y.max():
        out["note"] = "Only one class present, so ranking metrics can't be computed."
        return out
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    k = int(y.sum())
    top = np.argsort(-p)[:k]
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    order = np.argsort(-p, kind="stable")
    caught = np.cumsum(y[order]) / max(k, 1)
    fracs = [0, .001, .002, .005, .01, .02, .03, .05, .075, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    gains = [[f, round(float(caught[max(int(round(f * len(y))) - 1, 0)]) if f else 0.0, 4)] for f in fracs]
    out.update({"gains": gains, "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
                "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision_at_k": float(y[top].mean()) if k else 0.0, "k": k,
                "base_rate": float(y.mean()), "roc_curve": _thin(fpr, tpr), "pr_curve": _thin(rec[::-1], prec[::-1])})
    return out


def _hist(p, bins=20):
    return np.histogram(np.clip(p, 0, 1), np.linspace(0, 1, bins + 1))[0].tolist()


LOG_EDGES = np.logspace(-8, 0, 25)


def _log_hist(p):
    """Counts per score band on a log scale: transfer scores can sit far below 0.01 and still rank well."""
    return np.histogram(np.clip(p, 1e-8, 1), LOG_EDGES)[0].tolist()


# ------------------------------------------------------------------ training
def _split(n):
    a, b = int(n * SPLIT[0]), int(n * SPLIT[1])
    idx = np.arange(n)
    return idx[:a], idx[a:b], idx[b:]


def _merge_plans(plans: list[tuple[FSMOTEPlan, str]]) -> tuple[FSMOTEPlan, np.ndarray, dict]:
    seeds, nbrs, lams, doms, stats = [], [], [], [], {}
    for plan, dom in plans:
        seeds.append(plan.seed)
        nbrs.append(plan.nbr)
        lams.append(plan.lam)
        doms += [dom] * len(plan)
        stats[dom] = {k: v for k, v in plan.stats.items() if k != "membership_hist"}
    merged = FSMOTEPlan(np.concatenate(seeds), np.concatenate(nbrs), np.concatenate(lams), stats)
    return merged, np.array(doms), stats


def train_transfer(ieee_transaction, ieee_identity, brazil_path, out_dir, progress=lambda m, p: None,
                   brazil_share: float = 0.2, fsmote_ratio: float = 1.0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    progress("Reading IEEE-CIS", 3)
    ie = load_ieee(ieee_transaction, ieee_identity)
    progress("Reading the Brazilian card holder file", 8)
    br, br_raw = load_brazil(brazil_path)
    y_br, rule_stats = pseudo_label_brazil(br, br_raw)
    br["label"] = y_br
    if rule_stats["positives"] < 5:
        raise TransferError("The Brazilian rules marked fewer than 5 transactions, too few to learn from.")

    progress(f"Building shared features for {len(ie):,} IEEE-CIS transactions", 12)
    f_ie = shared_features(ie)
    progress("Building shared features for the Brazilian transactions", 30)
    f_br = shared_features(br)
    X = np.vstack([f_ie[MODEL_FEATURES].to_numpy(float), f_br[MODEL_FEATURES].to_numpy(float)])
    n_ie, n_br = len(ie), len(br)
    y = np.concatenate([ie["label"].to_numpy(int), y_br])
    k_ie, k_br = entity_keys(ie, "ieee"), entity_keys(br, "br")
    keys = {t: k_ie[t] + k_br[t] for t in ENTITY_TYPES}

    ie_tr, ie_va, ie_te = _split(n_ie)
    br_tr, br_va, br_te = (s + n_ie for s in _split(n_br))
    train_idx = np.concatenate([ie_tr, br_tr])

    progress("Oversampling each dataset's training frauds with FSMOTE", 34)
    if fsmote_ratio > 0:
        fs = FuzzySMOTE(ratio=fsmote_ratio)
        plan, syn_dom, fs_stats = _merge_plans([(fs.plan(X, y, ie_tr), "ieee"), (fs.plan(X, y, br_tr), "br")])
    else:
        plan, syn_dom, fs_stats = FSMOTEPlan.empty(), np.array([]), {"method": "off"}

    # Brazil's share of the loss is fixed: weight its nodes so they carry `brazil_share` of the total
    w_br = (brazil_share / (1 - brazil_share)) * (len(ie_tr) / max(len(br_tr), 1))
    sample_weight = np.concatenate([np.ones(len(ie_tr)), np.full(len(br_tr), w_br)])
    syn_n = {d: int((syn_dom == d).sum()) for d in ("ieee", "br")}
    w_br_syn = (brazil_share / (1 - brazil_share)) * (syn_n["ieee"] / syn_n["br"]) if syn_n["br"] and syn_n["ieee"] else w_br
    syn_weight = np.where(syn_dom == "br", w_br_syn, 1.0) if len(syn_dom) else None

    progress("Training the GNN on IEEE-CIS + Brazil", 40)
    t0 = time.time()
    gnn = TransactionGNN(epochs=600, lr=0.015).fit(
        X, keys, y, train_idx, ie_va, synth=plan if len(plan) else None,
        sample_weight=sample_weight, syn_weight=syn_weight,
        progress=lambda ep, ap: progress(f"Training the GNN (epoch {ep}, IEEE-CIS validation PR-AUC {ap:.3f})",
                                         min(88, 40 + ep // 12)))
    train_seconds = time.time() - t0
    p = gnn.predict_graph()
    thr = _best_threshold(y[ie_va], p[ie_va])   # alert threshold from real labels only

    progress("Evaluating on held-out data", 90)
    results = {
        "threshold": thr, "features": MODEL_FEATURES,
        "feature_labels": {k: v for k, v in FEATURE_LABELS.items() if k in MODEL_FEATURES}, "entity_types": ENTITY_TYPES,
        "brazil_share": brazil_share, "fsmote": fs_stats, "pseudo_labels": rule_stats,
        "history": gnn.history, "train_seconds": round(train_seconds, 1),
        "total_seconds": round(time.time() - t_start, 1),
        "sources": {
            "ieee": {"name": "IEEE-CIS", "rows": n_ie, "frauds": int(ie["label"].sum()),
                     "fraud_rate": float(ie["label"].mean()), "labels": "real (isFraud)",
                     "accounts": int(pd.Series(ie["account"]).nunique()),
                     "devices": int(pd.Series(ie["device"]).nunique()),
                     "split": {"train": len(ie_tr), "val": len(ie_va), "test": len(ie_te)},
                     "test": evaluate(y[ie_te], p[ie_te], thr), "val": evaluate(y[ie_va], p[ie_va], thr),
                     "hist_fraud": _hist(p[ie_te][y[ie_te] == 1]), "hist_legit": _hist(p[ie_te][y[ie_te] == 0])},
            "br": {"name": "Brazilian Card Holder", "rows": n_br, "frauds": int(y_br.sum()),
                   "fraud_rate": float(y_br.mean()), "labels": "rule-based pseudo-labels",
                   "accounts": int(pd.Series(br["account"]).nunique()),
                   "merchants": int(pd.Series(br["merchant"]).nunique()),
                   "split": {"train": len(br_tr), "val": len(br_va), "test": len(br_te)},
                   "test": evaluate(y[br_te], p[br_te], thr)},
        },
    }
    gnn._Z0 = gnn._aggs = None   # the training graph isn't needed to score new files
    joblib.dump({"gnn": gnn, "threshold": thr, "features": MODEL_FEATURES, "entity_types": ENTITY_TYPES,
                 "trained_on": ["IEEE-CIS", "Brazilian Card Holder"]}, out_dir / "transfer.joblib", compress=3)
    progress("Done", 100)
    return results


# ------------------------------------------------------------------ applying to an unlabelled file
REASONS = [
    ("f_new_device", "Device new to the account"), ("f_foreign", "Outside the account's usual country"),
    ("f_new_country", "Country new to the account"), ("f_new_merchant", "Merchant new to the account"),
    ("f_night", "Between midnight and 5am"),
]


def _reasons(raw: pd.DataFrame, i: int) -> list[str]:
    r = raw.iloc[i]
    out = [lab for k, lab in REASONS if r[k] > 0.5]
    if r["f_tx_24h"] >= math.log1p(3):
        out.append(f"{int(round(math.expm1(r['f_tx_24h'])))} earlier transactions on the account in 24 h")
    if r["f_amt_ratio"] >= math.log(3):
        out.append(f"About {math.exp(r['f_amt_ratio']):.0f}× the account's usual amount")
    if r["f_dev_accounts"] >= math.log1p(2):
        out.append(f"Device already used by {int(round(math.expm1(r['f_dev_accounts'])))} other accounts")
    if r["f_bal_ratio"] >= 0.5:
        out.append(f"{r['f_bal_ratio'] * 100:.0f}% of the balance")
    if r["f_log_gap"] and r["f_first"] < 0.5 and r["f_log_gap"] < math.log1p(120):
        out.append("Under 2 minutes since the account's last transaction")
    return out or ["No single standout signal; the score comes from the combination and the graph"]


def apply_transfer(bundle_path, csv_path, out_dir, answer_key=None, progress=lambda m, p: None,
                   alert_rate: float = 0.01) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    b = joblib.load(bundle_path)
    gnn, carried = b["gnn"], b["threshold"]

    progress("Reading the unlabelled file", 5)
    c, info = load_generic(csv_path)
    progress(f"Building shared features for {len(c):,} transactions", 15)
    f = shared_features(c)
    raw = f.attrs["raw"]
    keys = entity_keys(c, "target")
    progress("Scoring with the trained GNN", 55)
    p = gnn.predict_new(f[b["features"]].to_numpy(float), keys)
    # alert budget: the top `alert_rate` share of this file (rank-based, so ties can't inflate it)
    n_flag = max(1, int(math.ceil(len(p) * alert_rate)))
    order = np.argsort(-p, kind="stable")
    flagged = np.zeros(len(p), bool)
    flagged[order[:n_flag]] = True
    thr = float(p[order[n_flag - 1]])

    progress("Collecting the flagged transactions", 70)
    top = []
    for i in order[: min(200, int(flagged.sum()) or 25)]:
        top.append({"tx_id": c["tx_id"].iat[i], "account": c["account"].iat[i], "amount": float(c["amount"].iat[i]),
                    "device": c["device"].iat[i], "country": c["country"].iat[i], "score": float(p[i]),
                    "flagged": bool(flagged[i]), "reasons": _reasons(raw, i)})
    # full precision: transfer scores are often tiny, and rounding would turn the ranking into ties
    pd.DataFrame({"transaction_id": c["tx_id"], "gnn_score": p, "rank": np.argsort(order) + 1,
                  "flagged": flagged.astype(int)}).to_csv(out_dir / "scores.csv", index=False, float_format="%.6g")

    results = {"rows": int(len(c)), "threshold": float(thr), "flagged": int(flagged.sum()),
               "alert_rate": float(alert_rate), "carried_threshold": float(carried),
               "carried_flagged": int((p >= carried).sum()),
               "score_quantiles": {q: float(np.quantile(p, float(q))) for q in ("0.5", "0.9", "0.99", "0.999")},
               "flag_rate": float(flagged.mean()), "hist": _hist(p), "log_hist": _log_hist(p), "top": top, "file": info,
               "present": {t: int(pd.Series(c[t]).notna().sum()) for t in ENTITY_TYPES + ["country"]},
               "has_time": bool(c["has_time"].any()), "has_balance": bool(np.isfinite(c["balance"]).any())}

    if answer_key is not None:   # used only now, after scoring, to measure how well the transfer worked
        progress("Comparing with the answer key", 80)
        key = pd.read_csv(answer_key)
        sch = detect_schema(key)
        id_col, lab_col = sch.get("tx_id") or key.columns[0], sch.get("label")
        if not lab_col:
            raise TransferError("The answer key needs a 0/1 label column such as is_fraud.")
        m = pd.DataFrame({"tx_id": c["tx_id"].astype(str), "p": p}).merge(
            key.assign(**{id_col: key[id_col].astype(str)}), left_on="tx_id", right_on=id_col, how="inner")
        yk = pd.to_numeric(m[lab_col], errors="coerce").fillna(0).astype(int).to_numpy()
        pk = m["p"].to_numpy()
        ev = evaluate(yk, pk, thr)
        ev["carried"] = {"flagged": int((pk >= carried).sum()), "caught": int(((pk >= carried) & (yk == 1)).sum())}
        ev["matched"] = int(len(m))
        ev["hist_fraud"], ev["hist_legit"] = _hist(m["p"].to_numpy()[yk == 1]), _hist(m["p"].to_numpy()[yk == 0])
        if "pattern" in key.columns:
            pat = m["pattern"].fillna("").astype(str)
            ev["patterns"] = [{"pattern": name, "frauds": int(((pat == name) & (yk == 1)).sum()),
                               "caught": int(((pat == name) & (yk == 1) & (pk >= thr)).sum()),
                               "roc_auc": float(roc_auc_score(yk[(pat == name).to_numpy() | (yk == 0)],
                                                              pk[(pat == name).to_numpy() | (yk == 0)]))}
                              for name in sorted(set(pat[yk == 1]) - {""})]
        results["evaluation"] = ev

    progress("Laying out the transaction graph", 88)
    meta, blob = build_graph({t: keys[t] for t in ENTITY_TYPES}, flagged.astype(int),
                             np.full(len(c), 2, dtype=np.uint8), p, thr)
    tx_info = pd.DataFrame({"id": c["tx_id"].astype(str), "amount": c["amount"], "time": c["time_text"]})
    save_graph(out_dir, meta, blob, tx_info)
    results["graph"] = {k: v for k, v in meta.items() if k not in ("names", "arrays", "bounds", "rings")}
    results["graph"]["rings"] = [{k: v for k, v in r.items() if k not in ("tx", "ents")} for r in meta["rings"][:10]]
    progress("Done", 100)
    return results
