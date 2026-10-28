"""Adversarial fraud: transactions built to evade the GNN, planted in a synthetic South African card file.

An attacker who knows (or can probe) what the detector looks at changes the fraud so it looks ordinary. The
model sees, per transaction: the amount and how it compares with the account's history, the time since the
account's last transaction and its 24-hour velocity, the share of the balance spent, whether the device,
merchant or country is new to the account or foreign, the hour of day, and the graph through the account, card,
device and merchant. Every attack edits some of these; the fraud itself (money leaving a victim's account
without consent) stays the same.

Groups, each on its own victim accounts:
  C0  naive              control: an account-takeover burst at night, new device, often abroad, large amounts
                         at cash-out merchants the victim never used
  A1  full mimicry       every edit: the victim's own device, home country, usual hours, typical amount, a
                         merchant the victim already uses, one transaction a day
  A2  low and slow       small amounts at normal hours, one a day, at home; still a new device and new merchants
  A3  graph camouflage   the naive fraud routed through the busiest legitimate device and merchant in the file,
                         so the GNN's neighbourhood around it is overwhelmingly legitimate
  A4  optimised          white-box query attack: starting from the naive fraud, add the edit that lowers the
                         GNN's detection most, one at a time, until the GNN flags none of the group

Edits (with an attacker cost, used only to break ties in the optimised attack):
  country   transact from the victim's home country (VPN/residential proxy)   cost 1
  hour      at the victim's usual hours of day                               cost 1
  amount    within the victim's normal range (its 90th percentile, +-15%)    cost 2
  merchant  a merchant the victim already uses                              cost 2
  velocity  one transaction a day instead of a burst                         cost 3
  device    the victim's own device (remote access / SIM-swap + app)         cost 4

Ground truth goes in two evaluation-only columns, adversarial_group and adversarial_attack. No model reads them.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .fraud_rings import CASHOUT, DAY, REQUIRED, RingError, _period_days, assemble, posting_lag_distribution

ADV_COLUMNS = ("adversarial_group", "adversarial_attack")
EDITS = ["country", "hour", "amount", "merchant", "velocity", "device"]
EDIT_COST = {"country": 1, "hour": 1, "amount": 2, "merchant": 2, "velocity": 3, "device": 4}
EDIT_LABELS = {"country": "home country", "hour": "usual hours", "amount": "typical amount",
               "merchant": "known merchant", "velocity": "one a day", "device": "victim's own device"}
GROUPS = {
    "C0": ("naive", ()),
    "A1": ("full_mimicry", tuple(EDITS)),
    "A2": ("low_and_slow", ("amount", "velocity", "hour", "country")),
    "A3": ("graph_camouflage", ()),
    "A4": ("optimised", ()),          # edits set by optimise_attack
}
GROUP_LABELS = {"C0": "Naive takeover (control)", "A1": "Full mimicry", "A2": "Low and slow",
                "A3": "Graph camouflage", "A4": "Optimised (white-box)"}


class AdversarialError(ValueError):
    pass


class _Context:
    """Everything about the clean file that the attack generator reads, computed once."""

    def __init__(self, d: pd.DataFrame, seed: int, victims_per_group: int, periods):
        missing = [c for c in REQUIRED if c not in d.columns]
        if missing:
            raise AdversarialError("Not the synthetic South African schema: missing " + ", ".join(missing))
        if "adversarial_group" in d.columns and d["adversarial_group"].fillna("").astype(str).str.len().gt(0).any():
            raise AdversarialError("This file already has adversarial examples.")
        self.d = d.sort_values(["Time", "transaction_id"], kind="stable").reset_index(drop=True)
        self.cuts = _period_days(self.d)
        rng = np.random.default_rng(seed)
        self.post_lag_p = posting_lag_distribution(self.d)
        d = self.d
        bad = set(d.loc[d["Class"] == 1, "account_id"])
        if "fraud_ring_id" in d:
            bad |= set(d.loc[d["fraud_ring_id"].fillna("").astype(str) != "", "account_id"])
        ecom = d[(d["channel"] == "ECOM") & d["device_id"].notna()]
        stats = d.groupby("account_id").agg(n=("Time", "size"), min_bal=("account_balance", "min"))
        has_dev = set(ecom["account_id"])
        pool = [a for a in stats[(stats["n"] >= 8) & (stats["min_bal"] > 3000)].index if a in has_dev and a not in bad]
        pool = list(rng.permutation(sorted(pool)))
        need = victims_per_group * len(GROUPS) * len(periods)
        if len(pool) < need:
            raise AdversarialError(f"Only {len(pool)} suitable victim accounts; {need} needed.")
        self.victims = {}
        k = 0
        for p in periods:
            for gname in GROUPS:
                self.victims[(p, gname)] = pool[k:k + victims_per_group]
                k += victims_per_group
        self.by_acc = {a: g for a, g in d.groupby("account_id")}
        merchants = d.drop_duplicates("merchant_id").set_index("merchant_id")[["merchant_name", "merchant_category"]]
        self.merchants = merchants
        self.cash = merchants[merchants["merchant_category"].isin(CASHOUT)]
        if self.cash.empty:
            raise AdversarialError("No online cash-out merchants in the file.")
        legit = d[d["Class"] == 0]
        dev_counts = legit[legit["device_id"].notna()].groupby("device_id").agg(acc=("account_id", "nunique"), n=("Time", "size"))
        # graph camouflage routes through the busiest legitimate device and merchant; training-period attacks
        # use the second busiest, so adversarial training cannot just memorise the test hubs
        self.hubs = {}
        le = legit[legit["channel"] == "ECOM"]
        top_dev = dev_counts.sort_values(["acc", "n"], ascending=False).index[:2]
        top_mer = le["merchant_id"].value_counts().index[:2]
        for k, p in enumerate(("test", "train")):
            dev = top_dev[min(k, len(top_dev) - 1)]
            hub_rows = legit[legit["device_id"] == dev]
            ip = hub_rows["ip_address"].mode().iloc[0] if hub_rows["ip_address"].notna().any() else None
            self.hubs[p] = {"device": dev, "ip": ip, "dtype": hub_rows["device_type"].mode().iloc[0],
                            "merchant": top_mer[min(k, len(top_mer) - 1)]}
        self.profile_cols = [c for c in ["account_number", "cardholder_name", "customer_ref", "issuing_bank",
                                         "account_tenure_months", "home_city", "home_province", "card_token",
                                         "card_scheme", "card_product", "card_last4"] if c in d.columns]
        self.profiles = d.drop_duplicates("account_id").set_index("account_id")[self.profile_cols]
        self.city = d.groupby("tx_country")["tx_city"].agg(lambda s: s.value_counts().index[:6].tolist()).to_dict()
        self.seed = seed

    def window(self, period):
        c0, c1, end = self.cuts
        return {"train": (1.0, c0 - 3.5), "test": (c1 + 0.3, end - 3.2)}[period]


def _group_rows(ctx: _Context, period: str, gname: str, edits, camouflage=False) -> list[dict]:
    """The transactions of one attack group. Every random draw comes from a stream fixed by (seed, period,
    group), drawn in the same order whatever the edits, so two edit sets differ only in what they edit."""
    rng = np.random.default_rng([ctx.seed, {"train": 1, "test": 2}[period], list(GROUPS).index(gname)])
    lo, hi = ctx.window(period)
    attack = GROUPS[gname][0]
    edits = set(edits)
    rows, spent = [], {}
    for a in ctx.victims[(period, gname)]:
        t0 = rng.uniform(lo, hi) * DAY
        night = rng.uniform(0.5, 4.5) * 3600
        n_tx = int(rng.integers(2, 4))
        u = rng.random((n_tx, 10))                    # fixed per-transaction draws
        h = ctx.by_acc[a]
        past = h[h["Time"] < t0]
        if len(past) < 3:
            past = h.iloc[:max(3, len(h) // 2)]
        ecom_past = past[(past["channel"] == "ECOM") & past["device_id"].notna()]
        own = ecom_past if len(ecom_past) else h[(h["channel"] == "ECOM") & h["device_id"].notna()]
        own_device = own["device_id"].mode().iloc[0]
        own_ip = own.loc[own["device_id"] == own_device, "ip_address"].mode()
        own_ip = own_ip.iloc[0] if len(own_ip) else None
        own_dtype = own.loc[own["device_id"] == own_device, "device_type"].mode().iloc[0]
        home = past["tx_country"].mode().iloc[0]
        hours = ((past["Time"] % DAY) / 3600).to_numpy()
        big_amt = float(past["Amount"].quantile(0.9))   # the top of the victim's normal range
        used = past["merchant_id"].value_counts().index.tolist()
        new_dev = "DEV" + "".join(np.array(list("0123456789abcdef"))[rng.integers(0, 16, 10)])
        new_ip = f"{rng.choice([41, 102, 105, 154, 196, 197])}.{rng.integers(1, 250)}.{rng.integers(0, 4)}.{rng.integers(2, 254)}"
        abroad = ["NG", "GB", "US", "ZA"][int(rng.integers(0, 4))]
        later = h.loc[h["Time"] >= np.floor(t0 / DAY) * DAY, "account_balance"]
        floor = float(later.min()) if len(later) else float(h["account_balance"].iloc[-1])
        t = t0
        for j in range(n_tx):
            uj = u[j]
            # time
            if "velocity" in edits:
                day0 = np.floor(t0 / DAY) + j
            else:
                day0 = np.floor(t0 / DAY)
            if "hour" in edits and len(hours):
                hour = hours[int(uj[0] * len(hours))] * 3600 + uj[1] * 1800
            else:
                hour = night + j * (300 + uj[1] * 1500) if "velocity" not in edits else night + uj[1] * 1800
            t = day0 * DAY + hour
            # amount
            if "amount" in edits:
                amt = big_amt * (0.85 + 0.3 * uj[2])
            else:
                amt = 2000 + 4000 * uj[2]
            amt = min(amt, 0.85 * floor - spent.get(a, 0.0))
            if amt < 1:
                continue
            spent[a] = spent.get(a, 0.0) + amt
            # merchant, device, country
            hub = ctx.hubs[period]
            if camouflage:
                mid = hub["merchant"]
            elif "merchant" in edits and used:
                mid = used[min(int(uj[3] * min(3, len(used))), len(used) - 1)]
            else:
                choices = ctx.cash.index[~ctx.cash.index.isin(used)]
                choices = choices if len(choices) else ctx.cash.index
                mid = choices[int(uj[3] * len(choices))]
            if camouflage:
                dev, ip, dtype = hub["device"], hub["ip"], hub["dtype"]
            elif "device" in edits:
                dev, ip, dtype = own_device, own_ip, own_dtype
            else:
                dev, ip, dtype = new_dev, new_ip, "web_desktop"
            country = home if "country" in edits else abroad
            m = ctx.merchants.loc[mid]
            p = ctx.profiles.loc[a]
            cities = ctx.city.get(country) or ctx.city.get("ZA") or ["Johannesburg"]
            rows.append({**{c: p[c] for c in ctx.profile_cols}, "account_id": a, "Time": float(int(t)),
                         "Amount": round(float(amt), 2), "merchant_id": mid, "merchant_name": m["merchant_name"],
                         "merchant_category": m["merchant_category"], "channel": "ECOM",
                         "entry_mode": "ecom_3ds" if {"device", "merchant"} <= edits else "ecom_no_3ds",
                         "terminal_id": np.nan, "device_id": dev, "device_type": dtype, "ip_address": ip,
                         "tx_country": country, "tx_city": cities[int(uj[4] * len(cities))], "Class": 1.0,
                         "adversarial_group": gname, "adversarial_attack": attack,
                         "adversarial_period": period})
    return rows


def inject_adversarial(d: pd.DataFrame, seed: int = 7, victims_per_group: int = 8, periods=("test",),
                       optimised_edits=(), ctx: _Context | None = None, train_edits=None) -> pd.DataFrame:
    """The file with every attack group added in each period in `periods` ('test', and 'train' for adversarial
    training). The optimised group A4 uses `optimised_edits` (find them with optimise_attack); in the training
    period it uses `train_edits` when given (the attack the defender trained on, not the one it now faces)."""
    ctx = ctx or _Context(d, seed, victims_per_group, periods)
    rows = []
    for p in periods:
        for gname, (_, edits) in GROUPS.items():
            e = edits
            if gname == "A4":
                e = train_edits if (p == "train" and train_edits is not None) else optimised_edits
            rows += _group_rows(ctx, p, gname, e, camouflage=gname == "A3")
    base = ctx.d.copy()
    for c in ADV_COLUMNS + ("adversarial_period",):
        base[c] = ""
    rng = np.random.default_rng(seed + 1)
    try:
        return assemble(base, rows, rng, ctx.post_lag_p)
    except RingError as e:
        raise AdversarialError(str(e))


# ------------------------------------------------------------------ the optimised (white-box) attack
def _scores_for(df: pd.DataFrame, model, theta0: float):
    from . import transfer_rl as T
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    c, raw = T.load_sa(buf)
    g = T.target_graph(c, T.features(c))
    z, _ = T.full_scores(model, g)
    grp = raw["adversarial_group"].fillna("").astype(str).to_numpy()
    per = raw["adversarial_period"].fillna("").astype(str).to_numpy()
    return z, grp, per, g


def optimise_attack(d: pd.DataFrame, model, theta0: float, seed=7, victims_per_group=8, periods=("test",),
                    progress=print, train_edits=None) -> dict:
    """Greedy white-box attack on group A4 in the test period: at each step add the edit that most lowers the
    share of A4 transactions the GNN flags at its threshold (ties: lower mean logit, then cheaper edit), until
    none is flagged or no edit helps. `model` is the defender's GNN, queried as a black box for scores."""
    ctx = _Context(d, seed, victims_per_group, periods)

    def evaluate(edits):
        df = inject_adversarial(d, seed, victims_per_group, periods, optimised_edits=edits, ctx=ctx,
                                train_edits=train_edits)
        z, grp, per, _ = _scores_for(df, model, theta0)
        m = (grp == "A4") & (per == "test")
        return {"edits": list(edits), "flagged": float((z[m] >= theta0).mean()), "mean_logit": float(z[m].mean()),
                "n": int(m.sum()), "cost": int(sum(EDIT_COST[e] for e in edits))}

    cur = evaluate(())
    path = [cur]
    progress(f"A4 start: {cur['flagged']:.0%} flagged, mean logit {cur['mean_logit']:.2f}")
    while cur["flagged"] > 0 and len(cur["edits"]) < len(EDITS):
        trials = []
        for e in EDITS:
            if e in cur["edits"]:
                continue
            r = evaluate(tuple(cur["edits"]) + (e,))
            trials.append(r)
            progress(f"  try +{e}: {r['flagged']:.0%} flagged, mean logit {r['mean_logit']:.2f}")
        best = min(trials, key=lambda r: (r["flagged"], r["mean_logit"], r["cost"]))
        if best["flagged"] >= cur["flagged"] and best["mean_logit"] >= cur["mean_logit"]:
            break
        cur = best
        path.append({**cur, "added": cur["edits"][-1], "trials": trials})
        progress(f"A4 + {cur['edits'][-1]}: {cur['flagged']:.0%} flagged, mean logit {cur['mean_logit']:.2f}")
    return {"edits": cur["edits"], "path": path, "evaded": cur["flagged"] == 0}


# ------------------------------------------------------------------ scoring
def score_groups(decisions: pd.DataFrame, theta0: float) -> list[dict]:
    """Per attack group in the test period: what the GNN alone and the agent did with its transactions.
    decisions: the automation decision log with an adversarial_group column."""
    d = decisions.copy()
    d["adversarial_group"] = d["adversarial_group"].fillna("").astype(str)
    out = []
    for gname in GROUPS:
        m = d[(d["adversarial_group"] == gname) & (d["is_fraud"] == 1)]
        if m.empty:
            continue
        n = len(m)
        out.append({"group": gname, "label": GROUP_LABELS[gname], "transactions": n,
                    "accounts": int(m["account"].nunique()),
                    "gnn_alone": int((m["gnn_logit"] >= theta0).sum()),
                    "escalated": int(m["escalated"].sum()),
                    "in_case": int(m["case"].notna().sum()),
                    "held_back": int((m["decision"] == "held_back").sum()),
                    "approved": int((m["decision"] == "approve").sum()),
                    "median_prob": float(m["gnn_prob"].median()),
                    "gnn_recall": float((m["gnn_logit"] >= theta0).mean()),
                    "agent_recall": float(m["escalated"].mean()),
                    "value": round(float(m["amount"].sum()), 2),
                    "value_escalated": round(float(m.loc[m["escalated"].astype(bool), "amount"].sum()), 2),
                    "value_per_tx": round(float(m["amount"].mean()), 2)})
    return out


def adversarial_summary(decisions: pd.DataFrame, theta0: float, theta_offsets) -> dict:
    """What run_automation stores for a file with planted adversarial examples."""
    d = decisions
    grp = d["adversarial_group"].fillna("").astype(str)
    native = d[(d["is_fraud"] == 1) & (grp == "")]
    off = np.asarray(theta_offsets, float)
    return {"groups": score_groups(d, theta0), "theta0": float(theta0),
            "native_frauds": {"n": int(len(native)), "gnn_alone": int((native["gnn_logit"] >= theta0).sum()),
                              "escalated": int(native["escalated"].sum())},
            "windows_threshold_lowered": float((off < -1e-2).mean()) if len(off) else 0.0,   # thetas are rounded
            "false_alarms": int((d["escalated"].astype(bool) & (d["is_fraud"] == 0)).sum())}


# ------------------------------------------------------------------ end to end: build the attacked file
def _defender(df: pd.DataFrame, label_rate=0.2, mask_seed=1):
    """The GNN the attacker probes: the target-only GNN on the same revealed labels the run will use."""
    from . import transfer_rl as T
    from .fraud_automation import base_gnns
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    c, _ = T.load_sa(buf)
    g = T.target_graph(c, T.features(c))
    b = base_gnns(g, T.visible_mask(g.n, label_rate, mask_seed), {}, progress=lambda m: None)["target_only"]
    return b["model"], float(b["theta0"]), float(b["val_pr_auc"])


def _trim(path: dict, against: str) -> dict:
    steps = [{"edits": s["edits"], "added": s.get("added"), "flagged": s["flagged"], "mean_logit": s["mean_logit"],
              "n": s["n"], "cost": s["cost"],
              "trials": [{"edit": t["edits"][-1], "flagged": t["flagged"], "mean_logit": t["mean_logit"]}
                         for t in s.get("trials", [])]} for s in path["path"]]
    return {"against": against, "edits": path["edits"], "evaded": path["evaded"], "steps": steps}


def build_adversarial_file(d: pd.DataFrame, seed=7, victims_per_group=8, adversarial_training=False,
                           label_rate=0.2, mask_seed=1, progress=print) -> tuple[pd.DataFrame, dict]:
    """The attacked file, ready to run, and how the optimised attack was found.

    Without adversarial training: all five groups in the test period; A4's edits come from the greedy attack on
    the GNN trained on the file. With adversarial training: the groups are also planted in the training period
    (A4 there with the edits found first, i.e. the attack the defender has seen), and A4 in the test period is
    re-optimised against the GNN trained on that file (an adaptive attacker)."""
    if d is None or len(d) == 0:
        raise AdversarialError("The file is empty.")
    progress("Planting the attack groups in the test period")
    f0 = inject_adversarial(d, seed, victims_per_group, ("test",))
    progress("Training the GNN the attacker probes")
    model, theta0, ap = _defender(f0, label_rate, mask_seed)
    progress("White-box attack: searching edits that make the GNN miss group A4")
    first = optimise_attack(d, model, theta0, seed, victims_per_group, ("test",), progress=lambda m: None)
    paths = [_trim(first, "the GNN trained without adversarial examples")]
    if not adversarial_training:
        out = inject_adversarial(d, seed, victims_per_group, ("test",), optimised_edits=tuple(first["edits"]))
        return out, {"adversarial_training": False, "edits_test": first["edits"], "edits_train": None,
                     "paths": paths, "groups": group_table(out)}
    periods = ("train", "test")
    progress("Adversarial training: planting the same attacks in the training period")
    f1 = inject_adversarial(d, seed, victims_per_group, periods, train_edits=tuple(first["edits"]))
    model2, theta2, _ = _defender(f1, label_rate, mask_seed)
    progress("Adaptive attacker: re-optimising group A4 against the adversarially trained GNN")
    second = optimise_attack(d, model2, theta2, seed, victims_per_group, periods, progress=lambda m: None,
                             train_edits=tuple(first["edits"]))
    paths.append(_trim(second, "the adversarially trained GNN"))
    out = inject_adversarial(d, seed, victims_per_group, periods, optimised_edits=tuple(second["edits"]),
                             train_edits=tuple(first["edits"]))
    return out, {"adversarial_training": True, "edits_test": second["edits"], "edits_train": first["edits"],
                 "paths": paths, "groups": group_table(out)}


def group_table(out: pd.DataFrame) -> list[dict]:
    a = out[out["adversarial_group"].fillna("").astype(str) != ""]
    rows = []
    for (p, g), m in a.groupby(["adversarial_period", "adversarial_group"], sort=True):
        rows.append({"period": p, "group": g, "label": GROUP_LABELS[g], "attack": m["adversarial_attack"].iloc[0],
                     "transactions": int(len(m)), "accounts": int(m["account_id"].nunique()),
                     "value": round(float(m["Amount"].sum()), 2),
                     "foreign": round(float((m["tx_country"] != "ZA").mean()), 3)})
    return rows
