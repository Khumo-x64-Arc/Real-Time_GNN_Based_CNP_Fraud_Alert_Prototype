"""Fraud rings: inject known rings into a synthetic South African card file, and score how the agent handled them.

Three ring patterns, six rings by default, placed across the training, validation and test periods of the
60/15/25 time split, so the agent learns from some rings and meets new ones of the same kinds in the test:

  shared_device_ato   compromised accounts operated from a few shared devices on one IP subnet, cashing out
                      at high-risk online merchants within hours
  card_testing        many cards tried with tiny amounts at one online merchant from one device/IP in under an
                      hour, then a few of the cards used for large purchases
  collusive_merchant  compromised accounts funnelling money to one complicit merchant (a new merchant ID that
                      also takes some legitimate "cover" sales), each account from its own fresh device

Existing rows keep their transaction IDs and values. New rows get new IDs and every derived field stays
consistent: reference, dates, per-account sequence number, seconds since the account's previous transaction and
the running account balance (capped so no balance goes negative). Ground truth goes in two new columns,
fraud_ring_id and fraud_pattern, which are for evaluation only: no model feature reads them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RING_COLUMNS = ("fraud_ring_id", "fraud_pattern")
PATTERN_LABELS = {"shared_device_ato": "Shared-device account takeover", "card_testing": "Card testing, then cash-out",
                  "collusive_merchant": "Collusive merchant"}
REQUIRED = ["transaction_id", "Time", "Amount", "Class", "account_id", "account_balance", "account_tx_seq",
            "seconds_since_prev_account_tx", "merchant_id", "merchant_name", "merchant_category", "channel",
            "device_id", "device_type", "ip_address", "tx_country", "tx_city", "posting_date", "transaction_date",
            "timestamp", "reference", "card_token"]
T0 = pd.Timestamp("2013-09-02 00:00:00")
DAY = 86400
CASHOUT = ["electronics", "gift_cards", "crypto_exchange", "money_transfer", "online_marketplace", "telco_airtime"]

# (ring, generator, preferred day, period it must fall in, keyword arguments)
DEFAULT_PLAN = [
    ("R01", "shared_device_ato", 3, "train", {"n_acc": 5, "n_dev": 2, "dtype": "web_desktop"}),
    ("R02", "card_testing", 7, "train", {"n_acc": 9, "n_cashout": 3}),
    ("R03", "collusive_merchant", 10, "train", {"n_acc": 6, "merchant_id": "MER950001",
                                                "merchant_name": "Umhlanga Digital Vouchers", "category": "gift_cards",
                                                "n_cover": 6}),
    ("R04", "shared_device_ato", 13, "validation", {"n_acc": 5, "n_dev": 2, "dtype": "android",
                                                    "country_mix": ("ZA", "NG", "GB")}),
    ("R05", "card_testing", 18, "test", {"n_acc": 8, "n_cashout": 3}),
    ("R06", "collusive_merchant", 20, "test", {"n_acc": 5, "merchant_id": "MER950002",
                                               "merchant_name": "Highveld Coin Exchange", "category": "crypto_exchange",
                                               "n_cover": 5, "shared_subnet": True}),
]


class RingError(ValueError):
    pass


def _period_days(d: pd.DataFrame):
    """Day boundaries of the 60/15/25 split by row count: (train end, validation end, last day)."""
    t = np.sort(d["Time"].to_numpy(float))
    n = len(t)
    return t[int(n * .6)] / DAY, t[int(n * .75)] / DAY, t[-1] / DAY


def _place(day, period, cuts, span):
    """Keep the preferred day when the ring's first day falls inside its period; otherwise move the ring (up to
    `span` days long) to the middle of that period, so the plan also works on files with a different time span."""
    lo, hi = {"train": (0, cuts[0]), "validation": (cuts[0], cuts[1]), "test": (cuts[1], cuts[2])}[period]
    if lo <= day and day + 1 <= hi:
        return day
    mid = int((lo + hi - span) / 2)
    if mid < lo or mid + span > hi + 1:
        raise RingError(f"The {period} period ({hi - lo:.1f} days) is too short for a ring.")
    return max(int(np.ceil(lo)), mid)


def inject_rings(d: pd.DataFrame, seed: int = 2026, plan=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(file with the rings added, one summary row per ring). `d` is the synthetic South African schema."""
    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        raise RingError("Not the synthetic South African schema: missing " + ", ".join(missing))
    if any(c in d.columns and d[c].fillna("").astype(str).str.len().gt(0).any() for c in RING_COLUMNS):
        raise RingError("This file already has fraud rings (fraud_ring_id is filled in).")
    plan = DEFAULT_PLAN if plan is None else plan
    rng = np.random.default_rng(seed)
    d = d.copy()
    d["fraud_ring_id"] = ""
    d["fraud_pattern"] = ""
    cuts = _period_days(d)

    fraud_accounts = set(d.loc[d["Class"] == 1, "account_id"])
    acc_stats = d.groupby("account_id").agg(n=("Time", "size"), min_bal=("account_balance", "min"))
    eligible = acc_stats[(acc_stats["n"] >= 5) & (acc_stats["min_bal"] > 4000)].index.difference(list(fraud_accounts))
    eligible = list(rng.permutation(sorted(eligible)))
    profile_cols = [c for c in ["account_number", "cardholder_name", "customer_ref", "issuing_bank",
                                "account_tenure_months", "home_city", "home_province", "card_token", "card_scheme",
                                "card_product", "card_last4"] if c in d.columns]
    profiles = d.drop_duplicates("account_id").set_index("account_id")[profile_cols]

    merchants = d.drop_duplicates("merchant_id")[["merchant_id", "merchant_name", "merchant_category"]]
    cash_merchants = merchants[merchants["merchant_category"].isin(CASHOUT)]
    if len(cash_merchants) == 0:
        raise RingError("No online cash-out merchants (electronics, gift cards, crypto, ...) in the file.")
    test_merchants = merchants[merchants["merchant_category"].isin(["streaming", "online_marketplace", "telco_airtime"])]
    if len(test_merchants) == 0:
        test_merchants = cash_merchants
    city_by_country = d.groupby("tx_country")["tx_city"].agg(lambda s: s.value_counts().index[:6].tolist()).to_dict()
    post_lag = (pd.to_datetime(d["posting_date"]) - pd.to_datetime(d["transaction_date"])).dt.days
    post_lag_p = post_lag.value_counts(normalize=True).sort_index()
    taken: set = set()
    rows: list = []
    spent: dict = {}
    by_acc = {a: g.sort_values("Time")[["Time", "account_balance", "Amount"]].to_numpy(float)
              for a, g in d.groupby("account_id")}

    def take_accounts(k):
        out = []
        for a in eligible:
            if a not in taken:
                out.append(a)
                taken.add(a)
            if len(out) == k:
                return out
        raise RingError("Not enough accounts with enough history and balance to build the rings.")

    def device_id():
        return "DEV" + "".join(rng.choice(list("0123456789abcdef"), 10))

    def subnet():
        return f"{rng.choice([41, 102, 105, 154, 196, 197])}.{rng.integers(1, 250)}.{rng.integers(0, 4)}"

    def ip_in(sn):
        return f"{sn}.{rng.integers(2, 254)}"

    def room(account, t):
        h = by_acc[account]
        later = h[h[:, 0] >= t, 1]
        low = later.min() if len(later) else h[-1, 1]
        return 0.85 * low - spent.get(account, 0.0)

    def add(account, t, amount, merchant, device, dtype, ip, ring, pattern, label=1, country="ZA", entry="ecom_no_3ds"):
        amount = min(amount, room(account, t))
        if amount < 1:
            return
        spent[account] = spent.get(account, 0.0) + amount
        p = profiles.loc[account]
        rows.append({**{c: p[c] for c in profiles.columns}, "account_id": account, "Time": float(int(t)),
                     "Amount": round(float(amount), 2), "merchant_id": merchant["merchant_id"],
                     "merchant_name": merchant["merchant_name"], "merchant_category": merchant["merchant_category"],
                     "channel": "ECOM", "entry_mode": entry, "terminal_id": np.nan, "device_id": device,
                     "device_type": dtype, "ip_address": ip, "tx_country": country,
                     "tx_city": rng.choice(city_by_country.get(country, city_by_country.get("ZA", ["Johannesburg"]))),
                     "Class": float(label), "fraud_ring_id": ring if label else "",
                     "fraud_pattern": pattern if label else ""})

    def pick(df):
        return df.iloc[int(rng.integers(len(df)))]

    def shared_device_ato(ring, day, n_acc, n_dev, dtype, country_mix=("ZA",)):
        accs, devs, sn = take_accounts(n_acc), [device_id() for _ in range(n_dev)], subnet()
        start = day * DAY + rng.uniform(8, 14) * 3600
        for k, a in enumerate(accs):
            t = start + k * rng.uniform(600, 2400)
            for j in range(int(rng.integers(2, 5))):
                dev = devs[(k + j) % n_dev]           # the operators switch between the ring's devices
                add(a, t, rng.uniform(900, 6500), pick(cash_merchants), dev, dtype, ip_in(sn), ring,
                    "shared_device_ato", country=rng.choice(country_mix),
                    entry=rng.choice(["ecom_no_3ds", "ecom_no_3ds", "ecom_3ds"]))
                t += rng.uniform(120, 1500)

    def card_testing(ring, day, n_acc, n_cashout):
        accs, dev, ip = take_accounts(n_acc), device_id(), ip_in(subnet())
        m_test = pick(test_merchants)
        t = day * DAY + rng.uniform(1, 4) * 3600      # small hours, when cardholders are asleep
        for a in accs:
            add(a, t, rng.uniform(1, 9.5), m_test, dev, "web_desktop", ip, ring, "card_testing")
            t += rng.uniform(40, 300)
        t += rng.uniform(1, 3) * 3600
        for a in accs[:n_cashout]:
            for _ in range(int(rng.integers(1, 3))):
                add(a, t, rng.uniform(2500, 7500), pick(cash_merchants), dev, "web_desktop", ip, ring, "card_testing")
                t += rng.uniform(300, 1200)

    def collusive_merchant(ring, day, n_acc, merchant_id, merchant_name, category, n_cover, shared_subnet=False):
        m = {"merchant_id": merchant_id, "merchant_name": merchant_name, "merchant_category": category}
        accs = take_accounts(n_acc)
        sn = subnet()
        for a in accs:
            t = day * DAY + rng.uniform(0, 2 * DAY)
            dev = device_id()
            ip = ip_in(sn) if shared_subnet else ip_in(subnet())
            for _ in range(int(rng.integers(1, 4))):
                add(a, t, rng.uniform(1500, 4990), m, dev, rng.choice(["android", "web_mobile", "web_desktop"]), ip,
                    ring, "collusive_merchant")
                t += rng.uniform(600, 5400)
        cover = d.loc[(d["channel"] == "ECOM") & d["device_id"].notna() & ~d["account_id"].isin(fraud_accounts | taken)]
        for _ in range(n_cover):   # legitimate sales by ordinary customers around the same days (Class 0)
            r = cover.iloc[int(rng.integers(len(cover)))]
            add(r["account_id"], max(0, day - 2) * DAY + rng.uniform(0, 6 * DAY), rng.uniform(80, 900), m,
                r["device_id"], r["device_type"], r["ip_address"], ring, "", label=0, entry="ecom_3ds")

    gen = {"shared_device_ato": (shared_device_ato, 1), "card_testing": (card_testing, 1),
           "collusive_merchant": (collusive_merchant, 3)}
    for ring, kind, day, period, kw in plan:
        fn, span = gen[kind]
        fn(ring, _place(day, period, cuts, span), **kw)

    out = assemble(d, rows, rng, post_lag_p)
    return out, ring_summary(out)


def posting_lag_distribution(d: pd.DataFrame) -> pd.Series:
    lag = (pd.to_datetime(d["posting_date"]) - pd.to_datetime(d["transaction_date"])).dt.days
    return lag.value_counts(normalize=True).sort_index()


def assemble(d: pd.DataFrame, rows: list, rng, post_lag_p: pd.Series) -> pd.DataFrame:
    """Add new transaction rows to the file: new IDs, dates and reference, then recompute the derived fields
    (sequence number, seconds since the previous transaction, running balance) of every account they touch."""
    new = pd.DataFrame(rows)
    ids = d["transaction_id"].astype(str).str.extract(r"(\d+)$")[0].astype(int)
    start_id = int(ids.max()) + 1
    new["transaction_id"] = [f"TX{start_id + i:08d}" for i in range(len(new))]
    ts = T0 + pd.to_timedelta(new["Time"], unit="s")
    new["timestamp"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    new["transaction_date"] = ts.dt.strftime("%Y-%m-%d")
    lag = rng.choice(post_lag_p.index.to_numpy(), size=len(new), p=post_lag_p.to_numpy())
    new["posting_date"] = (ts.dt.normalize() + pd.to_timedelta(lag, unit="D")).dt.strftime("%Y-%m-%d")
    idx = new["transaction_id"].str[2:].astype(int)
    new["reference"] = ((3245 + (new["Time"] // DAY).astype(int)) * 10**8
                        + ((new["Time"] % DAY) // 3600).astype(int) * 10**6 + idx)

    out = pd.concat([d, new[[c for c in d.columns if c in new.columns]]], ignore_index=True)
    out = out.sort_values(["Time", "transaction_id"], kind="stable").reset_index(drop=True)
    original = set(d["transaction_id"])
    for a in set(new["account_id"]):
        m = out["account_id"] == a
        sub = out.loc[m].sort_values(["Time", "transaction_id"], kind="stable")
        first = sub.iloc[0]
        opening = (first["account_balance"] + first["Amount"] if first["transaction_id"] in original
                   else d.loc[d["account_id"] == a].sort_values("Time").iloc[0][["account_balance", "Amount"]].sum())
        out.loc[sub.index, "account_tx_seq"] = np.arange(1, len(sub) + 1)
        out.loc[sub.index, "seconds_since_prev_account_tx"] = sub["Time"].diff().to_numpy()
        out.loc[sub.index, "account_balance"] = (opening - sub["Amount"].cumsum()).round(2).to_numpy()
    for c in ("account_tx_seq", "card_last4", "account_number", "reference"):
        if c in out:
            out[c] = out[c].astype("int64")
    out = out[list(d.columns)]
    return out


def ring_summary(out: pd.DataFrame) -> pd.DataFrame:
    """One row per ring: pattern, size, value, time span and the period of the 60/15/25 split it falls in."""
    rid = out["fraud_ring_id"].fillna("").astype(str)
    r = out[rid != ""]
    if r.empty:
        return pd.DataFrame()
    s = r.groupby("fraud_ring_id").agg(
        pattern=("fraud_pattern", "first"), transactions=("transaction_id", "size"), accounts=("account_id", "nunique"),
        devices=("device_id", "nunique"),
        ip_subnets=("ip_address", lambda x: x.astype(str).str.rsplit(".", n=1).str[0].nunique()),
        merchants=("merchant_id", "nunique"), amount_total=("Amount", "sum"),
        first=("timestamp", "min"), last=("timestamp", "max"), t_first=("Time", "min"))
    s["amount_total"] = s["amount_total"].round(2)
    cut = np.sort(out["Time"].to_numpy(float))
    n = len(cut)
    c0, c1 = cut[int(n * .6)], cut[int(n * .75)]
    s["period"] = ["train" if t < c0 else "validation" if t < c1 else "test" for t in s["t_first"]]
    return s.drop(columns="t_first")


# ------------------------------------------------------------------ scoring the agent
def _outcome(m: pd.DataFrame) -> str:
    in_ring = m["case_type"] == "ring"
    n_cases = m.loc[in_ring, "case"].nunique()
    if in_ring.sum() >= max(2, int(np.ceil(len(m) / 2))) and n_cases == 1:
        return "caught as one ring case"
    if n_cases > 1:
        return f"split across {n_cases} ring cases"
    if in_ring.any():
        return "partly caught as a ring"
    if m["escalated"].any():
        return "escalated, but not linked as a ring"
    return "missed"


def score_known_rings(decisions: pd.DataFrame, ring_rows: pd.DataFrame) -> list[dict]:
    """How the agent handled each known ring.

    decisions: the automation decision log (test period) with a `known_ring` column.
    ring_rows: every ring transaction in the file, with columns ring, pattern, period, account, revealed,
               is_fraud, transaction_id (training and validation rings report what the agent could learn from).
    """
    out = []
    d = decisions.copy()
    d["known_ring"] = d["known_ring"].fillna("").astype(str)
    th0 = d["threshold"] - d["threshold_offset"]
    d["gnn_flag"] = d["gnn_logit"] >= th0
    for ring, rr in ring_rows[ring_rows["is_fraud"] == 1].groupby("ring", sort=True):
        base = {"ring": ring, "pattern": rr["pattern"].iloc[0],
                "pattern_label": PATTERN_LABELS.get(rr["pattern"].iloc[0], rr["pattern"].iloc[0]),
                "period": rr["period"].iloc[0], "transactions": int(len(rr)), "accounts": int(rr["account"].nunique()),
                "revealed_labels": int(rr["revealed"].sum())}
        m = d[(d["known_ring"] == ring) & (d["is_fraud"] == 1)]
        if base["period"] != "test" or m.empty:
            out.append(base)
            continue
        cases = []
        for c in m.loc[m["case"].notna(), "case"].unique():
            cm = d[d["case"] == c]
            mine = cm["known_ring"] == ring
            cases.append({"case": c, "type": cm["case_type"].iloc[0], "size": int(len(cm)), "ring_members": int(mine.sum()),
                          "other_frauds": int((~mine & (cm["is_fraud"] == 1)).sum()),
                          "legit": int((cm["is_fraud"] == 0).sum()), "accounts": int(cm["account"].nunique())})
        rows = m.sort_values("t")[["transaction_id", "timestamp", "account", "device", "ip", "merchant", "merchant_name",
                                   "amount", "gnn_prob", "decision", "case", "case_type", "ring_linked"]]
        out.append({**base, "scored": True, "outcome": _outcome(m),
                    "gnn_alone": int(m["gnn_flag"].sum()), "escalated": int(m["escalated"].sum()),
                    "in_ring_case": int((m["case_type"] == "ring").sum()),
                    "in_burst": int((m["case_type"] == "account burst").sum()),
                    "escalated_alone": int((m["decision"] == "escalate").sum()),
                    "ring_linked": int(m["ring_linked"].notna().sum()),
                    "held_back": int((m["decision"] == "held_back").sum()),
                    "approved": int((m["decision"] == "approve").sum()),
                    "accounts_in_ring_case": int(m.loc[m["case_type"] == "ring", "account"].nunique()),
                    "cases": cases, "network": ring_network(rows),
                    "rows": rows.to_dict("records")})
    return out


# ------------------------------------------------------------------ drawing a ring
def _subnet(ip):
    s = str(ip)
    return s.rsplit(".", 1)[0] + ".*" if s.count(".") == 3 else s


def ring_network(rows: pd.DataFrame, width=320, height=230) -> dict:
    """Positions for a small SVG: accounts (coloured by what the agent did) joined to the devices, IP subnets
    and merchants they used. Merchants used by only one account are left out unless the ring has one merchant."""
    order = {"block_case": 3, "escalate": 2, "held_back": 1, "approve": 0}
    acc_dec, edges = {}, set()
    for r in rows.itertuples(index=False):
        a = ("account", r.account)
        acc_dec[r.account] = max(acc_dec.get(r.account, "approve"), r.decision, key=lambda x: order.get(x, 0))
        for kind, v in (("device", r.device), ("ip", _subnet(r.ip) if r.ip else None), ("merchant", r.merchant)):
            if v and str(v) != "nan":
                edges.add((a, (kind, str(v))))
    deg: dict = {}
    for _, b in edges:
        deg[b] = deg.get(b, 0) + 1
    one_merchant = rows["merchant"].nunique() == 1
    edges = {(a, b) for a, b in edges if b[0] != "merchant" or deg[b] >= 2 or one_merchant}
    nodes = sorted({x for e in edges for x in e} | {("account", a) for a in acc_dec})
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        pos = nx.spring_layout(G, seed=3, k=1.2 / max(1, len(nodes)) ** .5)
    except ImportError:
        ang = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
        pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(nodes, ang)}
    xy = np.array([pos[n] for n in nodes], float)
    lo, hi = xy.min(0), xy.max(0)
    span = np.where(hi - lo > 1e-9, hi - lo, 1)
    pad = 18
    sc = lambda p: (pad + (p[0] - lo[0]) / span[0] * (width - 2 * pad), pad + (p[1] - lo[1]) / span[1] * (height - 2 * pad))  # noqa: E731
    P = {n: sc(pos[n]) for n in nodes}
    return {"w": width, "h": height,
            "nodes": [{"kind": k, "id": v, "x": round(P[(k, v)][0], 1), "y": round(P[(k, v)][1], 1),
                       "decision": acc_dec.get(v) if k == "account" else None} for k, v in nodes],
            "edges": [{"x1": round(P[a][0], 1), "y1": round(P[a][1], 1), "x2": round(P[b][0], 1), "y2": round(P[b][1], 1)}
                      for a, b in sorted(edges)]}
