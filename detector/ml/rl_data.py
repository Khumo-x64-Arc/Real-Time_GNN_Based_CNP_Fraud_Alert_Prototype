"""Datasets for the semi-supervised GNN + reinforcement-learning experiments.

Two public benchmarks, loaded into one shape (`GraphData`):

European Cardholder (ULB, Kaggle "creditcardfraud", creditcard.csv)
    284,807 transactions over two days, 492 frauds. Columns: Time (seconds since the first
    transaction), V1..V28 (PCA components), Amount, Class. There are no account, card or device
    columns, so the graph is built from similarity, as FraudGNN-RL (Cui et al., 2025, Sec. V-A4)
    does: each transaction is linked to earlier transactions in the past hour whose feature vectors
    have cosine similarity above 0.9 (at most k of them, the most similar first). Links only ever
    point backwards in time, so the graph is causal and full-graph scores are streaming scores.

IEEE-CIS (Kaggle "ieee-fraud-detection", train_transaction.csv + train_identity.csv)
    590,540 e-commerce transactions over about six months, 3.5% fraud. The graph uses entities:
    account (card1 + addr1 + P_emaildomain, the usual IEEE-CIS "uid"), card (card1-card5) and a
    device fingerprint (DeviceInfo + OS + browser + screen + id_19/id_20). Node features: amount,
    ProductCD, card network and type, C1-C14 counts, D-columns (time deltas, with missing flags),
    M1-M9 matches, distance, hour of day, device type, and the V block compressed to 16 principal
    components (fitted on the training period only).

Both loaders return transactions sorted by time with features, labels, timestamps and the graph,
plus the "label groups" the RL environment uses for label propagation: READ[i, g] = transaction i
can see labels held by group g, WRITE[j, g] = a label on transaction j is added to group g. For
entities a group is an entity (READ = WRITE = incidence). For the similarity graph a group is one
earlier transaction (READ = adjacency, WRITE = identity).
"""
from __future__ import annotations

import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA

from .gnn import build_incidence

SPLIT = (0.60, 0.75)   # train | validation | test, by time


class RLDataError(Exception):
    """Shown to the user as-is."""


@dataclass
class GraphData:
    name: str
    kind: str                      # "ulb" or "ieee"
    X: np.ndarray                  # node features (float32)
    y: np.ndarray                  # 0/1 labels (used for training only where visible; always for evaluation)
    t: np.ndarray                  # seconds, ascending
    amount: np.ndarray
    feature_names: list[str]
    entity_keys: dict[str, list] = field(default_factory=dict)
    relations: dict[str, sp.csr_matrix] = field(default_factory=dict)
    read: sp.csr_matrix | None = None
    write: sp.csr_matrix | None = None
    graph_stats: dict = field(default_factory=dict)
    tx_id: np.ndarray | None = None

    @property
    def n(self):
        return len(self.y)

    def split(self):
        n = self.n
        a, b = int(n * SPLIT[0]), int(n * SPLIT[1])
        return np.arange(a), np.arange(a, b), np.arange(b, n)


# ------------------------------------------------------------------ similarity graph (ULB)
def causal_knn(Z: np.ndarray, t: np.ndarray, k: int = 5, window: float = 3600.0, min_sim: float = 0.9,
               progress=None) -> tuple[sp.csr_matrix, dict]:
    """A[i, j] = 1 for up to k earlier transactions j with t_i - window <= t_j and cos(z_i, z_j) >= min_sim.

    Rows must be sorted by t. Works block by block (one block = one window of time), comparing each
    block with itself and the block before, so memory stays at a few block-sized matrices.
    """
    n = len(t)
    U = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    U = U.astype(np.float32)
    blk = np.floor((t - t[0]) / window).astype(np.int64)
    starts = np.searchsorted(blk, np.arange(blk.max() + 2))
    rows, cols, sims = [], [], []
    for b in range(blk.max() + 1):
        s, e = starts[b], starts[b + 1]
        if e <= s:
            continue
        c0 = starts[b - 1] if b > 0 else s
        S = U[s:e] @ U[c0:e].T                                  # block x (previous block + block)
        cand = np.arange(c0, e)
        me = np.arange(s, e)[:, None]
        valid = (cand[None, :] < me) & (t[cand][None, :] >= t[s:e][:, None] - window) & (S >= min_sim)
        S = np.where(valid, S, -np.inf)
        kk = min(k, S.shape[1])
        if kk == 0:
            continue
        top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        val = np.take_along_axis(S, top, 1)
        ok = np.isfinite(val)
        r = np.repeat(np.arange(s, e), kk).reshape(-1, kk)
        rows.append(r[ok]), cols.append(cand[top][ok]), sims.append(val[ok])
        if progress and b % 10 == 0:
            progress(b, blk.max() + 1)
    if rows:
        rows, cols = np.concatenate(rows), np.concatenate(cols)
    else:
        rows = cols = np.zeros(0, dtype=np.int64)
    A = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    deg = np.asarray(A.sum(1)).ravel()
    return A, {"edges": int(A.nnz), "mean_degree": float(deg.mean()), "isolated": float((deg == 0).mean()),
               "k": k, "window_s": window, "min_sim": min_sim}


def _groups(entity_keys: dict[str, list], relations: dict[str, sp.csr_matrix], n: int):
    """READ / WRITE matrices for label propagation (see the module docstring)."""
    reads, writes = [], []
    for keys in entity_keys.values():
        B, _ = build_incidence(keys)
        reads.append(B), writes.append(B)
    for A in relations.values():
        reads.append((A != 0).astype(np.float32)), writes.append(sp.identity(n, format="csr", dtype=np.float32))
    if not reads:
        z = sp.csr_matrix((n, 1))
        return z, z
    return sp.hstack(reads).tocsr().astype(np.float32), sp.hstack(writes).tocsr().astype(np.float32)


def _hour_feats(t):
    h = (np.asarray(t, float) / 3600.0) % 24
    return np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)


# ------------------------------------------------------------------ European cardholder (ULB)
ULB_V = [f"V{i}" for i in range(1, 29)]


def load_ulb(path, k: int = 5, window: float = 3600.0, min_sim: float = 0.9, max_rows: int | None = None,
             progress=lambda m, p: None) -> GraphData:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next((n for n in z.namelist() if n.lower().endswith(".csv")), None)
            if not name:
                raise RLDataError("No CSV inside the zip.")
            with z.open(name) as fh:
                df = pd.read_csv(fh, nrows=max_rows)
    else:
        df = pd.read_csv(path, nrows=max_rows)
    missing = [c for c in ["Time", "Amount", "Class"] + ULB_V if c not in df.columns]
    if missing:
        raise RLDataError("This doesn't look like the European Cardholder file (creditcard.csv): missing "
                          + ", ".join(missing[:6]) + ("…" if len(missing) > 6 else "") + ".")
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    t = df["Time"].to_numpy(float)
    amt = df["Amount"].to_numpy(float)
    hs, hc = _hour_feats(t)
    X = np.column_stack([df[ULB_V].to_numpy(float), np.log1p(np.maximum(amt, 0)), hs, hc]).astype(np.float32)
    names = ULB_V + ["log_amount", "hour_sin", "hour_cos"]
    y = df["Class"].astype(int).to_numpy()
    progress("Linking each transaction to similar ones from the past hour", 8)
    tr = int(len(df) * SPLIT[0])
    mu, sd = X[:tr].mean(0), X[:tr].std(0) + 1e-6   # standardise on the training period only
    A, stats = causal_knn((X - mu) / sd, t, k=k, window=window, min_sim=min_sim)
    rel = {"similar": A}
    R, W = _groups({}, rel, len(df))
    return GraphData("European Cardholder (ULB)", "ulb", X, y, t, amt, names, {}, rel, R, W,
                     {"relation": "similar", **stats}, np.arange(len(df)).astype(str))


# ------------------------------------------------------------------ IEEE-CIS
IEEE_C = [f"C{i}" for i in range(1, 15)]
IEEE_D = ["D1", "D2", "D3", "D4", "D5", "D10", "D11", "D15"]
IEEE_M = [f"M{i}" for i in range(1, 10)]
IEEE_BASE = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD", "card1", "card2", "card3",
             "card4", "card5", "card6", "addr1", "dist1", "P_emaildomain", "R_emaildomain"]
IEEE_ID = ["TransactionID", "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33", "id_19", "id_20"]
V_COMPONENTS = 16
V_MAX_MISSING = 0.15


def _join(df: pd.DataFrame, cols: list[str]) -> list:
    parts = [df[c] for c in cols]
    allna = np.logical_and.reduce([p.isna().to_numpy() for p in parts])
    s = parts[0].map(str)
    for p in parts[1:]:
        s = s + "|" + p.map(str)
    return s.astype(object).where(~allna, None).tolist()


def resolve_ieee(path) -> tuple[Path, Path | None]:
    """Accepts the Kaggle zip, the folder it unpacks to, or train_transaction.csv."""
    p = Path(path)
    if p.suffix.lower() == ".zip":
        dest = p.with_suffix("")
        with zipfile.ZipFile(p) as z:
            for want in ["train_transaction.csv", "train_identity.csv"]:
                hit = next((n for n in z.namelist() if Path(n).name.lower() == want), None)
                if hit and not (dest / want).exists():
                    dest.mkdir(parents=True, exist_ok=True)
                    with z.open(hit) as src, open(dest / want, "wb") as out:
                        while chunk := src.read(1 << 22):
                            out.write(chunk)
        p = dest
    folder = p if p.is_dir() else p.parent
    trans = p if p.is_file() else folder / "train_transaction.csv"
    ident = folder / "train_identity.csv"
    if not trans.exists():
        raise RLDataError("train_transaction.csv was not found in the IEEE-CIS input.")
    return trans, (ident if ident.exists() else None)


def load_ieee(path, max_rows: int | None = None, knn: bool = False, k: int = 5, window: float = 3600.0,
              min_sim: float = 0.9, progress=lambda m, p: None) -> GraphData:
    trans, ident = resolve_ieee(path)
    header = pd.read_csv(trans, nrows=0).columns.tolist()
    missing = [c for c in IEEE_BASE if c not in header]
    if missing:
        raise RLDataError("This doesn't look like IEEE-CIS train_transaction.csv: missing " + ", ".join(missing) + ".")
    vcols = [c for c in header if c.startswith("V") and c[1:].isdigit()]
    if vcols:   # keep the V columns that are mostly present (judged on the first 100,000 rows)
        sample = pd.read_csv(trans, usecols=vcols, nrows=min(100_000, max_rows or 100_000), dtype=np.float32)
        vcols = [c for c in vcols if sample[c].isna().mean() <= V_MAX_MISSING]
    use = IEEE_BASE + [c for c in IEEE_C + IEEE_D + IEEE_M if c in header] + vcols
    progress("Reading train_transaction.csv", 4)
    df = pd.read_csv(trans, usecols=use, nrows=max_rows, dtype={c: np.float32 for c in vcols})
    if ident is not None:
        id_head = pd.read_csv(ident, nrows=0).columns
        idf = pd.read_csv(ident, usecols=[c for c in IEEE_ID if c in id_head])
        df = df.merge(idf, on="TransactionID", how="left")
    for c in IEEE_ID[1:]:
        if c not in df.columns:
            df[c] = np.nan
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    n = len(df)
    tr = int(n * SPLIT[0])
    progress("Building IEEE-CIS node features", 8)

    cols, names = [], []

    def add(v, name):
        cols.append(np.asarray(v, dtype=np.float32))
        names.append(name)

    amt = df["TransactionAmt"].to_numpy(float)
    add(np.log1p(amt), "log_amount")
    add(np.round(amt - np.floor(amt), 3), "amount_cents")
    for col, levels in [("ProductCD", ["W", "C", "R", "H", "S"]), ("card4", ["visa", "mastercard", "american express", "discover"]),
                        ("card6", ["debit", "credit"]), ("DeviceType", ["desktop", "mobile"])]:
        v = df[col].astype(str).str.lower()
        for lv in levels:
            add((v == lv.lower()).astype(float), f"{col}={lv}")
    for c in IEEE_C:
        if c in df:
            add(np.log1p(np.maximum(df[c].fillna(0).to_numpy(float), 0)), c)
    for c in IEEE_D:
        if c in df:
            v = df[c].to_numpy(float)
            add(np.sign(np.nan_to_num(v)) * np.log1p(np.abs(np.nan_to_num(v))), c)
            add(np.isnan(v).astype(float), f"{c}_missing")
    for c in IEEE_M:
        if c not in df:
            continue
        if c == "M4":
            for lv in ["M0", "M1", "M2"]:
                add((df[c] == lv).astype(float), f"M4={lv}")
        else:
            add(df[c].map({"T": 1.0, "F": -1.0}).fillna(0).to_numpy(float), c)
    d1 = df["dist1"].to_numpy(float)
    add(np.log1p(np.nan_to_num(d1)), "log_dist1"), add(np.isnan(d1).astype(float), "dist1_missing")
    add(df["P_emaildomain"].isna().astype(float), "email_missing")
    add((df["P_emaildomain"].astype(str) == df["R_emaildomain"].astype(str)).astype(float), "email_match")
    hs, hc = _hour_feats(df["TransactionDT"].to_numpy(float))
    add(hs, "hour_sin"), add(hc, "hour_cos")
    v_info = {"v_columns_used": len(vcols), "v_components": 0, "v_variance_kept": 0.0}
    if vcols:
        V = np.nan_to_num(df[vcols].to_numpy(np.float32))      # missing V values count as 0
        V = np.sign(V) * np.log1p(np.abs(V))
        ncomp = min(V_COMPONENTS, len(vcols))
        pca = PCA(n_components=ncomp, random_state=0).fit(V[:tr])   # training period only
        P = pca.transform(V)
        for j in range(ncomp):
            add(P[:, j], f"V_pc{j + 1}")
        v_info.update(v_components=ncomp, v_variance_kept=float(pca.explained_variance_ratio_.sum()))
        del V
    X = np.column_stack(cols).astype(np.float32)

    keys = {
        "account": [None if v is None else f"account:{v}" for v in _join(df, ["card1", "addr1", "P_emaildomain"])],
        "card": [None if v is None else f"card:{v}" for v in _join(df, ["card1", "card2", "card3", "card5"])],
        "device": [f"device:{d}" if isinstance(d, str) and d.count("nan") < 3 else None
                   for d in _join(df, ["DeviceInfo", "id_30", "id_31", "id_33", "id_19", "id_20"])],
    }
    t = df["TransactionDT"].to_numpy(float)
    rel, stats = {}, {}
    if knn:
        progress("Linking each transaction to similar ones from the past hour", 10)
        mu, sd = X[:tr].mean(0), X[:tr].std(0) + 1e-6
        A, stats = causal_knn((X - mu) / sd, t, k=k, window=window, min_sim=min_sim)
        rel = {"similar": A}
    R, W = _groups(keys, rel, n)
    ent = {tname: {"entities": len(set(v for v in vals if v)), "linked": float(np.mean([v is not None for v in vals]))}
           for tname, vals in keys.items()}
    return GraphData("IEEE-CIS", "ieee", X, df["isFraud"].astype(int).to_numpy(), t, amt, names, keys, rel, R, W,
                     {"entities": ent, **({"relation": "similar", **stats} if knn else {}), **v_info,
                      "identity_file": ident is not None}, df["TransactionID"].astype(str).to_numpy())


def load(kind: str, path, **kw) -> GraphData:
    if kind == "ulb":
        return load_ulb(path, **{k: v for k, v in kw.items() if k in {"k", "window", "min_sim", "max_rows", "progress"}})
    if kind == "ieee":
        return load_ieee(path, **kw)
    raise RLDataError(f"Unknown dataset kind {kind!r}.")


# ------------------------------------------------------------------ synthetic look-alikes (for testing)
def synth_ulb(n: int = 60_000, fraud_rate: float = 0.004, seed: int = 7) -> pd.DataFrame:
    """A file with creditcard.csv's columns. Frauds come in campaigns: bursts of similar transactions
    a few minutes apart, around a campaign centre that shifts the usual ULB fraud components (V14, V12,
    V17, V10 down; V4, V11 up), plus some camouflaged frauds that sit close to legitimate ones."""
    rng = np.random.default_rng(seed)
    span = 172_792.0
    # day/night rhythm like the real file: fewer transactions between ~1am and 7am
    u = rng.uniform(0, 1, n * 3)
    tt = np.sort(u * span)
    hour = (tt / 3600) % 24
    keep = rng.uniform(0, 1, len(tt)) < np.where((hour > 1) & (hour < 7), 0.25, 1.0)
    tt = np.sort(rng.choice(tt[keep], n, replace=False))
    scale = np.linspace(2.0, 0.3, 28)
    V = rng.normal(0, 1, (n, 28)) * scale
    amount = np.round(np.exp(rng.normal(3.2, 1.3, n)), 2)
    y = np.zeros(n, dtype=int)
    n_fraud = max(10, int(n * fraud_rate))
    shift = np.zeros(28)
    for j, s in {13: -4.0, 11: -3.5, 16: -3.5, 9: -3.0, 3: 2.5, 10: 2.5}.items():
        shift[j] = s
    placed = 0
    while placed < n_fraud:
        size = int(rng.integers(2, 9))
        centre = shift * rng.uniform(0.35, 1.0) + rng.normal(0, 0.8, 28) * scale * 0.5
        start = rng.uniform(0, span - 3600)
        times = start + np.cumsum(rng.uniform(20, 400, size))
        camo = rng.uniform() < 0.3
        for tm in times:
            if placed >= n_fraud:
                break
            i = int(np.clip(np.searchsorted(tt, tm), 0, n - 1))
            if y[i]:
                continue
            y[i] = 1
            mix = 0.35 if camo else 1.0
            V[i] = (centre * mix + rng.normal(0, 0.35, 28) * scale * (1.5 if camo else 1.0))
            amount[i] = 1.0 if rng.uniform() < 0.3 else np.round(np.exp(rng.normal(4.5, 1.0)), 2)
            placed += 1
    df = pd.DataFrame(V, columns=ULB_V)
    df.insert(0, "Time", np.round(tt))
    df["Amount"] = amount
    df["Class"] = y
    return df


def synth_ieee(n: int = 80_000, fraud_rate: float = 0.035, days: int = 60, seed: int = 11,
               n_v: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train_transaction.csv / train_identity.csv look-alikes. Fraud: compromised accounts that switch to
    a new device and spend in bursts, and rings of accounts sharing a few mule device fingerprints."""
    rng = np.random.default_rng(seed)
    n_acc = max(200, n // 25)
    acc_card1 = rng.integers(1000, 18000, n_acc)
    acc_addr = rng.integers(100, 540, n_acc).astype(float)
    acc_mail = rng.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com", None], n_acc)
    acc_dev = rng.integers(0, n_acc * 2, n_acc)
    acc_type = rng.choice(["debit", "credit"], n_acc, p=[0.75, 0.25])
    acc_net = rng.choice(["visa", "mastercard", "american express", "discover"], n_acc, p=[0.65, 0.3, 0.03, 0.02])
    acc_age = rng.uniform(0, 600, n_acc)
    a = rng.integers(0, n_acc, n)
    t = np.sort(rng.uniform(86_400, 86_400 * (days + 1), n))
    y = np.zeros(n, dtype=int)
    dev = acc_dev[a].astype(object)
    target = int(n * fraud_rate)
    mules = [f"mule{m}" for m in range(max(3, n_acc // 300))]
    placed = 0
    while placed < target:
        acc = int(rng.integers(0, n_acc))
        rows = np.where(a == acc)[0]
        rows = rows[rows > n // 50]
        if len(rows) < 2:
            continue
        start = rng.choice(rows)
        burst = rows[(rows >= start) & (t[rows] <= t[start] + 86_400 * rng.uniform(0.2, 2.0))][: int(rng.integers(2, 10))]
        ring = rng.uniform() < 0.4
        new_dev = rng.choice(mules) if ring else f"new{acc}_{start}"
        y[burst] = 1
        dev[burst] = new_dev
        placed += len(burst)
    amt = np.round(np.exp(rng.normal(4.0, 1.0, n)) * np.where(y == 1, rng.uniform(1.0, 3.0, n), 1.0), 3)
    product = np.where(y == 1, rng.choice(["W", "C", "H"], n, p=[0.45, 0.45, 0.10]),
                       rng.choice(["W", "C", "R", "H", "S"], n, p=[0.74, 0.11, 0.07, 0.06, 0.02]))
    df = pd.DataFrame({
        "TransactionID": np.arange(2_987_000, 2_987_000 + n), "isFraud": y, "TransactionDT": np.round(t).astype(int),
        "TransactionAmt": amt, "ProductCD": product, "card1": acc_card1[a], "card2": rng.integers(100, 600, n_acc)[a].astype(float),
        "card3": 150.0, "card4": acc_net[a], "card5": rng.integers(100, 240, n_acc)[a].astype(float), "card6": acc_type[a],
        "addr1": acc_addr[a], "addr2": 87.0, "dist1": np.where(rng.uniform(0, 1, n) < 0.6, np.nan,
                                                              np.round(np.exp(rng.normal(2, 1.5, n)) * (1 + 3 * y))),
        "P_emaildomain": acc_mail[a], "R_emaildomain": np.where(rng.uniform(0, 1, n) < 0.7, None, acc_mail[a]),
    })
    order = np.argsort(a, kind="stable")
    cnt = np.zeros(n)
    for acc_rows in np.split(order, np.flatnonzero(np.diff(a[order])) + 1):
        acc_rows = np.sort(acc_rows)
        cnt[acc_rows] = np.arange(1, len(acc_rows) + 1)
    for i in range(1, 15):
        base = cnt * rng.uniform(0.2, 1.5) + rng.poisson(1, n)
        df[f"C{i}"] = np.round(base * (1 + y * rng.uniform(0.5, 3.0, n) * (i in (1, 2, 4, 8, 11))))
    df["D1"] = np.round(acc_age[a] + (t - t[0]) / 86_400) * np.where(y == 1, rng.uniform(0, 0.4, n), 1)
    for c in ["D2", "D3", "D4", "D5", "D10", "D11", "D15"]:
        v = np.round(rng.exponential(30, n) * np.where(y == 1, 0.3, 1.0))
        df[c] = np.where(rng.uniform(0, 1, n) < (0.5 if c in ("D5", "D11") else 0.2), np.nan, v)
    for i in range(1, 10):
        if i == 4:
            df["M4"] = rng.choice(["M0", "M1", "M2", None], n, p=[0.35, 0.1, 0.1, 0.45])
        else:
            p_t = np.where(y == 1, 0.35, 0.75)
            df[f"M{i}"] = np.where(rng.uniform(0, 1, n) < 0.4, None, np.where(rng.uniform(0, 1, n) < p_t, "T", "F"))
    lat = rng.normal(0, 1, (n, 6)) + y[:, None] * rng.normal(1.2, 0.5, 6)
    W = rng.normal(0, 1, (6, n_v))
    Vm = np.abs(lat @ W) + rng.exponential(0.5, (n, n_v))
    for j in range(n_v):
        col = np.round(Vm[:, j], 2)
        if j % 5 == 4:
            col = np.where(rng.uniform(0, 1, n) < 0.6, np.nan, col)
        df[f"V{j + 1}"] = col
    has_id = (rng.uniform(0, 1, n) < 0.25) | (y == 1) & (rng.uniform(0, 1, n) < 0.8)
    os_ = rng.choice(["Windows 10", "iOS 11.2.1", "Android 7.0", "Mac OS X 10_13_2"], n_acc * 2)
    br = rng.choice(["chrome 63.0", "mobile safari 11.0", "chrome 63.0 for android", "edge 16.0"], n_acc * 2)
    ident = pd.DataFrame({
        "TransactionID": df["TransactionID"][has_id].to_numpy(),
        "DeviceType": rng.choice(["desktop", "mobile"], int(has_id.sum())),
        "DeviceInfo": [f"dev-{d}" for d in dev[has_id]],
        "id_30": [os_[zlib.crc32(str(d).encode()) % len(os_)] for d in dev[has_id]],
        "id_31": [br[zlib.crc32(str(d).encode()) % len(br)] for d in dev[has_id]],
        "id_33": "1920x1080", "id_19": rng.integers(100, 600, int(has_id.sum())).astype(float),
        "id_20": rng.integers(100, 600, int(has_id.sum())).astype(float),
    })
    return df, ident
