"""Generate an unlabelled synthetic card-transaction file plus a separate answer key.

The transaction file has no label column, so a model scoring it learns nothing from labels.
The answer key (transaction_id, is_fraud, pattern) is written to its own file and is only
read after scoring, to measure how well a model trained elsewhere transfers.

The data is deliberately unlike the training sources: South African rand, its own merchants,
devices and IP addresses, and fraud of four kinds:

    account_takeover  a burst of larger purchases from a new device, often abroad, at night,
                      draining a good part of the balance
    fraud_ring        several accounts' cards used from the same few mule devices and IPs
    card_testing      a run of tiny amounts at new merchants, then one large purchase
    bust_out          an account that slowly ramps up, then spends far above its habit

Everything is seeded, so the same arguments always give the same files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PATTERNS = ["account_takeover", "fraud_ring", "card_testing", "bust_out"]


def generate_unlabelled(rows: int = 50_000, accounts: int = 3_000, fraud_rate: float = 0.012,
                        seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    acc_ids = [f"ZA{100000 + i}" for i in range(accounts)]
    home_dev = {a: f"DV{rng.integers(10**7):07d}" for a in acc_ids}
    home_ip = {a: f"196.{rng.integers(1, 255)}.{rng.integers(255)}.{rng.integers(255)}" for a in acc_ids}
    typical = {a: float(rng.lognormal(5.6, 0.7)) for a in acc_ids}          # ~R270 typical spend
    balance = {a: float(typical[a] * rng.uniform(15, 60)) for a in acc_ids}
    merchants = [f"MZ{i:04d}" for i in range(800)]
    categories = ["grocery", "fuel", "restaurant", "clothing", "electronics", "online", "pharmacy", "travel"]
    m_cat = {m: categories[rng.integers(len(categories))] for m in merchants}
    ring_devs = [f"MULE{i:02d}" for i in range(4)]
    ring_ips = [f"41.{rng.integers(100, 200)}.{rng.integers(255)}.7" for _ in range(4)]

    out, t = [], pd.Timestamp("2026-03-01 06:00")
    target_frauds = int(rows * fraud_rate)
    n_fraud = 0

    def tx(acc, amount, dev, ip, mer, country, channel, fraud, pattern):
        nonlocal t
        bal = balance[acc]
        balance[acc] = max(bal - amount, 0) + (typical[acc] * 30 if rng.random() < 0.01 else 0)  # salary top-ups
        out.append({"transaction_id": f"Z{len(out):07d}", "timestamp": t, "account_id": acc,
                    "card_token": f"K{acc[2:]}", "device_id": dev, "ip_address": ip, "merchant_id": mer,
                    "merchant_category": m_cat[mer], "tx_country": country, "channel": channel,
                    "amount": round(float(amount), 2), "account_balance": round(bal, 2),
                    "_fraud": int(fraud), "_pattern": pattern})

    while len(out) < rows:
        # daytime-heavy arrivals
        hour = t.hour
        t += pd.Timedelta(seconds=float(rng.exponential(55 if 7 <= hour <= 21 else 240)))
        acc = acc_ids[rng.integers(accounts)]
        want_fraud = n_fraud < target_frauds and rng.random() < fraud_rate / 2.2
        if want_fraud:
            kind = PATTERNS[rng.integers(len(PATTERNS))]
            if kind == "account_takeover":
                dev, ip = f"DV{rng.integers(10**7):07d}", f"102.{rng.integers(255)}.{rng.integers(255)}.{rng.integers(255)}"
                country = rng.choice(["ZA", "NG", "GB", "AE"], p=[0.4, 0.3, 0.15, 0.15])
                if rng.random() < 0.6:
                    t = t.normalize() + pd.Timedelta(hours=float(rng.uniform(0, 4.5)), days=1) if t.hour > 4 else t
                for _ in range(rng.integers(3, 7)):
                    t += pd.Timedelta(seconds=float(rng.exponential(150)))
                    mer = merchants[rng.integers(40)]   # a few online merchants
                    tx(acc, balance[acc] * rng.uniform(0.15, 0.4) + rng.uniform(200, 2000), dev, ip, mer,
                       country, "ECOM", True, kind)
                    n_fraud += 1
            elif kind == "fraud_ring":
                k = rng.integers(len(ring_devs))
                for _ in range(rng.integers(2, 5)):
                    t += pd.Timedelta(seconds=float(rng.exponential(600)))
                    tx(acc, typical[acc] * rng.uniform(1.5, 4), ring_devs[k], ring_ips[k],
                       merchants[rng.integers(800)], "ZA", "ECOM", True, kind)
                    n_fraud += 1
            elif kind == "card_testing":
                dev = f"DV{rng.integers(10**7):07d}"
                for _ in range(rng.integers(4, 9)):
                    t += pd.Timedelta(seconds=float(rng.exponential(25)))
                    tx(acc, rng.uniform(1, 9), dev, home_ip[acc], merchants[rng.integers(800)], "ZA", "ECOM", True, kind)
                    n_fraud += 1
                t += pd.Timedelta(seconds=float(rng.exponential(120)))
                tx(acc, rng.uniform(2500, 9000), dev, home_ip[acc], merchants[rng.integers(40)], "ZA", "ECOM", True, kind)
                n_fraud += 1
            else:  # bust_out
                for _ in range(rng.integers(2, 4)):
                    t += pd.Timedelta(seconds=float(rng.exponential(1800)))
                    tx(acc, typical[acc] * rng.uniform(6, 15), home_dev[acc], home_ip[acc],
                       merchants[rng.integers(40, 120)], "ZA", rng.choice(["ECOM", "POS"]), True, kind)
                    n_fraud += 1
            continue
        dev = home_dev[acc] if rng.random() < 0.94 else f"DV{rng.integers(10**7):07d}"
        ip = home_ip[acc] if rng.random() < 0.9 else f"105.{rng.integers(255)}.{rng.integers(255)}.{rng.integers(255)}"
        country = "ZA" if rng.random() < 0.985 else rng.choice(["NA", "BW", "MZ", "GB"])
        mer = merchants[rng.integers(800)]
        channel = "ECOM" if m_cat[mer] == "online" or rng.random() < 0.25 else "POS"
        tx(acc, rng.lognormal(np.log(typical[acc]), 0.55), dev, ip, mer, country, channel, False, "")

    df = pd.DataFrame(out[:rows])
    key = df[["transaction_id", "_fraud", "_pattern"]].rename(columns={"_fraud": "is_fraud", "_pattern": "pattern"})
    return df.drop(columns=["_fraud", "_pattern"]), key
