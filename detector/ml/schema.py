"""Detect which columns of an uploaded CSV play which role.

Uploaded datasets rarely share column names (ULB `Class`, IEEE-CIS `isFraud`,
Sparkov `is_fraud` ...), so every role has a list of candidate names. The first
match (case-insensitive) wins. Everything downstream works off this mapping.
"""
from __future__ import annotations

import pandas as pd

ROLE_CANDIDATES: dict[str, list[str]] = {
    "label": ["class", "isfraud", "is_fraud", "fraud", "fraud_flag", "label", "target"],
    "tx_id": ["transaction_id", "transactionid", "trans_num", "tx_id", "id"],
    "time": ["timestamp", "trans_date_trans_time", "datetime", "transaction_datetime",
             "transactiondt", "time", "unix_time", "step"],
    "amount": ["amount", "transactionamt", "amt", "transaction_amount"],
    "account": ["account_id", "customer_ref", "customer_id", "account_number", "cc_num", "card1", "nameorig"],
    "card": ["card_token", "cc_num", "card_number"],
    "device": ["device_id", "deviceinfo", "device"],
    "ip": ["ip_address", "ip", "ipaddress"],
    "merchant": ["merchant_id", "merchant", "merchant_name", "namedest"],
    "merchant_name": ["merchant_name", "merchant", "namedest"],  # display only
    "terminal": ["terminal_id"],
    "country": ["tx_country", "country", "merchant_country"],
    "tx_city": ["tx_city", "city"],
    "home_city": ["home_city"],
    "balance": ["account_balance", "oldbalanceorg", "balance"],
    "channel": ["channel"],
    "entry_mode": ["entry_mode"],
    "category": ["merchant_category", "category", "productcd"],
}

# Roles whose values identify an entity; these become GNN graph nodes.
ENTITY_ROLES = ["account", "card", "device", "ip", "merchant", "terminal"]

# Columns that are identifiers or leak the answer; never used as model inputs.
NEVER_FEATURES = {
    "reference", "account_number", "card_last4", "cardholder_name", "customer_ref",
    "posting_date", "transaction_date", "first", "last", "street", "dob", "trans_num",
    "cc_num", "unix_time", "zip",
}


def detect_schema(df: pd.DataFrame) -> dict[str, str | None]:
    lower = {c.lower(): c for c in df.columns}
    schema: dict[str, str | None] = {}
    used: set[str] = set()
    for role, candidates in ROLE_CANDIDATES.items():
        schema[role] = None
        for cand in candidates:
            col = lower.get(cand)
            # card / merchant_name may legitimately reuse a column (cc_num, merchant); others may not.
            if col is not None and (col not in used or role in {"card", "merchant_name"}):
                schema[role] = col
                used.add(col)
                break
    # A label column must be binary 0/1; the IEEE sample_submission has isFraud=0.5 everywhere.
    lab = schema.get("label")
    if lab is not None:
        vals = pd.to_numeric(df[lab], errors="coerce").dropna().unique()
        if len(vals) == 0 or not set(vals).issubset({0, 1}) or len(set(vals)) < 2:
            schema["label_problem"] = (
                f"Column '{lab}' is not a 0/1 fraud label "
                f"(values seen: {', '.join(map(str, sorted(vals)[:5]))})."
            )
            schema["label"] = None
    return schema


def describe_problems(df: pd.DataFrame, schema: dict) -> list[str]:
    """Human-readable reasons a dataset can't be trained on (empty list = OK)."""
    problems = []
    if schema.get("label") is None:
        problems.append(schema.get("label_problem") or
                        "No fraud label column found. Name it Class, isFraud or is_fraud with 0/1 values.")
    label_like = {schema.get("label"), schema.get("tx_id")}
    usable = [c for c in df.columns if c not in label_like]
    if len(usable) < 2:
        problems.append("The file has no transaction features to learn from, only an ID"
                        + (" and a label." if schema.get("label") else "."))
    if schema.get("label") is not None:
        pos = int(pd.to_numeric(df[schema["label"]], errors="coerce").fillna(0).sum())
        if pos < 20:
            problems.append(f"Only {pos} fraudulent rows; at least 20 are needed to train and test fairly.")
    return problems
