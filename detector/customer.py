"""Everything that feeds the customer portal.

Only accounts with a CustomerProfile are recorded, so the portal stores
nothing about customers who haven't enrolled. Customers are shown behavioural
reasons in everyday language, never model internals or feature attributions:
telling a fraudster exactly which signals the models react to would help
them evade detection.
"""
from __future__ import annotations

import logging
import math
import threading
import time

import pandas as pd
from django.conf import settings
from django.core.mail import send_mail

from .models import AccountTransaction, CustomerProfile

log = logging.getLogger(__name__)

_enrolled: dict[int, tuple[float, set]] = {}
_lock = threading.Lock()
TTL = 5.0  # seconds; new enrolments show up in the live stream within this delay

STATUS_FROM_LEVEL = {"clear": "normal", "review": "review", "fraud": "flagged"}


def enrolled_accounts(dataset_id: int) -> set:
    now = time.time()
    with _lock:
        hit = _enrolled.get(dataset_id)
        if hit and now - hit[0] < TTL:
            return hit[1]
    accts = set(CustomerProfile.objects.filter(dataset_id=dataset_id).values_list("account_id", flat=True))
    with _lock:
        _enrolled[dataset_id] = (now, accts)
    return accts


def forget_enrolments(dataset_id: int):
    with _lock:
        _enrolled.pop(dataset_id, None)


def _v(row, col):
    if not col:
        return None
    v = row.get(col)
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def customer_reasons(signals: list[dict]) -> list[str]:
    return [s["text"].replace("this account's", "your").replace("this account", "your account")
            .replace("This account", "Your account") for s in signals]


def _row_fields(ds, row):
    sc = ds.schema or {}
    amt = _v(row, sc.get("amount"))
    try:
        amt = float(amt) if amt is not None else None
    except (TypeError, ValueError):
        amt = None
    return {
        "amount": amt,
        "merchant": str(_v(row, sc.get("merchant_name")) or _v(row, sc.get("merchant")) or "")[:200],
        "category": str(_v(row, sc.get("category")) or "")[:80],
        "country": str(_v(row, sc.get("country")) or "")[:60],
        "channel": str(_v(row, sc.get("channel")) or "")[:40],
        "occurred_at": str(_v(row, sc.get("time")) or "")[:60],
    }


def record_customer_tx(ds, row: dict, result: dict, alert=None):
    """Store a scored transaction if its account belongs to an enrolled customer."""
    sc = ds.schema or {}
    acct = _v(row, sc.get("account"))
    if acct is None or str(acct) not in enrolled_accounts(ds.pk):
        return None
    acct = str(acct)
    tx_id = str(_v(row, sc.get("tx_id")) or (alert.transaction_id if alert else f"{acct}@{_v(row, sc.get('time'))}"))
    status = STATUS_FROM_LEVEL[result["level"]]
    tx, created = AccountTransaction.objects.update_or_create(
        dataset=ds, transaction_id=tx_id,
        defaults={"account_id": acct, "status": status, "alert": alert,
                  "reasons": customer_reasons(result["signals"]) if status != "normal" else [],
                  **_row_fields(ds, row)},
    )
    if created and status == "flagged":
        notify_customers(ds, acct, tx)
    return tx


def notify_customers(ds, account_id, tx):
    """Email every enrolled customer on the account who has email alerts on."""
    for prof in CustomerProfile.objects.filter(dataset=ds, account_id=account_id, email_alerts=True).select_related("user"):
        if not prof.user.email:
            continue
        reasons = "\n".join(f"  - {r}" for r in tx.reasons) or "  - It doesn't match how you normally use your card."
        amount = f"{tx.amount:,.2f}" if tx.amount is not None else "an unknown amount"
        body = (f"Hi {prof.display_name or prof.user.username},\n\n"
                f"We flagged a transaction on your account {prof.masked_account} as possibly fraudulent:\n\n"
                f"  {amount} at {tx.merchant or 'an unknown merchant'}"
                f"{' in ' + tx.country if tx.country else ''}, {tx.occurred_at}\n\n"
                f"Why it looked unusual:\n{reasons}\n\n"
                f"Please sign in to Khumo Detection and tell us whether this was you.\n"
                f"We will never ask for your PIN or password by email or phone.\n")
        try:
            send_mail(f"Was this you? {amount} at {tx.merchant or 'unknown merchant'}", body,
                      settings.DEFAULT_FROM_EMAIL, [prof.user.email])
        except Exception:
            log.exception("Could not email customer %s", prof.user_id)


def verify_account(ds, account_id: str, card_last4: str) -> bool:
    """Check the account exists in the dataset and the card digits match one of its cards."""
    sc = ds.schema or {}
    acct_col = sc.get("account")
    if not acct_col or "card_last4" not in dataset_columns(ds):
        return False
    try:
        df = pd.read_csv(ds.file.path, usecols=[acct_col, "card_last4"], dtype=str)
    except Exception:
        return False
    cards = df.loc[df[acct_col] == account_id, "card_last4"].dropna().str.zfill(4)
    return card_last4.strip().zfill(4) in set(cards)


def dataset_columns(ds) -> list:
    """Column names; read from the file header for datasets added by the management command."""
    if ds.columns:
        return ds.columns
    try:
        cols = list(pd.read_csv(ds.file.path, nrows=0).columns)
    except Exception:
        return []
    type(ds).objects.filter(pk=ds.pk).update(columns=cols)
    ds.columns = cols
    return cols


def can_self_enrol(ds) -> bool:
    return bool((ds.schema or {}).get("account")) and "card_last4" in dataset_columns(ds)


def backfill_history(ds, account_id: str, limit=30) -> int:
    """Copy the account's recent pre-monitoring transactions so a new customer isn't staring at an empty list.

    Rows in the live-replay period are skipped: they arrive through the stream.
    """
    sc = ds.schema or {}
    acct_col = sc.get("account")
    if not acct_col:
        return 0
    try:
        df = pd.read_csv(ds.file.path, low_memory=False)
    except Exception:
        return 0
    df = df[df[acct_col].astype(str) == account_id]
    replay_ids = set()
    replay_path = ds.model_dir / "replay.pkl"
    if sc.get("tx_id") and replay_path.exists():
        replay_ids = set(pd.read_pickle(replay_path)[sc["tx_id"]].astype(str))
        df = df[~df[sc["tx_id"]].astype(str).isin(replay_ids)]
    if sc.get("time"):
        df = df.sort_values(sc["time"])
    rows = df.tail(limit).to_dict("records")
    made = 0
    for row in rows:
        tx_id = str(_v(row, sc.get("tx_id")) or f"{account_id}@{_v(row, sc.get('time'))}")
        _, created = AccountTransaction.objects.get_or_create(
            dataset=ds, transaction_id=tx_id,
            defaults={"account_id": account_id, "status": "history", **_row_fields(ds, row)})
        made += created
    return made
