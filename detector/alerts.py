"""Persist alerts and notify people (email + the in-browser notifier polls the table)."""
import logging

from django.conf import settings
from django.core.mail import send_mail

from .models import Alert

log = logging.getLogger(__name__)


def _s(v, limit=200):
    return "" if v is None else str(v)[:limit]


def record_alert(dataset, row: dict, result: dict, source="monitor"):
    """Create/update the alert for a scored transaction. Returns the Alert or None if clear."""
    if result["level"] == "clear":
        return None
    sc = dataset.schema or {}
    tx_id = _s(row.get(sc.get("tx_id"))) if sc.get("tx_id") else ""
    if not tx_id:
        tx_id = f"{_s(row.get(sc.get('account')))}@{_s(row.get(sc.get('time')))}"
    label = row.get(sc.get("label")) if sc.get("label") else None
    try:
        amount = float(row.get(sc.get("amount"))) if sc.get("amount") else None
    except (TypeError, ValueError):
        amount = None
    alert, created = Alert.objects.update_or_create(
        dataset=dataset, transaction_id=tx_id,
        defaults=dict(
            level=result["level"], score=result["probabilities"]["GNN"],
            probabilities=result["probabilities"], signals=result["signals"], amount=amount,
            explanation=result.get("explanation") or {},
            account=_s(row.get(sc.get("account"))), merchant=_s(row.get(sc.get("merchant_name") or sc.get("merchant"))),
            country=_s(row.get(sc.get("country"))), occurred_at=_s(row.get(sc.get("time")), 60),
            actual_label=int(label) if label in (0, 1, "0", "1") else None, source=source,
        ),
    )
    if created and settings.ALERT_EMAIL_TO and alert.level in settings.ALERT_EMAIL_LEVELS:
        send_alert_email(alert)
    return alert


def send_alert_email(alert: Alert):
    reasons = "\n".join(f"  - {s['text']}" for s in alert.signals) or "  - Model score only"
    thr = (alert.explanation.get("thresholds") or {}).get("GNN")
    probs = f"  GNN: {alert.score:.1%}" + (f" (alert threshold {thr:.1%})" if thr is not None else "")
    why = alert.explanation.get("summary", "")
    drivers = "\n".join(f"  - {d['label']}: {d['value']} (typical {d['typical']}), "
                        f"{d.get('delta', 0):+.2f} log-odds" for d in alert.explanation.get("drivers", [])[:5])
    body = (f"{alert.get_level_display()} on transaction {alert.transaction_id}\n\n"
            f"{why}\n\n" + (f"Top drivers (GNN log-odds; +1 = about 2.7x the odds of fraud):\n{drivers}\n\n" if drivers else "") +
            f"Amount: {alert.amount}\nAccount: {alert.account}\nMerchant: {alert.merchant}\n"
            f"Country: {alert.country}\nTime: {alert.occurred_at}\n\n"
            f"Behavioural signals:\n{reasons}\n\nGNN score:\n{probs}\n")
    try:
        send_mail(f"[Khumo Detection] {alert.get_level_display()}: {alert.transaction_id}", body,
                  settings.DEFAULT_FROM_EMAIL, settings.ALERT_EMAIL_TO)
        Alert.objects.filter(pk=alert.pk).update(emailed=True)
    except Exception:
        log.exception("Could not send alert email for %s", alert.transaction_id)
