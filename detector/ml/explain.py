"""Explain why the GNN flagged a transaction.

Three complementary views, computed at alert time for flagged rows only:

Attributions are in log-odds (the GNN's output before the sigmoid). The GNN is
often near-certain, and at 99.99% a probability barely moves whatever is
changed, while the log-odds still show how much each piece of evidence adds.
+1 log-odds multiplies the odds of fraud by e (about 2.7).

1. Feature attribution (occlusion)
   Replace one feature group (a numeric column, a whole categorical column, or
   one behavioural feature) with its value in each of 12 legitimate training
   transactions, rescore with the GNN against the same graph neighbourhood, and
   measure how far the log-odds fall on average. Averaging over a
   background sample (as interventional SHAP does) avoids blaming a feature
   just because a single reference value happens to be unusual. Only the
   transaction's own features are swapped; its neighbours stay as they are.

2. Graph attribution
   Hide one entity at a time (account, device, IP, ...) so the GNN scores the
   transaction as if it had never seen that entity, and measure the drop in
   log-odds.

3. Counterfactual
   Swap the strongest drivers for values from typical legitimate transactions,
   one more at a time, until the GNN score falls below its alert threshold.
"""
from __future__ import annotations

import math

import numpy as np

from .behavior import BEHAVIOR_FEATURES

TOP_N = 10
MAX_COUNTERFACTUAL = 8
MIN_DRIVER = 0.25   # log-odds a feature must add before the summary names it
MIN_LINK = 0.5      # log-odds an entity link must add before the summary names it

BEHAVIOR_LABELS = {
    "bh_prior_tx": "Earlier transactions on the account",
    "bh_amt_ratio": "Amount vs the account's average",
    "bh_amt_z": "Amount z-score vs account history",
    "bh_log_gap": "Time since the account's previous transaction",
    "bh_tx_1h": "Account transactions in the past hour",
    "bh_tx_24h": "Account transactions in the past 24 hours",
    "bh_amt_1h": "Amount spent on the account in the past hour",
    "bh_new_device": "Device is new to the account",
    "bh_new_ip": "IP address is new to the account",
    "bh_new_country": "Country is new to the account",
    "bh_new_merchant": "Merchant is new to the account",
    "bh_foreign": "Outside the account's usual country",
    "bh_away_home": "Away from the home city",
    "bh_device_accounts": "Accounts seen on this device",
    "bh_ip_accounts": "Accounts seen on this IP address",
    "bh_balance_ratio": "Amount as a share of the balance",
    "bh_hour": "Hour of day",
    "bh_night": "Between midnight and 5am",
    "bh_signal_score": "Behavioural risk score (sum of the signals)",
}
BINARY = {"bh_new_device", "bh_new_ip", "bh_new_country", "bh_new_merchant", "bh_foreign",
          "bh_away_home", "bh_night"}
# Phrases for the one-line summary when a yes/no behaviour drives the score.
BINARY_PHRASE = {
    "bh_new_device": "the device has never been used on this account",
    "bh_new_ip": "the IP address is new to this account",
    "bh_new_country": "the account has never transacted in this country",
    "bh_new_merchant": "the account has never paid this merchant",
    "bh_foreign": "it happened outside the account's usual country",
    "bh_away_home": "it happened away from the cardholder's home city",
    "bh_night": "it happened between midnight and 5am",
}
ENTITY_LABELS = {"account": "Account", "card": "Card", "device": "Device", "ip": "IP address",
                 "merchant": "Merchant", "terminal": "Terminal", "amount_bucket": "Amount bucket"}


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}"


def _humanize_seconds(s: float) -> str:
    if s < 0:
        return "no earlier transaction"
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    if s < 172800:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.0f} days"


class Explainer:
    def __init__(self, fb, baseline: np.ndarray | None, amount_col: str | None, background=None):
        self.fb = fb
        self.amount_col = amount_col
        cols = fb.columns
        self.groups: list[dict] = []
        for c in fb.numeric:
            self.groups.append({"name": c, "kind": "numeric", "idx": [cols.index(c)]})
        for c, levels in fb.categorical.items():
            self.groups.append({"name": c, "kind": "category", "levels": levels,
                                "idx": [cols.index(f"{c}={v}") for v in levels]})
        for c in BEHAVIOR_FEATURES:
            self.groups.append({"name": c, "kind": "behavior", "idx": [cols.index(c)]})
        self.col_group = np.zeros(len(cols), dtype=int)
        for g_i, g in enumerate(self.groups):
            self.col_group[g["idx"]] = g_i
        if baseline is None:  # bundles trained before XAI existed
            baseline = np.zeros(len(cols))
            for c in fb.numeric:
                baseline[cols.index(c)] = fb.medians.get(c, 0.0)
            for c, m in getattr(fb, "cat_modes", {}).items():
                baseline[cols.index(f"{c}={m}")] = 1.0
        self.base = baseline
        self.bg = background if background is not None and len(background) else baseline[None, :]

    # ------------------------------------------------------------- formatting
    def _label(self, g):
        n = g["name"]
        if g["kind"] == "behavior":
            return BEHAVIOR_LABELS.get(n, n)
        if n == self.amount_col:
            return "Transaction amount"
        if n[:1] in "Vv" and n[1:].isdigit():
            return f"{n} (anonymised PCA component)"
        nice = n.lower()
        for a, b in (("tx_", "transaction "), ("_id", " ID"), ("_", " ")):
            nice = nice.replace(a, b)
        return nice[:1].upper() + nice[1:]

    def _value(self, g, vec):
        n = g["name"]
        if g["kind"] == "category":
            hot = [lv for lv, i in zip(g["levels"], g["idx"]) if vec[i] > 0.5]
            return hot[0] if hot else "other"
        v = float(vec[g["idx"][0]])
        if n in BINARY:
            return "yes" if v > 0.5 else "no"
        if n == "bh_log_gap":
            return _humanize_seconds(math.expm1(v) if v >= 0 else -1)
        if n == "bh_amt_ratio":
            return f"{v:.1f}×"
        if n == "bh_balance_ratio":
            return f"{v * 100:.0f}%"
        if n == "bh_hour":
            return f"{int(v):02d}:00" if v >= 0 else "–"
        if n in {"bh_prior_tx", "bh_tx_1h", "bh_tx_24h", "bh_device_accounts", "bh_ip_accounts"}:
            return f"{int(round(v))}"
        return _fmt(v)

    def _phrase(self, d):
        n = d["group"]
        if n in BINARY_PHRASE:
            return BINARY_PHRASE[n]
        if d["kind"] == "category":
            return f"{d['label'].lower()} is {d['value']} (typically {d['typical']})"
        if n == "bh_tx_1h":
            return f"the account made {int(d['raw']) + 1} transactions within an hour"
        if n == "bh_log_gap":
            return f"only {d['value']} passed since the account's previous transaction"
        if n == "bh_amt_ratio":
            return f"the amount is {d['value']} the account's usual spend"
        if n == "bh_balance_ratio":
            return f"the amount is {d['value']} of the available balance"
        if n == "bh_device_accounts":
            return f"the device has been used by {d['value']} accounts"
        if n == "bh_ip_accounts":
            return f"the IP address has been used by {d['value']} accounts"
        direction = "higher" if d["raw"] > d["typical_raw"] else "lower"
        return f"{d['label'].split(' (')[0]} is {direction} than usual ({d['value']} vs {d['typical']})"

    # ------------------------------------------------------------- main
    def explain(self, x, keys, p, threshold, gnn, behavior_state=None, account_key=None):
        """p = the GNN's probability for x; threshold = its alert threshold. Returns a JSON-serialisable dict."""
        G, nb = len(self.groups), len(self.bg)
        z = gnn.score_one(x, keys, update=False, as_logit=True)

        def logits(variants):
            return np.array([gnn.score_one(v, keys, update=False, as_logit=True) for v in variants]).reshape(-1, nb)

        def avg_probs(variants):
            return (1 / (1 + np.exp(-np.clip(logits(variants), -30, 30)))).mean(1)

        # 1. occlusion averaged over the background sample, in log-odds
        variants = np.repeat(x[None, :], G * nb, axis=0)
        for g_i, g in enumerate(self.groups):
            variants[g_i * nb:(g_i + 1) * nb][:, g["idx"]] = self.bg[:, g["idx"]]
        delta = z - logits(variants).mean(1)

        order = np.argsort(-np.abs(delta))
        drivers = []
        for g_i in order[:TOP_N]:
            g = self.groups[g_i]
            raw = float(x[g["idx"][0]]) if g["kind"] != "category" else 0.0
            drivers.append({"group": g["name"], "kind": g["kind"], "label": self._label(g),
                            "value": self._value(g, x), "typical": self._value(g, self.base),
                            "raw": raw, "typical_raw": float(self.base[g["idx"][0]]) if g["kind"] != "category" else 0.0,
                            "delta": float(delta[g_i])})

        # 2. graph attribution
        graph = []
        for t in gnn.types:
            if not keys.get(t):
                continue
            hidden = dict(keys, **{t: None})
            zh = gnn.score_one(x, hidden, update=False, as_logit=True)
            st = gnn.state[t]
            j = st["idx"].get(keys[t])
            past = int(st["cnt"][j]) if j is not None else 0
            entry = {"type": t, "label": ENTITY_LABELS.get(t, t.title()), "key": keys[t].split(":", 1)[-1],
                     "delta": float(z - zh), "p_without": float(1 / (1 + math.exp(-max(-30.0, min(30.0, zh))))),
                     "past_tx": past}
            if behavior_state is not None and t in {"device", "ip"}:
                pool = behavior_state.device_accounts if t == "device" else behavior_state.ip_accounts
                entry["accounts"] = len(pool.get(entry["key"], set()) | ({account_key} if account_key else set()))
            graph.append(entry)
        graph.sort(key=lambda e: -abs(e["delta"]))

        # 3. counterfactual: reset top positive drivers cumulatively
        pos = [i for i in order if delta[i] > 0][:MAX_COUNTERFACTUAL]
        cf = None
        if pos:
            steps = np.repeat(x[None, :], len(pos) * nb, axis=0)
            for st in range(len(pos)):
                block = steps[st * nb:(st + 1) * nb]
                for g_i in pos[: st + 1]:
                    block[:, self.groups[g_i]["idx"]] = self.bg[:, self.groups[g_i]["idx"]]
            path = avg_probs(steps)
            below = np.where(path < threshold)[0]
            k = int(below[0]) + 1 if len(below) else None
            cf = {"changed": [self._label(self.groups[i]) for i in pos[: (k or len(pos))]],
                  "new_score": float(path[(k or len(pos)) - 1]), "crosses": k is not None,
                  "threshold": float(threshold), "path": [float(c) for c in path]}

        top_pos = [d for d in drivers if d["delta"] >= MIN_DRIVER][:3]
        risky = [e for e in graph if e["delta"] >= MIN_LINK]
        parts = []
        if risky:
            e = risky[0]
            parts.append(f"its {e['label'].lower()} {e['key']} links it to {e['past_tx']} earlier "
                         f"transaction{'s' if e['past_tx'] != 1 else ''} whose pattern the GNN reads as risky"
                         + (f" (without that link it would score {e['p_without']:.0%}, under the threshold)"
                            if e["p_without"] < threshold else ""))
        parts += [self._phrase(d) for d in top_pos[: 3 - len(parts)]]
        summary = ""
        if parts:
            summary = "The GNN flagged it mainly because " + (
                parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]) + "."
        elif p >= threshold:
            summary = (f"The GNN scored it {p:.0%}, above its {threshold:.0%} threshold, from the combination of "
                       f"its features and graph neighbourhood rather than any single factor.")
        if cf and cf["crosses"]:
            names = [nm.split(" (")[0] for nm in cf["changed"]]
            k = len(names)
            what = (f"Changing {names[0]} alone" if k == 1 else
                    f"Changing {', '.join(names[:-1])} and {names[-1]}" if k <= 3 else
                    f"Changing the top {k} factors")
            summary += (f" {what} to values typical of legitimate transactions would lower the GNN score "
                        f"from {p:.0%} to {cf['new_score']:.0%}, below the {cf['threshold']:.0%} alert threshold.")

        return {
            "summary": summary, "drivers": drivers, "graph": graph, "counterfactual": cf,
            "probabilities": {"GNN": float(p)}, "thresholds": {"GNN": float(threshold)}, "model": "GNN",
            "logit": float(z), "unit": "log-odds",
        }
