"""Stateful, one-transaction-at-a-time GNN scoring used by the live monitor and the API."""
from __future__ import annotations

import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .behavior import _clean, time_of
from .explain import Explainer
from .pipeline import PIPELINE_VERSION

REVIEW_SIGNAL_SCORE = 5   # behavioural weight that escalates a borderline GNN score to "review"
REVIEW_FRACTION = 1 / 3   # ...when the GNN score is at least this share of its alert threshold
MODEL = "GNN"


class StaleBundle(RuntimeError):
    """The saved models predate the GNN-only pipeline; the dataset needs retraining."""


class LiveEngine:
    def __init__(self, bundle_path: str | Path):
        b = joblib.load(bundle_path)
        if b.get("version") != PIPELINE_VERSION:
            raise StaleBundle("This dataset was trained by an older version of the app. Retrain it to test the GNN.")
        self.schema, self.fb, self.gnn = b["schema"], b["features"], b["gnn"]
        self.types = b["gnn_types"]
        self.thr = float(b["threshold"])   # chosen on streamed validation scores, the way this engine scores
        self.state, self.origin = b["behavior"], b["origin"]
        self.lock = threading.Lock()
        self.explainer = Explainer(self.fb, b.get("baseline"), self.schema.get("amount"), b.get("background"))

    def _keys(self, row):
        keys = {}
        for t in self.types:
            col = self.schema.get(t)
            if t == "amount_bucket" and self.schema.get("amount"):   # same fallback graph as training
                try:
                    keys[t] = f"amt:{round(float(row.get(self.schema['amount'])), -1)}"
                except (TypeError, ValueError):
                    keys[t] = "amt:0.0"
                continue
            v = row.get(col) if col else None
            keys[t] = None if v is None or str(v).strip() in {"", "nan", "None"} else f"{t}:{v}"
        return keys

    def score(self, row: dict) -> dict:
        return self.score_batch([row])[0]

    def score_batch(self, rows: list[dict]) -> list[dict]:
        """Score transactions in arrival order.

        Behaviour and graph state are updated row by row, so the 2nd row in a batch sees the 1st.
        """
        modes = getattr(self.fb, "cat_modes", {})
        rows = [{**{c: m for c, m in modes.items() if c not in r},
                 **{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}} for r in rows]
        if not rows:
            return []
        feats, sigs, out = [], [], []
        with self.lock:  # behaviour + graph state are mutated per transaction
            for row in rows:
                t, hour = time_of(row, self.schema, self.origin)
                f, s = self.state.process(row, t, hour)
                feats.append(f)
                sigs.append(s)
            X = self.fb.transform(pd.DataFrame(rows), pd.DataFrame(feats))
            for i, row in enumerate(rows):
                keys = self._keys(row)
                # score against the graph *before* this transaction joins it, explain, then add it
                p = self.gnn.score_one(X[i], keys, update=False)
                flagged = p >= self.thr
                sig_score = feats[i]["bh_signal_score"]
                if flagged:
                    level = "fraud"
                elif sig_score >= REVIEW_SIGNAL_SCORE and p >= self.thr * REVIEW_FRACTION:
                    level = "review"
                else:
                    level = "clear"
                res = {"probabilities": {MODEL: p}, "flagged": {MODEL: bool(flagged)}, "threshold": self.thr,
                       "level": level, "signals": sigs[i], "signal_score": sig_score}
                if level != "clear":
                    acct = _clean(row.get(self.schema.get("account"))) if self.schema.get("account") else None
                    res["explanation"] = self.explainer.explain(
                        X[i], keys, p, self.thr, self.gnn, behavior_state=self.state, account_key=acct)
                self.gnn.score_one(X[i], keys, update=True)
                out.append(res)
        return out


_cache: dict[int, LiveEngine] = {}
_cache_lock = threading.Lock()


def api_engine(dataset_id: int, bundle_path) -> LiveEngine:
    """One long-lived engine per dataset for the JSON API (its history keeps growing)."""
    with _cache_lock:
        if dataset_id not in _cache:
            _cache[dataset_id] = LiveEngine(bundle_path)
        return _cache[dataset_id]


def drop_engine(dataset_id: int):
    with _cache_lock:
        _cache.pop(dataset_id, None)
