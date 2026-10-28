"""Turn raw rows + behavioural features into a fixed numeric matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .behavior import BEHAVIOR_FEATURES
from .schema import NEVER_FEATURES

MAX_LEVELS = 40


class FeatureBuilder:
    def fit(self, df: pd.DataFrame, schema: dict):
        skip = {c for r, c in schema.items() if r in
                {"label", "tx_id", "time", "account", "card", "device", "ip", "merchant", "merchant_name", "terminal"} and c}
        skip |= {c for c in df.columns if c.lower() in NEVER_FEATURES}
        self.numeric, self.categorical = [], {}
        n = len(df)
        for c in df.columns:
            if c in skip:
                continue
            col = df[c]
            if pd.api.types.is_numeric_dtype(col):
                # integer columns that are unique per row are identifiers, not signals
                if pd.api.types.is_integer_dtype(col) and col.nunique() > 0.9 * n:
                    continue
                self.numeric.append(c)
            else:
                vc = col.astype(str).value_counts()
                if 1 < len(vc) <= MAX_LEVELS:
                    self.categorical[c] = list(vc.index)
                elif len(vc) > MAX_LEVELS and (vc.iloc[:MAX_LEVELS].sum() / n) > 0.8:
                    self.categorical[c] = list(vc.index[:MAX_LEVELS])  # long tail -> "other"
        self.medians = df[self.numeric].median(numeric_only=True).to_dict()
        # used when a live transaction omits a categorical column entirely
        self.cat_modes = {c: lv[0] for c, lv in self.categorical.items()}
        self.columns = (self.numeric
                        + [f"{c}={v}" for c, lv in self.categorical.items() for v in lv]
                        + BEHAVIOR_FEATURES)
        return self

    def transform(self, df: pd.DataFrame, behavior: pd.DataFrame) -> np.ndarray:
        parts = []
        if self.numeric:
            num = df.reindex(columns=self.numeric).apply(pd.to_numeric, errors="coerce")
            parts.append(num.fillna(self.medians).fillna(0).to_numpy(float))
        for c, levels in self.categorical.items():
            vals = df[c].astype(str).to_numpy() if c in df else np.array([""] * len(df))
            parts.append((vals[:, None] == np.array(levels)[None, :]).astype(float))
        parts.append(behavior.reindex(columns=BEHAVIOR_FEATURES).fillna(0).to_numpy(float))
        return np.hstack(parts)
