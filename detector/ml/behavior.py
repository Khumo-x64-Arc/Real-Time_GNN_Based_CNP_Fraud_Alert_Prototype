"""Behavioural signals computed one transaction at a time.

The same `BehaviorState.process()` call is used when building training
features (rows fed in time order) and when scoring live transactions, so a
feature can only ever see the account's *past*. That keeps offline metrics
honest and guarantees the live engine sees identically-built features.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import pandas as pd

HOUR = 3600
DAY = 86400

BEHAVIOR_FEATURES = [
    "bh_prior_tx", "bh_amt_ratio", "bh_amt_z", "bh_log_gap", "bh_tx_1h", "bh_tx_24h",
    "bh_amt_1h", "bh_new_device", "bh_new_ip", "bh_new_country", "bh_new_merchant",
    "bh_foreign", "bh_away_home", "bh_device_accounts", "bh_ip_accounts",
    "bh_balance_ratio", "bh_hour", "bh_night", "bh_signal_score",
]

# Signal code -> (weight, plain-language template). Weights feed bh_signal_score
# and decide whether a borderline model score is escalated to a review alert.
SIGNALS = {
    "foreign_country": (3, "Transaction in {country}; this account normally transacts in {usual}"),
    "new_device": (2, "First time this account has used device {device}"),
    "new_ip": (1, "New IP address for this account ({ip})"),
    "amount_spike": (3, "Amount is {ratio:.1f}× this account's average"),
    "velocity": (2, "{n} transactions on this account in the last hour"),
    "rapid_succession": (2, "Only {secs:.0f}s since the account's previous transaction"),
    "shared_device": (3, "Device used by {n} different accounts"),
    "shared_ip": (2, "IP address used by {n} different accounts"),
    "away_from_home": (1, "Transaction in {city}, away from home city {home}"),
    "night_time": (1, "Made between midnight and 5am ({hour:02d}:00)"),
    "drains_balance": (2, "Amount is {pct:.0f}% of the available balance"),
}


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    return s if s and s.lower() not in {"nan", "none", "null"} else None


def _num(v, default=0.0):
    try:
        f = float(v)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


@dataclass
class AccountProfile:
    n: int = 0
    amt_sum: float = 0.0
    amt_sq: float = 0.0
    last_t: float | None = None
    recent: deque = field(default_factory=deque)  # (t, amount) within 24h
    recent_1h: deque = field(default_factory=deque)  # (t, amount) within 1h
    amt_1h: float = 0.0  # running sum over recent_1h, so velocity is O(1) per row
    devices: set = field(default_factory=set)
    ips: set = field(default_factory=set)
    merchants: set = field(default_factory=set)
    countries: Counter = field(default_factory=Counter)

    def __setstate__(self, state):
        # bundles trained before recent_1h existed: rebuild it from the 24h window
        self.__dict__.update(state)
        if "recent_1h" not in state:
            last = self.last_t if self.last_t is not None else 0.0
            self.recent_1h = deque((rt, a) for rt, a in self.recent if last - rt <= HOUR)
            self.amt_1h = sum(a for _, a in self.recent_1h)


class BehaviorState:
    def __init__(self, schema: dict, default_country: str | None = None):
        self.schema = schema
        self.accounts: dict[str, AccountProfile] = defaultdict(AccountProfile)
        self.device_accounts: dict[str, set] = defaultdict(set)
        self.ip_accounts: dict[str, set] = defaultdict(set)
        self.default_country = default_country
        self.datetime_time = False  # set by timestamp parser

    # ------------------------------------------------------------------ helpers
    def _get(self, row: dict, role: str):
        col = self.schema.get(role)
        return row.get(col) if col else None

    def process(self, row: dict, t: float, hour: int | None) -> tuple[dict, list[dict]]:
        """Return (features, signals) for `row` and then fold it into the history."""
        # No account or card ID (e.g. the ULB dataset, or a row with the ID missing): the row gets
        # neutral history features instead of being pooled into one giant fake account, which
        # made velocity fire on every row and cost O(transactions per day) per row.
        acct_key = _clean(self._get(row, "account")) or _clean(self._get(row, "card"))
        amount = _num(self._get(row, "amount"))
        device = _clean(self._get(row, "device"))
        ip = _clean(self._get(row, "ip"))
        merchant = _clean(self._get(row, "merchant"))
        country = _clean(self._get(row, "country"))
        tx_city = _clean(self._get(row, "tx_city"))
        home_city = _clean(self._get(row, "home_city"))
        balance = self._get(row, "balance")

        p = self.accounts[acct_key] if acct_key else AccountProfile()
        while p.recent and t - p.recent[0][0] > DAY:
            p.recent.popleft()
        while p.recent_1h and t - p.recent_1h[0][0] > HOUR:
            p.amt_1h -= p.recent_1h.popleft()[1]
        if not p.recent_1h:
            p.amt_1h = 0.0  # stop floating-point drift from accumulating
        tx_1h = len(p.recent_1h)
        amt_1h = p.amt_1h

        mean = p.amt_sum / p.n if p.n else 0.0
        var = max(p.amt_sq / p.n - mean * mean, 0.0) if p.n else 0.0
        std = math.sqrt(var)
        gap = max(t - p.last_t, 0.0) if p.last_t is not None else None  # late arrivals clamp to 0
        usual_country = p.countries.most_common(1)[0][0] if p.countries else self.default_country
        has_hist = p.n > 0

        f = {
            "bh_prior_tx": p.n,
            "bh_amt_ratio": amount / (mean + 1.0) if has_hist else 1.0,
            "bh_amt_z": (amount - mean) / (std + 1.0) if p.n >= 2 else 0.0,
            "bh_log_gap": math.log1p(gap) if gap is not None else -1.0,
            "bh_tx_1h": tx_1h,
            "bh_tx_24h": len(p.recent),
            "bh_amt_1h": amt_1h,
            "bh_new_device": int(bool(device) and has_hist and device not in p.devices),
            "bh_new_ip": int(bool(ip) and has_hist and ip not in p.ips),
            "bh_new_country": int(bool(country) and has_hist and country not in p.countries),
            "bh_new_merchant": int(bool(merchant) and has_hist and merchant not in p.merchants),
            "bh_foreign": int(bool(country) and usual_country is not None and country != usual_country),
            "bh_away_home": int(bool(tx_city) and bool(home_city) and tx_city != home_city),
            "bh_device_accounts": len(self.device_accounts[device] | ({acct_key} if acct_key else set())) if device else 0,
            "bh_ip_accounts": len(self.ip_accounts[ip] | ({acct_key} if acct_key else set())) if ip else 0,
            "bh_balance_ratio": amount / (_num(balance) + 1.0) if balance is not None else 0.0,
            "bh_hour": hour if hour is not None else -1,
            "bh_night": int(hour is not None and 0 <= hour < 5),
        }

        signals = []

        def add(code, **kw):
            w, tpl = SIGNALS[code]
            signals.append({"code": code, "weight": w, "text": tpl.format(**kw)})

        if f["bh_foreign"]:
            add("foreign_country", country=country, usual=usual_country)
        if f["bh_new_device"]:
            add("new_device", device=device)
        if f["bh_new_ip"]:
            add("new_ip", ip=ip)
        if p.n >= 3 and amount >= 50 and (f["bh_amt_ratio"] >= 5 or f["bh_amt_z"] >= 4):
            add("amount_spike", ratio=amount / (mean + 1e-9))
        if tx_1h + 1 >= 3:
            add("velocity", n=tx_1h + 1)
        if gap is not None and gap < 60:
            add("rapid_succession", secs=gap)
        if f["bh_device_accounts"] >= 3:
            add("shared_device", n=f["bh_device_accounts"])
        if f["bh_ip_accounts"] >= 3:
            add("shared_ip", n=f["bh_ip_accounts"])
        if f["bh_away_home"]:
            add("away_from_home", city=tx_city, home=home_city)
        if f["bh_night"] and self.datetime_time:
            add("night_time", hour=hour)
        if balance is not None and amount >= 100 and f["bh_balance_ratio"] >= 0.5:
            add("drains_balance", pct=100 * amount / (_num(balance) + 1e-9))
        signals.sort(key=lambda s: -s["weight"])
        f["bh_signal_score"] = sum(s["weight"] for s in signals)

        # ---- fold the transaction into history (after features were taken)
        if not acct_key:
            return f, signals
        p.n += 1
        p.amt_sum += amount
        p.amt_sq += amount * amount
        p.last_t = t
        p.recent.append((t, amount))
        p.recent_1h.append((t, amount))
        p.amt_1h += amount
        if device:
            p.devices.add(device)
            self.device_accounts[device].add(acct_key)
        if ip:
            p.ips.add(ip)
            self.ip_accounts[ip].add(acct_key)
        if merchant:
            p.merchants.add(merchant)
        if country:
            p.countries[country] += 1
        return f, signals


def parse_times(df: pd.DataFrame, schema: dict) -> tuple[pd.Series, pd.Series | None, bool]:
    """Return (seconds as float, hour of day or None, is_real_datetime)."""
    col = schema.get("time")
    if col is None:
        return pd.Series(range(len(df)), index=df.index, dtype=float), None, False
    s = df[col]
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.95:
        secs = num.fillna(num.median()).astype(float)
        # ULB `Time` / IEEE `TransactionDT` are seconds from an arbitrary origin.
        hour = ((secs // HOUR) % 24).astype(int)
        return secs, hour, False
    dt = pd.to_datetime(s, errors="coerce")
    base = dt.min()
    secs = (dt - base).dt.total_seconds().fillna(0.0)
    return secs, dt.dt.hour.fillna(-1).astype(int), True


def time_of(row: dict, schema: dict, origin) -> tuple[float, int | None]:
    """Time for a single live transaction, relative to the training origin."""
    col = schema.get("time")
    if col is None or row.get(col) is None:
        return 0.0, None
    v = row[col]
    try:
        secs = float(v)
        return secs, int((secs // HOUR) % 24)
    except (TypeError, ValueError):
        dt = pd.to_datetime(v, errors="coerce")
        if pd.isna(dt):
            return 0.0, None
        return (dt - origin).total_seconds() if origin is not None else 0.0, int(dt.hour)
