from django import template

register = template.Library()


@register.filter
def pct(v, digits=1):
    try:
        return f"{float(v) * 100:.{int(digits)}f}%"
    except (TypeError, ValueError):
        return "–"


@register.filter
def num(v, digits=3):
    try:
        return f"{float(v):.{int(digits)}f}"
    except (TypeError, ValueError):
        return "–"


@register.filter
def money(v):
    try:
        return f"{float(v):,.2f}"
    except (TypeError, ValueError):
        return "–"


@register.filter
def get(d, k):
    try:
        return d.get(k)
    except AttributeError:
        return None


@register.filter
def mul(v, k):
    try:
        return float(v) * float(k)
    except (TypeError, ValueError):
        return 0


@register.filter
def signed(v, scale=1):
    """+17.8 / −3.2 with a real minus sign; scale=100 turns probabilities into points."""
    try:
        x = float(v) * float(scale)
    except (TypeError, ValueError):
        return "–"
    digits = 1 if float(scale) != 1 else 2
    return f"+{x:.{digits}f}" if x > 0 else f"−{abs(x):.{digits}f}" if x < 0 else f"{0:.{digits}f}"


@register.filter
def sig(v, digits=3):
    """A score that may be tiny: 0.742, 0.0381, 3.2e-05."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "–"
    return f"{v:.{int(digits)}g}" if v and abs(v) < 0.001 else f"{v:.{int(digits)}f}"


@register.filter
def words(v):
    """account_takeover -> Account takeover"""
    s = str(v or "").replace("_", " ").strip()
    return s[:1].upper() + s[1:]


@register.filter
def ring_grouped(known_rings):
    """'9 of 21' for the planted test-period ring transactions the agent grouped into ring cases."""
    scored = [r for r in known_rings or [] if r.get("scored")]
    if not scored:
        return "–"
    return f"{sum(r['in_ring_case'] for r in scored)} of {sum(r['transactions'] for r in scored)}"


@register.filter
def adv_escalated(adv):
    """'12 of 98' adversarial test frauds the agent escalated (the naive control group excluded)."""
    g = [x for x in (adv or {}).get("groups", []) if x["group"] != "C0"]
    if not g:
        return "–"
    return f"{sum(x['escalated'] for x in g)} of {sum(x['transactions'] for x in g)}"


@register.filter
def delta(v, digits=3):
    """+0.007 / −0.012 / 0.000: a signed difference with a real minus sign."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "–"
    s = f"{abs(v):.{int(digits)}f}"
    if float(s) == 0:
        return s
    return ("+" if v > 0 else "−") + s
