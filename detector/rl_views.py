"""RL agent section: semi-supervised GNN + DQN experiments on the European Cardholder and IEEE-CIS data."""
import math
import shutil
from pathlib import Path

from django import forms
from django.conf import settings
from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

from .ml.rl_pipeline import POLICY_LABELS, POLICY_ORDER
from .models import RLRun
from .tasks import start_rl

POLICY_COLORS = {"fixed": "var(--muted)", "static": "var(--rust)", "agent": "var(--m-gnn)",
                 "online": "var(--orange)", "no_gnn": "var(--g-account)"}
POLICY_DASH = {"fixed": "6 4", "static": "2 3", "agent": "", "online": "", "no_gnn": "8 3 2 3"}


class RunForm(forms.Form):
    kind = forms.ChoiceField(choices=RLRun.Kind.choices, label="Dataset")
    file = forms.FileField(required=False, label="Upload the data",
                           help_text="creditcard.csv for the European Cardholder data; the Kaggle zip for IEEE-CIS.")
    source = forms.CharField(required=False, max_length=500, label="…or a path on this server",
                             help_text="For big files: a path the server can read, e.g. /data/ieee-fraud-detection.zip")
    label_rate = forms.FloatField(initial=20, min_value=1, max_value=100, label="Labels revealed (%)",
                                  help_text="Share of transactions whose true label is ever known. The rest stay unlabelled.")
    episodes = forms.IntegerField(required=False, min_value=20, max_value=5000, label="DQN training episodes",
                                  help_text="Leave empty for the default: 300 for the European data, 600 for IEEE-CIS.")
    max_rows = forms.IntegerField(required=False, min_value=5000, label="Use only the first N transactions",
                                  help_text="Optional, for a quicker run.")
    name = forms.CharField(required=False, max_length=200)

    def clean(self):
        c = super().clean()
        if not c.get("file") and not c.get("source"):
            raise forms.ValidationError("Upload a file or give a path on the server.")
        if c.get("source") and not Path(c["source"]).exists():
            self.add_error("source", "The server can't find that path.")
        return c


def rl_overview(request):
    form = RunForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        c = form.cleaned_data
        if c.get("file"):
            dest = Path(settings.MEDIA_ROOT) / "rl_sources"
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / Path(c["file"].name).name
            with open(target, "wb") as out:
                for chunk in c["file"].chunks():
                    out.write(chunk)
            source = str(target)
        else:
            source = str(Path(c["source"]).resolve())
        params = {"label_rate": c["label_rate"] / 100, "episodes": c.get("episodes"), "max_rows": c.get("max_rows")}
        run = RLRun.objects.create(name=c["name"] or f"{RLRun.Kind(c['kind']).label} · {c['label_rate']:.0f}% labels",
                                   kind=c["kind"], source=source, params=params)
        start_rl(run.pk)
        messages.success(request, f"Started {run.name}.")
        return redirect(run)
    return render(request, "detector/rl.html", {"form": form, "runs": RLRun.objects.all()[:50]})


@require_GET
def rl_status(request, pk):
    r = get_object_or_404(RLRun, pk=pk)
    return JsonResponse({"status": r.status, "progress": r.progress, "message": r.progress_message, "error": r.error})


@require_POST
def rl_delete(request, pk):
    r = get_object_or_404(RLRun, pk=pk)
    shutil.rmtree(r.run_dir, ignore_errors=True)
    r.delete()
    messages.success(request, f"Deleted {r.name}.")
    return redirect("rl")


def _nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start, end = math.floor(lo / step) * step, math.ceil(hi / step) * step   # always cover the data
    return [round(start + i * step, 10) for i in range(int(round((end - start) / step)) + 1)]


def _line_chart(series, w=640, h=260, x_label="", y_label="", zero=False, hline=None):
    """series: [{key, label, color, dash, y: [...]}] sharing x = 1..len. Returns SVG geometry for the template."""
    left, right, top, bottom = 56, 14, 12, 38
    ys = [v for s in series for v in s["y"] if v is not None]
    if hline is not None:
        ys.append(hline["y"])
    if not ys:
        return None
    lo, hi = min(ys), max(ys)
    if zero:
        lo, hi = min(lo, 0), max(hi, 0)
    ticks = _nice_ticks(lo, hi)
    lo, hi = ticks[0], ticks[-1]
    nmax = max(len(s["y"]) for s in series)

    def sx(i):
        return left + (i / max(nmax - 1, 1)) * (w - left - right)

    def sy(v):
        return top + (1 - (v - lo) / (hi - lo or 1)) * (h - top - bottom)

    out = []
    for s in series:
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(s["y"]) if v is not None)
        out.append({**s, "points": pts, "last_x": round(sx(len(s["y"]) - 1), 1), "last_y": round(sy(s["y"][-1]), 1)})
    xt = _nice_ticks(1, nmax, 6)
    return {"w": w, "h": h, "series": out, "x0": left, "x1": w - right, "y0": top, "y1": h - bottom,
            "yticks": [{"y": round(sy(t), 1), "label": f"{t:g}"} for t in ticks],
            "xticks": [{"x": round(sx(t - 1), 1), "label": f"{t:g}"} for t in xt if 1 <= t <= nmax],
            "x_label": x_label, "y_label": y_label, "zero_y": round(sy(0), 1) if lo < 0 < hi else None,
            "hline": {**hline, "yy": round(sy(hline["y"]), 1)} if hline else None}


def _smooth(v, k=10):
    out, acc = [], []
    for x in v:
        acc.append(x)
        acc = acc[-k:]
        out.append(sum(acc) / len(acc))
    return out


def rl_run(request, pk):
    run = get_object_or_404(RLRun, pk=pk)
    ctx = {"run": run}
    res = run.results or {}
    if run.status == RLRun.Status.READY and res:
        pol = res["policies"]
        rows = []
        best = {k: max(pol[p][k] for p in pol) for k in ("f1", "pr_auc", "recall_at_1pct", "reward_all")}
        for key in [k for k in POLICY_ORDER if k in pol]:
            p = pol[key]
            rows.append({"key": key, "label": POLICY_LABELS[key], "color": POLICY_COLORS[key], **p,
                         "best": {k: p[k] == best[k] for k in best}})
        tr = res["traces"]
        cum = _line_chart([{"key": k, "label": POLICY_LABELS[k], "color": POLICY_COLORS[k], "dash": POLICY_DASH[k],
                            "y": tr[k]["cum_reward_all"]} for k in POLICY_ORDER if k in tr],
                          x_label="Test window", y_label="Cumulative reward (all labels)", zero=True)
        caught = _line_chart([{"key": k, "label": POLICY_LABELS[k], "color": POLICY_COLORS[k], "dash": POLICY_DASH[k],
                               "y": tr[k]["cum_tp"]} for k in POLICY_ORDER if k in tr],
                             x_label="Test window", y_label="Frauds caught (cumulative)", zero=True)
        alarms = _line_chart([{"key": k, "label": POLICY_LABELS[k], "color": POLICY_COLORS[k], "dash": POLICY_DASH[k],
                               "y": tr[k]["cum_fp"]} for k in POLICY_ORDER if k in tr],
                             x_label="Test window", y_label="False alarms (cumulative)", zero=True)
        theta0 = res["semi_supervised"]["report"]["semi"]["threshold_logit"]
        decisions = _line_chart([
            {"key": "theta", "label": "Threshold offset (log-odds)", "color": "var(--m-gnn)", "dash": "",
             "y": [round(t - theta0, 3) for t in tr["agent"]["theta"]]},
            {"key": "w", "label": "Label-propagation weight", "color": "var(--g-account)", "dash": "6 3", "y": tr["agent"]["w"]},
        ], x_label="Test window", y_label="Agent setting", zero=True, h=220)
        tc = res["training"]
        learn = _line_chart([
            {"key": "agent", "label": "RL agent (10-episode average)", "color": "var(--m-gnn)", "dash": "",
             "y": [round(x, 3) for x in _smooth([e["reward"] for e in tc["agent"]])]},
            {"key": "no_gnn", "label": "RL agent without the GNN", "color": "var(--g-account)", "dash": "8 3 2 3",
             "y": [round(x, 3) for x in _smooth([e["reward"] for e in tc["no_gnn"]])]},
        ], x_label="Training episode", y_label="Episode reward (validation)", zero=True, w=1100, h=300,
            hline={"y": tc["static_val_reward"], "label": "Best fixed setting"})
        eps_end = next((e["episode"] for e in tc["agent"] if e["epsilon"] < 1.0), None)
        usage = res["action_usage"]
        tot = max(sum(usage["theta_offset"].values()), 1)

        def bars(d, fmt):
            return [{"label": fmt(k), "n": v, "w": v / tot * 100} for k, v in d.items()]
        ss = res["semi_supervised"]
        semi_rows = []
        for key, label in [("supervised", f"Supervised only ({res['params']['label_rate']:.0%} of labels)"),
                           ("semi", f"Semi-supervised (self-training, round {ss['best_round'] + 1} kept)"),
                           ("features", "Semi-supervised, links removed (features only)"),
                           ("reference", "Reference: every label")]:
            if key in ss["report"]:
                semi_rows.append({"key": key, "label": label, **ss["report"][key]})
        ctx.update({
            "res": res, "rows": rows, "cum": cum, "caught": caught, "alarms": alarms, "decisions": decisions,
            "learn": learn, "eps_end": eps_end, "semi_rows": semi_rows, "history": ss["history"],
            "usage_theta": bars(usage["theta_offset"], lambda k: f"{float(k):+g}"),
            "usage_w": bars(usage["lp_weight"], lambda k: f"{float(k):g}"),
            "usage_pl": bars(usage["pseudo_conf"], lambda k: "off" if k == "None" else f"p ≥ {float(k):g}"),
            "static_label": (f"threshold {res['static']['theta_offset']:+g}, LP weight {res['static']['lp_weight']:g}, "
                             f"pseudo-labels {'off' if res['static']['pseudo_conf'] is None else 'p ≥ ' + str(res['static']['pseudo_conf'])}"),
            "agent_vs_fixed": pol["agent"]["reward_all"] - pol["fixed"]["reward_all"],
            "agent_vs_static": pol["agent"]["reward_all"] - pol["static"]["reward_all"],
        })
    return render(request, "detector/rl_run.html", ctx)
