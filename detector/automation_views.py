"""Automated detection: the GNN with the RL agent deciding every transaction of a labelled card file."""
import io
import shutil
from pathlib import Path

import pandas as pd
from django import forms
from django.contrib import messages
from django.core.files.base import ContentFile
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .ml.fraud_automation import DECISION_LABELS
from .ml.transfer_target import SOURCE_NAMES
from .models import AutomationRun
from .rl_views import _line_chart
from .tasks import start_automation

# zero_shot / fine_tuned only appear in runs made before transfer learning was removed from the prototype
BASE_LABELS = {"zero_shot": "Zero-shot", "fine_tuned": "Fine-tuned", "target_only": "GNN trained on the revealed labels"}


def _base_label(key, src):
    if key == "zero_shot":
        return f"Zero-shot: {SOURCE_NAMES.get(src, src)} GNN, unchanged"
    if key == "fine_tuned":
        return f"Fine-tuned from {SOURCE_NAMES.get(src, src)}"
    return BASE_LABELS[key]
POLICY_LABELS = {"fixed": "GNN, fixed threshold", "static": "Best fixed setting", "agent": "RL agent",
                 "online": "RL agent, learning online"}
DECISION_ORDER = ["block_case", "escalate", "held_back", "approve"]


class AutomationForm(forms.Form):
    file = forms.FileField(label="Target transactions (CSV)",
                           help_text="The synthetic South African schema, with the Class column (used for the "
                                     "delayed feedback and for evaluation).")
    name = forms.CharField(max_length=200, required=False)
    inject_rings = forms.BooleanField(required=False, label="Plant six known fraud rings in the file first",
                                      help_text="Shared-device takeover, card testing and collusive-merchant rings in "
                                                "the training, validation and test periods. The page then scores how "
                                                "many ring transactions the agent grouped into ring cases.")
    ring_seed = forms.IntegerField(required=False, initial=2026, min_value=0, label="Ring seed")
    inject_adversarial = forms.BooleanField(
        required=False, label="Plant adversarial examples (evasion test)",
        help_text="Five groups of frauds built to evade the GNN: a naive control, full mimicry of the victim, low and "
                  "slow, graph camouflage, and a white-box attack that queries the GNN until it stops flagging them. "
                  "Adds about a minute.")
    adversarial_training = forms.BooleanField(
        required=False, label="Adversarial training",
        help_text="Also plant the attacks in the training period, then let the attacker re-optimise against the "
                  "retrained GNN. Adds about two minutes.")

    def clean_file(self):
        f = self.cleaned_data["file"]
        if not f.name.lower().endswith(".csv"):
            raise forms.ValidationError("Upload a .csv file.")
        return f

    def clean(self):
        c = super().clean()
        if c.get("inject_rings") and c.get("file"):
            from .ml.fraud_rings import RingError, inject_rings
            try:
                df = pd.read_csv(c["file"], low_memory=False)
                out, summary = inject_rings(df, seed=c.get("ring_seed") if c.get("ring_seed") is not None else 2026)
            except RingError as e:
                raise forms.ValidationError(str(e))
            except (ValueError, pd.errors.ParserError) as e:
                raise forms.ValidationError(f"Could not read the file: {e}")
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            c["file"] = ContentFile(buf.getvalue().encode(), name=Path(c["file"].name).stem + "_rings.csv")
            c["rings"] = _summary_records(summary)
        return c


def _summary_records(summary: pd.DataFrame) -> list[dict]:
    from .ml.fraud_rings import PATTERN_LABELS
    return [{"ring": rid, "pattern_label": PATTERN_LABELS.get(r["pattern"], r["pattern"]),
             **{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}} for rid, r in summary.iterrows()]


def automation_overview(request):
    form = AutomationForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        c = form.cleaned_data
        params = {}
        if c.get("rings"):
            params["rings"] = c["rings"]
        if c.get("inject_adversarial") or c.get("adversarial_training"):
            params["adversarial"] = {"training": bool(c.get("adversarial_training")), "seed": 7}
        run = AutomationRun.objects.create(name=c["name"] or Path(c["file"].name).stem, file=c["file"], params=params)
        start_automation(run.pk)
        messages.success(request, f"Started {run.name}.")
        return redirect(run)
    return render(request, "detector/automation.html", {"form": form, "runs": AutomationRun.objects.all()[:50]})


def automation_status(request, pk):
    r = get_object_or_404(AutomationRun, pk=pk)
    return JsonResponse({"status": r.status, "progress": r.progress, "message": r.progress_message, "error": r.error})


@require_POST
def automation_delete(request, pk):
    r = get_object_or_404(AutomationRun, pk=pk)
    shutil.rmtree(r.run_dir, ignore_errors=True)
    r.delete()
    messages.success(request, f"Deleted {r.name}.")
    return redirect("automation")


def automation_decisions(request, pk):
    """The decision log as CSV; ?decision=held_back (or block_case, escalate, approve, ring) filters it."""
    r = get_object_or_404(AutomationRun, pk=pk)
    p = r.run_dir / "decisions.csv"
    if not p.exists():
        raise Http404
    df = pd.read_csv(p, low_memory=False)
    d = request.GET.get("decision")
    if d == "ring":
        df = df[df["case_type"] == "ring"]
    elif d == "ring_linked":
        df = df[df["ring_linked"].notna()]
    elif d == "adversarial" and "adversarial_group" in df:
        df = df[df["adversarial_group"].notna() & (df["adversarial_group"].astype(str) != "")]
    elif d and d[:1] in "AC" and d[1:].isdigit() and "adversarial_group" in df:
        df = df[df["adversarial_group"] == d]
    elif d == "known_ring" and "known_ring" in df:
        df = df[df["known_ring"].notna() & (df["known_ring"].astype(str) != "")]
    elif d and d.startswith("R") and "known_ring" in df:
        df = df[df["known_ring"] == d]
    elif d in DECISION_LABELS:
        df = df[df["decision"] == d]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    resp = HttpResponse(buf.getvalue(), content_type="text/csv")
    resp["Content-Disposition"] = f'attachment; filename="automation_{pk}_{d or "all"}.csv"'
    return resp


def automation_file(request, pk):
    """The target file the run used (with the planted rings, when it has them)."""
    r = get_object_or_404(AutomationRun, pk=pk)
    if not r.file or not Path(r.file.path).exists():
        raise Http404
    return FileResponse(open(r.file.path, "rb"), as_attachment=True, filename=Path(r.file.name).name)


def automation_run(request, pk):
    run = get_object_or_404(AutomationRun, pk=pk)
    ctx = {"run": run}
    res = run.results or {}
    if run.status != "ready" or not res:
        return render(request, "detector/automation_run.html", ctx)
    a = res["automation"]
    dep = res["deployed"]
    fun = {f["decision"]: f for f in a["funnel"]}
    test_frauds = res["target"]["test_frauds"]
    tiles = [{**fun[d], "key": d} for d in DECISION_ORDER]

    comp = []
    for b, v in res["bases"].items():
        src = v.get("source")
        for p in ("fixed", "static", "agent", "online"):
            m = v["policies"][p]
            comp.append({"base": _base_label(b, src), "base_key": b, "policy": POLICY_LABELS[p], "first": p == "fixed",
                         **m, "val_reward": v["agent_val_reward"] if p in ("agent", "online") else
                         (v["static_val_reward"] if p == "static" else None)})
            if b == dep and p == "online":
                comp.append({"base": "", "base_key": b, "policy": "Deployed: the best of the agents, learning online",
                             "deployed": True, **a["metrics"], "val_reward": None})
    fixed = res["bases"][dep]["policies"]["fixed"]

    tl = a["timeline"]
    n = len(tl["theta_offset"])
    cum = lambda xs: [sum(xs[:k + 1]) for k in range(len(xs))]  # noqa: E731
    decisions = _line_chart([
        {"key": "theta", "label": "Threshold offset set by the agent (log-odds)", "color": "var(--m-gnn)", "dash": "", "y": tl["theta_offset"]},
        {"key": "w", "label": "Graph-evidence weight", "color": "var(--rust)", "dash": "5 4", "y": tl["w"]}],
        x_label="Test window (2 h each)", y_label="Agent's setting", zero=True)
    outcomes = _line_chart([
        {"key": "tp", "label": "Frauds escalated", "color": "var(--m-gnn)", "dash": "", "y": cum(tl["tp"])},
        {"key": "fp", "label": "False alarms", "color": "var(--rust)", "dash": "5 4", "y": cum(tl["fp"])},
        {"key": "fn", "label": "Frauds not escalated", "color": "var(--muted)", "dash": "2 3", "y": cum(tl["fn"])}],
        x_label="Test window (2 h each)", y_label="Cumulative count", zero=True)
    ctx.update({"res": res, "a": a, "tiles": tiles, "comp": comp, "dep_label": BASE_LABELS[dep], "fixed": fixed,
                "dep_name": _base_label(dep, res["bases"][dep].get("source")),
                "dep_source": SOURCE_NAMES.get(res["bases"][dep].get("source"), res["bases"][dep].get("source")),
                "test_frauds": test_frauds, "decisions_chart": decisions, "outcomes_chart": outcomes, "n_windows": n,
                "rings": [c for c in a["cases"] if c["type"] == "ring"],
                "bursts": [c for c in a["cases"] if c["type"] != "ring"],
                "dep": res["bases"][dep], "decision_labels": DECISION_LABELS,
                **_known_ring_context(run, a), **_adversarial_context(res, a)})
    return render(request, "detector/automation_run.html", ctx)


RING_DECISION_COLOURS = {"block_case": "var(--fraud)", "escalate": "var(--orange)", "held_back": "var(--gold)",
                         "approve": "var(--panel)"}
RING_NODE_COLOURS = {"device": "var(--g-device)", "ip": "var(--g-ip)", "merchant": "var(--ink)"}


def _known_ring_context(run, a) -> dict:
    """The planted rings: what the agent could learn from (training, validation) and how it handled the test ones."""
    kr = a.get("known_rings")
    if not kr:
        return {}
    summary = {r["ring"]: r for r in run.params.get("rings") or []}
    learn, scored = [], []
    for r in kr:
        r = {**summary.get(r["ring"], {}), **r}
        if r.get("scored"):
            net = r["network"]
            for n in net["nodes"]:
                n["colour"] = (RING_DECISION_COLOURS.get(n["decision"], "var(--panel)") if n["kind"] == "account"
                               else RING_NODE_COLOURS[n["kind"]])
            r["caught"] = r["outcome"] == "caught as one ring case"
            scored.append(r)
        else:
            learn.append(r)
    tot = sum(r["transactions"] for r in scored)
    grouped = sum(r["in_ring_case"] for r in scored)
    return {"known_learn": learn, "known_scored": scored, "known_count": len(kr), "known_total": tot, "known_grouped": grouped,
            "known_escalated": sum(r["escalated"] for r in scored)}


EDIT_LABELS_VIEW = {"country": "home country", "hour": "usual hours", "amount": "normal-range amount",
                    "merchant": "known merchant", "velocity": "one a day", "device": "victim's own device"}


def _adversarial_context(res, a) -> dict:
    adv = a.get("adversarial")
    if not adv:
        return {}
    from .ml.adversarial import GROUP_LABELS
    setup = res.get("adversarial_setup") or {}
    groups = []
    for gr in adv["groups"]:
        groups.append({**gr, "gnn_pct": 100 * gr["gnn_recall"], "agent_pct": 100 * gr["agent_recall"],
                       "value_through": gr["value"] - gr["value_escalated"]})
    paths = []
    for p in setup.get("paths", []):
        steps = []
        for st in p["steps"]:
            steps.append({"label": "no edit" if not st["added"] else "+ " + EDIT_LABELS_VIEW.get(st["added"], st["added"]),
                          "pct": 100 * st["flagged"], "n": st["n"],
                          "trials": [{"label": EDIT_LABELS_VIEW.get(t["edit"], t["edit"]), "pct": 100 * t["flagged"]}
                                     for t in st["trials"]]})
        paths.append({"against": p["against"], "steps": steps, "evaded": p["evaded"],
                      "edits": [EDIT_LABELS_VIEW.get(e, e) for e in p["edits"]]})
    train_groups = [g for g in setup.get("groups", []) if g["period"] == "train"]
    return {"adv": adv, "adv_groups": groups, "adv_paths": paths, "adv_setup": setup,
            "adv_train_groups": train_groups, "adv_group_labels": GROUP_LABELS,
            "adv_lowered_pct": 100 * adv["windows_threshold_lowered"]}
