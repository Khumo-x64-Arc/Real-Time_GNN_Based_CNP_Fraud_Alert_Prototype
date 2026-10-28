import json
import math
import shutil
import time
from pathlib import Path

import pandas as pd
from django.contrib import messages
from django.http import Http404, HttpResponseBadRequest, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from django.conf import settings

from .alerts import record_alert
from .customer import record_customer_tx
from .forms import UploadForm
from .ml.engine import LiveEngine, api_engine
from .ml.schema import describe_problems, detect_schema
from .models import Alert, Dataset, ModelResult
from .tasks import start_training

MODEL_COLORS = {"GNN": "var(--m-gnn)", "GNN live": "var(--m-gnn-live)"}
MODEL_LABELS = {"GNN": "GNN · full graph", "GNN live": "GNN · live (streaming)"}
METRICS = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1", "F1"), ("accuracy", "Accuracy"),
           ("precision", "Precision"), ("recall", "Recall")]


# ------------------------------------------------------------------ datasets
def dataset_list(request):
    datasets = Dataset.objects.prefetch_related("model_results")
    return render(request, "detector/dataset_list.html", {"datasets": datasets})


def upload(request):
    form = UploadForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        f = form.cleaned_data["file"]
        ds = Dataset.objects.create(name=form.cleaned_data["name"] or Path(f.name).stem, file=f)
        try:
            sample = pd.read_csv(ds.file.path, nrows=5000, low_memory=False)
        except Exception as e:
            ds.status, ds.error = Dataset.Status.UNUSABLE, f"The file couldn't be read as CSV ({e})."
            ds.save()
            return redirect(ds)
        schema = detect_schema(sample)
        ds.columns = list(sample.columns)
        ds.schema = {k: v for k, v in schema.items() if v and k != "label_problem"}
        problems = describe_problems(sample, schema)
        # the fraud-count check needs the whole file, so only block on structural problems here
        blocking = [p for p in problems if not p.startswith("Only ")]
        if blocking:
            ds.status, ds.error = Dataset.Status.UNUSABLE, " ".join(blocking)
            ds.save()
        else:
            ds.save()
            start_training(ds.pk)
            messages.success(request, f"Uploaded {ds.name}. Training has started.")
        return redirect(ds)
    return render(request, "detector/upload.html", {"form": form})


def _roc_chart(results):
    """ROC curves on a log-scaled FPR axis: fraud models only differ at FPR below 1%."""
    W, H, left, bottom, top, right = 470, 330, 58, 40, 12, 14
    lo = -4

    def sx(fpr):
        v = (math.log10(max(fpr, 10 ** lo)) - lo) / -lo
        return left + v * (W - left - right)

    def sy(tpr):
        return top + (1 - tpr) * (H - top - bottom)

    series = [{"name": r.name, "label": MODEL_LABELS.get(r.name, r.name), "color": MODEL_COLORS.get(r.name), "auc": r.roc_auc,
               "points": " ".join(f"{sx(f):.1f},{sy(t):.1f}" for f, t in r.roc_curve)} for r in results]
    return {"w": W, "h": H, "series": series,
            "xticks": [{"x": round(sx(10 ** e), 1), "label": f"{10 ** e:g}"} for e in range(lo, 1)],
            "yticks": [{"y": round(sy(v), 1), "label": f"{v:g}"} for v in (0, 0.25, 0.5, 0.75, 1.0)],
            "x0": left, "x1": W - right, "y0": top, "y1": H - bottom, "xlabel_y": H - 6}


def _gnn_results(ds):
    """The dataset's GNN results in display order (older trainings also stored other models; they're skipped)."""
    return sorted((r for r in ds.model_results.all() if r.name in ModelResult.ORDER),
                  key=lambda r: ModelResult.ORDER.index(r.name))


def _ablation_rows(rows):
    """Bar geometry for the graph-ablation table: change in PR-AUC, negative = the GNN needed those links."""
    m = max([abs(r["d_pr_auc"]) for r in rows] or [0]) or 1
    return [{**r, "w": abs(r["d_pr_auc"]) / m * 50, "pos": r["d_pr_auc"] > 0} for r in rows]


def dataset_detail(request, pk):
    ds = get_object_or_404(Dataset, pk=pk)
    ctx = {"ds": ds}
    if ds.status == Dataset.Status.READY and ds.results:
        results = _gnn_results(ds)
        rows = []
        for r in results:
            neg, pos = r.tn + r.fp, r.tp + r.fn
            rows.append({
                "r": r, "color": MODEL_COLORS.get(r.name), "label": MODEL_LABELS.get(r.name, r.name),
                "cells": [{"v": getattr(r, k)} for k, _ in METRICS],
                "cm": {k: getattr(r, k) / max(r.tp, r.fp, r.tn, r.fn, 1) for k in ("tp", "fp", "tn", "fn")},
                "fpr": r.fp / neg if neg else 0, "caught": r.tp / pos if pos else 0,
            })
        res = ds.results
        test_n = res["sizes"]["test"]
        sweep = res.get("sweep") or {}
        ctx.update({
            "rows": rows, "metrics": METRICS, "res": res, "roc": _roc_chart(results),
            "null_accuracy": 1 - res["test_frauds"] / test_n if test_n else 0,
            "ablation": _ablation_rows(res.get("ablation") or []),
            "sweep": sweep.get("rows", []), "sweep_label": MODEL_LABELS.get(sweep.get("model"), ""),
            "legacy": not ds.is_gnn_only, "alert_count": ds.alerts.count(),
        })
    return render(request, "detector/dataset_detail.html", ctx)


@require_GET
def dataset_status(request, pk):
    ds = get_object_or_404(Dataset, pk=pk)
    return JsonResponse({"status": ds.status, "progress": ds.progress, "message": ds.progress_message,
                         "error": ds.error})


@require_POST
def dataset_retrain(request, pk):
    ds = get_object_or_404(Dataset, pk=pk)
    if ds.status == Dataset.Status.TRAINING:
        messages.info(request, "This dataset is already training.")
    else:
        Dataset.objects.filter(pk=pk).update(status=Dataset.Status.QUEUED, progress=0, error="")
        start_training(ds.pk)
        messages.success(request, "Retraining started.")
    return redirect(ds)


@require_POST
def dataset_delete(request, pk):
    ds = get_object_or_404(Dataset, pk=pk)
    shutil.rmtree(ds.model_dir, ignore_errors=True)
    ds.file.delete(save=False)
    name = ds.name
    ds.delete()
    messages.success(request, f"Deleted {name} and its results.")
    return redirect("dataset_list")


# ------------------------------------------------------------------ live monitor
def monitor(request, pk=None):
    ready = [d for d in Dataset.objects.filter(status=Dataset.Status.READY) if d.is_live_ready]
    if pk is None:
        if ready:
            return redirect("monitor_dataset", pk=ready[0].pk)
        return render(request, "detector/monitor.html", {"ds": None, "ready": ready})
    ds = get_object_or_404(Dataset, pk=pk)
    if not ds.is_live_ready:
        messages.error(request, f"{ds.name} has no GNN trained by this version yet. Retrain it first."
                       if ds.status == Dataset.Status.READY else f"{ds.name} has no trained GNN yet.")
        return redirect(ds)
    return render(request, "detector/monitor.html", {
        "ds": ds, "ready": ready, "replay_rows": ds.results.get("replay_rows", 0),
        "thresholds": {"GNN": ds.results.get("live_threshold", 0.5)},
        "live_result": ds.gnn_result, "live_label": MODEL_LABELS.get(ds.results.get("live_model"), "GNN"),
    })


def _event_row(ds, row, result, alert):
    sc = ds.schema

    def g(role):
        v = row.get(sc.get(role)) if sc.get(role) else None
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else v

    lab = g("label")
    return {
        "id": str(g("tx_id") or ""), "time": str(g("time") or ""), "amount": g("amount"),
        "account": str(g("account") or ""), "merchant": str(g("merchant_name") or g("merchant") or ""),
        "category": str(g("category") or ""), "country": str(g("country") or ""),
        "channel": str(g("channel") or ""),
        "level": result["level"], "probs": result["probabilities"], "flagged": result["flagged"],
        "signals": result["signals"], "actual": int(lab) if lab is not None else None,
        "alert_id": alert.pk if alert else None,
        "why": (result.get("explanation") or {}).get("summary", ""),
        "drivers": [{"label": d["label"].split(" (")[0], "value": d["value"], "lo": d["delta"]}
                    for d in (result.get("explanation") or {}).get("drivers", [])[:3]],
    }


@require_GET
def monitor_stream(request, pk):
    """Server-Sent Events: replay the unseen test period through the live engine."""
    ds = get_object_or_404(Dataset, pk=pk)
    if not ds.is_live_ready:
        raise Http404
    try:
        rate = max(1, min(int(request.GET.get("rate", 4)), 200))  # transactions per second
        start = max(0, int(request.GET.get("start", 0)))
    except ValueError:
        return HttpResponseBadRequest("rate and start must be integers")

    def gen():
        engine = LiveEngine(ds.bundle_path)  # fresh history per replay, so replays are repeatable
        rows = pd.read_pickle(ds.model_dir / "replay.pkl").to_dict("records")
        s = min(start, len(rows))
        for i in range(0, s, 500):  # fast-forward silently so behaviour profiles stay correct
            engine.score_batch(rows[i:min(i + 500, s)])
        yield f"event: meta\ndata: {json.dumps({'total': len(rows), 'start': s})}\n\n"
        tick = 0.5
        per_tick = max(1, round(rate * tick))
        i = s
        while i < len(rows):
            t0 = time.time()
            batch = rows[i:i + per_tick]
            out = []
            for row, res in zip(batch, engine.score_batch(batch)):
                alert = record_alert(ds, row, res, source="monitor")
                record_customer_tx(ds, row, res, alert)
                out.append(_event_row(ds, row, res, alert))
            i += len(batch)
            yield f"event: tx\ndata: {json.dumps({'pos': i, 'rows': out}, default=str)}\n\n"
            time.sleep(max(0.0, tick - (time.time() - t0)))
        yield "event: done\ndata: {}\n\n"

    resp = StreamingHttpResponse(gen(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    return resp


# ------------------------------------------------------------------ alerts
def alert_list(request):
    qs = Alert.objects.select_related("dataset")
    f = {k: request.GET.get(k, "") for k in ("dataset", "level", "review")}
    if f["dataset"]:
        qs = qs.filter(dataset_id=f["dataset"])
    if f["level"]:
        qs = qs.filter(level=f["level"])
    if f["review"]:
        qs = qs.filter(review=f["review"])
    return render(request, "detector/alerts.html", {
        "alerts": qs[:300], "total": qs.count(), "filters": f, "datasets": Dataset.objects.all(),
        "levels": Alert.Level.choices, "reviews": Alert.Review.choices,
    })


@require_POST
def alert_review(request, pk):
    alert = get_object_or_404(Alert, pk=pk)
    value = request.POST.get("review")
    if value not in Alert.Review.values:
        return HttpResponseBadRequest("Unknown review value")
    alert.review = value
    alert.save(update_fields=["review"])
    if request.headers.get("x-requested-with") == "fetch":
        return JsonResponse({"ok": True, "review": alert.review, "label": alert.get_review_display()})
    return redirect(request.POST.get("next") or "alert_list")


@require_POST
def alerts_clear(request):
    ds_id = request.POST.get("dataset")
    qs = Alert.objects.filter(dataset_id=ds_id) if ds_id else Alert.objects.all()
    n = qs.count()
    qs.delete()
    messages.success(request, f"Cleared {n} alerts.")
    return redirect("alert_list")


@require_GET
def api_latest_alerts(request):
    """Polled by every page so a new alert pops up wherever the user is."""
    try:
        after = int(request.GET.get("after", 0) or 0)
    except ValueError:
        after = 0
    qs = Alert.objects.select_related("dataset").order_by("-id")
    if not after:
        last = qs.first()
        return JsonResponse({"last_id": last.pk if last else 0, "alerts": [],
                             "open": Alert.objects.filter(review="open").count()})
    new = list(qs.filter(pk__gt=after)[:10])
    return JsonResponse({
        "last_id": new[0].pk if new else after,
        "open": Alert.objects.filter(review="open").count(),
        "alerts": [{"id": a.pk, "level": a.level, "level_label": a.get_level_display(),
                    "tx": a.transaction_id, "amount": a.amount, "merchant": a.merchant,
                    "dataset": a.dataset.name, "score": a.score,
                    "reason": a.signals[0]["text"] if a.signals else "High model score"} for a in new],
    })


# ------------------------------------------------------------------ scoring API
@csrf_exempt
@require_POST
def api_score(request, pk):
    """POST {"transaction": {...}} or {"transactions": [...]} using the dataset's column names."""
    key = getattr(settings, "SCORING_API_KEY", "")
    if key and request.headers.get("X-Api-Key") != key and not request.user.is_staff:
        return JsonResponse({"error": "Missing or wrong X-Api-Key header."}, status=401)
    ds = get_object_or_404(Dataset, pk=pk)
    if not ds.is_live_ready:
        return JsonResponse({"error": "This dataset has no GNN trained by this version yet. Retrain it."}, status=409)
    try:
        body = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Body must be JSON."}, status=400)
    txs = body.get("transactions") or ([body["transaction"]] if "transaction" in body else None)
    if not txs or not all(isinstance(t, dict) for t in txs):
        return JsonResponse({"error": 'Send {"transaction": {...}} or {"transactions": [{...}]}.'}, status=400)
    engine = api_engine(ds.pk, ds.bundle_path)
    out = []
    for tx, res in zip(txs, engine.score_batch(txs)):
        alert = record_alert(ds, tx, res, source="api")
        record_customer_tx(ds, tx, res, alert)
        out.append({**res, "alert_id": alert.pk if alert else None})
    return JsonResponse({"results": out})


# ------------------------------------------------------------------ results overview
RESULT_METRICS = METRICS  # ROC-AUC, PR-AUC, F1, Accuracy, Precision, Recall


def _ordered(results):
    return sorted((r for r in results if r.name in ModelResult.ORDER), key=lambda r: ModelResult.ORDER.index(r.name))


LINE_DASH = {"GNN": "", "GNN live": "6 4"}


def _chart_payload(selected, results, cross):
    """Everything charts.js draws, as one JSON blob (rendered with json_script)."""
    res = selected.results or {}
    extra = res.get("charts") or {}
    per_model = extra.get("models", {})
    models = []
    for r in results:
        e = per_model.get(r.name, {})
        neg = r.tn + r.fp
        models.append({
            "name": MODEL_LABELS.get(r.name, r.name), "color": MODEL_COLORS.get(r.name, "var(--muted)"), "dash": LINE_DASH.get(r.name, ""),
            "roc": r.roc_curve, "auc": r.roc_auc, "pr": e.get("pr_curve"), "ap": r.pr_auc, "thr": r.threshold,
            "tp": r.tp, "fp": r.fp, "fn": r.fn, "tn": r.tn,
            "op_roc": [r.fp / neg if neg else 0, r.recall], "op_pr": [r.recall, r.precision],
            "hist_fraud": e.get("hist_fraud"), "hist_legit": e.get("hist_legit"),
        })
    training = extra.get("training")
    if not training and res.get("gnn_history"):  # datasets trained before charts existed still kept the GNN's history
        training = {"GNN": {"unit": "epoch", "points": [[h["epoch"], h["val_ap"]] for h in res["gnn_history"]]}}
    training = {MODEL_LABELS["GNN"]: v for k, v in (training or {}).items() if k == "GNN"}  # only the GNN trains
    sizes = res.get("sizes", {})
    return {
        "models": models, "training": training, "bins": extra.get("hist_bins", 20),
        "base_rate": res.get("test_frauds", 0) / sizes["test"] if sizes.get("test") else res.get("fraud_rate", 0),
        "cross": [{"dataset": b["ds"].name, "values": {MODEL_LABELS[row["r"].name]: row["r"].f1 for row in b["rows"]}}
                  for b in cross] if len(cross) > 1 else [],
        "model_order": [MODEL_LABELS[m] for m in ModelResult.ORDER],
        "colors": {MODEL_LABELS[m]: c for m, c in MODEL_COLORS.items()},
        "dashes": {MODEL_LABELS[m]: d for m, d in LINE_DASH.items()},
    }


def results_overview(request):
    ready = list(Dataset.objects.filter(status=Dataset.Status.READY).prefetch_related("model_results"))
    ctx = {"ready": ready, "metrics": RESULT_METRICS}
    if not ready:
        return render(request, "detector/results.html", ctx)
    try:
        selected = next(d for d in ready if str(d.pk) == request.GET.get("dataset"))
    except StopIteration:
        selected = ready[0]
    results = _ordered(selected.model_results.all())
    if not results:
        return render(request, "detector/results.html", ctx)

    cards = []
    for r in results:
        pos, neg = r.tp + r.fn, r.tn + r.fp
        cards.append({
            "r": r, "color": MODEL_COLORS.get(r.name), "label": MODEL_LABELS.get(r.name, r.name),
            "metrics": [{"label": label, "v": getattr(r, k)} for k, label in RESULT_METRICS],
            "caught": r.tp / pos if pos else 0, "fpr": r.fp / neg if neg else 0,
        })
    by = {r.name: r for r in results}
    gap = None
    if "GNN" in by and "GNN live" in by:   # what scoring without future neighbours costs
        gap = [{"label": label, "full": getattr(by["GNN"], k), "live": getattr(by["GNN live"], k),
                "d": getattr(by["GNN live"], k) - getattr(by["GNN"], k)} for k, label in RESULT_METRICS]

    # every trained dataset side by side, so the GNN can be compared across uploads
    cross = []
    for d in ready:
        rs = _ordered(d.model_results.all())
        if rs:
            cross.append({"ds": d, "legacy": not d.is_gnn_only,
                          "rows": [{"r": r, "color": MODEL_COLORS.get(r.name), "label": MODEL_LABELS.get(r.name, r.name)}
                                   for r in rs]})

    ctx.update({"selected": selected, "cards": cards, "cross": cross, "gap": gap,
                "res": selected.results, "chart_data": _chart_payload(selected, results, cross),
                "legacy": not selected.is_gnn_only,
                "graph_ready": (selected.model_dir / "graph.bin.gz").exists(),
                **_oversampling_context(selected.results), "os_study": _oversampling_study()})
    return render(request, "detector/results.html", ctx)


OVERSAMPLING_ROWS = [("roc_auc", "ROC-AUC", "num"), ("pr_auc", "PR-AUC", "num"), ("f1", "F1", "num"),
                     ("precision", "Precision", "num"), ("recall", "Recall", "num"), ("accuracy", "Accuracy", "num"),
                     ("tp", "Frauds caught", "int"), ("fp", "False alarms", "int"), ("fn", "Frauds missed", "int")]
LOWER_IS_BETTER = {"fp", "fn"}


def _oversampling_context(res: dict) -> dict:
    """Side-by-side numbers for the selected dataset: the GNN with and without FSMOTE, full graph and live."""
    block = (res or {}).get("oversampling")
    if not block:
        return {"os_state": "missing"}
    if block.get("status") in ("running", "failed"):
        return {"os_state": block["status"], "os_block": block}
    w, wo = block.get("with") or {}, block.get("without") or {}
    evals = [m for m in ("GNN", "GNN live") if m in (w.get("models") or {}) and m in (wo.get("models") or {})]
    tables = []
    for m in evals:
        a, b = w["models"][m], wo["models"][m]
        rows = []
        for k, label, kind in OVERSAMPLING_ROWS:
            va, vb = a.get(k), b.get(k)
            if va is None or vb is None:
                continue
            d = vb - va
            better = None
            if abs(d) > (0 if kind == "int" else 0.0005):
                better = "without" if (d < 0 if k in LOWER_IS_BETTER else d > 0) else "with"
            rows.append({"label": label, "kind": kind, "with": va, "without": vb, "d": d, "better": better})
        tables.append({"key": m, "label": MODEL_LABELS.get(m, m), "rows": rows,
                       "thr_with": a.get("threshold"), "thr_without": b.get("threshold")})
    fs = w.get("fsmote") or {}
    return {"os_state": "ready", "os_block": block, "os_tables": tables, "os_main": block.get("main"),
            "os_fs": fs, "os_with": w, "os_without": wo}


def _oversampling_study():
    """The study's cycle-4 ablation (semi-supervised GNN + RL agent, three datasets), shipped with the app."""
    from pathlib import Path
    f = Path(__file__).parent / "data" / "oversampling_study.json"
    try:
        st = json.loads(f.read_text())
    except (OSError, ValueError):
        return None

    def cell(v, lower=False):
        d = v["without"] - v["with"]
        better = None
        if abs(d) > 0.0005:
            better = "without" if (d < 0 if lower else d > 0) else "with"
        return {**v, "d": d, "better": better}

    rows = [{"label": p["label"], "kind": "num", "cells": [cell(ds["f1"][p["key"]]) for ds in st["datasets"]]}
            for p in st["policies"]]
    rows.append({"label": "GNN PR-AUC", "kind": "num", "sep": True,
                 "cells": [cell(ds["gnn_pr_auc"]) for ds in st["datasets"]]})
    rows.append({"label": "False alarms, fixed threshold", "kind": "int",
                 "cells": [cell(ds["fixed_false_alarms"], lower=True) for ds in st["datasets"]]})
    return {**st, "rows": rows}


@require_POST
def dataset_oversampling(request, pk):
    from .tasks import start_oversampling_comparison
    ds = get_object_or_404(Dataset, pk=pk)
    if ds.status != Dataset.Status.READY:
        messages.error(request, "Train the dataset first.")
    elif ((ds.results or {}).get("oversampling") or {}).get("status") == "running":
        messages.info(request, "The comparison is already running.")
    else:
        start_oversampling_comparison(ds.pk)
        messages.success(request, f"Training {ds.name} with oversampling switched the other way. "
                                  "The comparison appears here when it is done.")
    return redirect(f"{reverse('results')}?dataset={ds.pk}#oversampling")


@require_GET
def results_export(request):
    """Every model's metrics for every trained dataset as a CSV file."""
    import csv
    from django.http import HttpResponse

    resp = HttpResponse(content_type="text/csv")
    resp["Content-Disposition"] = 'attachment; filename="khumo-gnn-results.csv"'
    w = csv.writer(resp)
    w.writerow(["dataset", "evaluation", "roc_auc", "pr_auc", "f1", "accuracy", "precision", "recall",
                "threshold", "tp", "fp", "tn", "fn", "train_seconds", "trained_at"])
    for d in Dataset.objects.filter(status=Dataset.Status.READY).prefetch_related("model_results"):
        for r in _ordered(d.model_results.all()):
            w.writerow([d.name, MODEL_LABELS.get(r.name, r.name), f"{r.roc_auc:.6f}", f"{r.pr_auc:.6f}", f"{r.f1:.6f}",
                        f"{r.accuracy:.6f}", f"{r.precision:.6f}", f"{r.recall:.6f}", f"{r.threshold:.6f}",
                        r.tp, r.fp, r.tn, r.fn, r.train_seconds,
                        d.trained_at.isoformat() if d.trained_at else ""])
        block = (d.results or {}).get("oversampling") or {}
        main = block.get("main")
        for key, tag in (("with", "with FSMOTE"), ("without", "without oversampling")):
            if key == main or not (block.get(key) or {}).get("models"):
                continue   # the main run's rows are already above
            for name, m in block[key]["models"].items():
                w.writerow([d.name, f"{MODEL_LABELS.get(name, name)} ({tag})", f"{m['roc_auc']:.6f}", f"{m['pr_auc']:.6f}",
                            f"{m['f1']:.6f}", f"{m['accuracy']:.6f}", f"{m['precision']:.6f}", f"{m['recall']:.6f}",
                            f"{m['threshold']:.6f}", m["tp"], m["fp"], m["tn"], m["fn"], m.get("train_seconds", ""),
                            block.get("computed_at", "")])
    return resp


# ------------------------------------------------------------------ transaction graph
def _graph_file(ds, name):
    """ds: a Dataset, or any object with `graph_dir` and `graph_ready`."""
    folder = getattr(ds, "graph_dir", None) or ds.model_dir
    ready = getattr(ds, "graph_ready", None)
    ready = (ds.status == Dataset.Status.READY) if ready is None else ready
    f = folder / name
    if not ready or not f.exists():
        raise Http404("No transaction graph has been built for this yet.")
    return f


def _gz_response(path, content_type):
    """The files are stored gzip-compressed; the browser unpacks them because of Content-Encoding."""
    from django.http import HttpResponse

    resp = HttpResponse(path.read_bytes(), content_type=content_type)
    resp["Content-Encoding"] = "gzip"
    resp["Cache-Control"] = "private, max-age=300"
    return resp


@require_GET
def graph_meta(request, pk):
    return _gz_response(_graph_file(get_object_or_404(Dataset, pk=pk), "graph.json.gz"), "application/json")


@require_GET
def graph_data(request, pk):
    return _gz_response(_graph_file(get_object_or_404(Dataset, pk=pk), "graph.bin.gz"), "application/octet-stream")


_graph_tx_cache: dict = {}


def _graph_tables(ds):
    """(meta, typed arrays, transaction info) for one dataset, cached by file time."""
    import gzip
    import numpy as np

    f = _graph_file(ds, "graph.bin.gz")
    key = (str(f), f.stat().st_mtime)
    if key not in _graph_tx_cache:
        meta = json.loads(gzip.decompress(_graph_file(ds, "graph.json.gz").read_bytes()))
        raw = gzip.decompress(f.read_bytes())
        arrs = {k: np.frombuffer(raw, dtype=v["dtype"], count=v["length"], offset=v["offset"])
                for k, v in meta["arrays"].items()}
        info = pd.read_pickle(_graph_file(ds, "graph_tx.pkl"))
        _graph_tx_cache.clear()  # keep one dataset in memory
        _graph_tx_cache[key] = (meta, arrs, info)
    return _graph_tx_cache[key]


def _plain(view):
    """Keep an undecorated handle (`__wrapped__`) so other endpoints can reuse the body."""
    wrapped = require_GET(view)
    wrapped.__wrapped__ = view
    return wrapped


@_plain
def graph_tx(request, pk, i, obj=None):
    ds = obj or get_object_or_404(Dataset, pk=pk)
    meta, arrs, info = _graph_tables(ds)
    if not 0 <= i < meta["n_tx"]:
        raise Http404("No such transaction.")
    T = len(meta["types"])
    ents = arrs["tx_ent"][i * T:(i + 1) * T]
    flags = int(arrs["tx_flags"][i])
    row = info.iloc[i]
    amount = row["amount"]
    return JsonResponse({
        "index": i, "id": row["id"], "time": row["time"],
        "amount": None if pd.isna(amount) else float(amount),
        "fraud": bool(flags & 1), "split": ["training", "validation", "test"][(flags >> 1) & 3],
        "gnn_score": int(arrs["tx_score"][i]) / 255 if meta.get("has_score") else None,
        "gnn_flagged": bool(flags & 8),
        "entities": [{"type": meta["types"][t], "label": meta["type_labels"][t], "index": int(e),
                      "name": meta["names"][int(e)]} for t, e in enumerate(ents) if e >= 0],
    })


@_plain
def graph_export(request, pk, obj=None):
    """Edge list (transaction, entity) for Gephi, NetworkX or a spreadsheet."""
    import csv

    ds = obj or get_object_or_404(Dataset, pk=pk)
    meta, arrs, info = _graph_tables(ds)
    T, types, names = len(meta["types"]), meta["types"], meta["names"]
    tx_ent, flags, ids = arrs["tx_ent"], arrs["tx_flags"], info["id"].to_numpy()

    class Echo:
        def write(self, v):
            return v

    def rows():
        w = csv.writer(Echo())
        yield w.writerow(["transaction_id", getattr(ds, "graph_flag_column", "is_fraud"), "split", "entity_type", "entity_id"])
        for i in range(meta["n_tx"]):
            f = int(flags[i])
            for t in range(T):
                e = int(tx_ent[i * T + t])
                if e >= 0:
                    yield w.writerow([ids[i], f & 1, ["train", "validation", "test"][(f >> 1) & 3], types[t], names[e]])

    resp = StreamingHttpResponse(rows(), content_type="text/csv")
    resp["Content-Disposition"] = f'attachment; filename="transaction-graph-{ds.pk}.csv"'
    return resp


# ------------------------------------------------------------------ explainability
def _bars(items, key):
    """Diverging-bar geometry: positive (towards fraud) grows right, negative grows left."""
    m = max([abs(i[key]) for i in items] or [0]) or 1
    out = []
    for i in items:
        v = i[key]
        out.append({**i, "v": v, "w": abs(v) / m * 50, "pos": v > 0})
    return out


def alert_detail(request, pk):
    alert = get_object_or_404(Alert.objects.select_related("dataset"), pk=pk)
    ex = alert.explanation or {}
    # alerts raised before the GNN-only version explain a multi-model score; they can't be shown here
    ctx = {"a": alert, "ex": ex, "has_ex": ex.get("model") == "GNN" and bool(ex.get("drivers")),
           "legacy_ex": bool(ex) and ex.get("model") != "GNN"}
    if ctx["has_ex"]:
        p, t = ex["probabilities"]["GNN"], ex["thresholds"]["GNN"]
        ctx["meter"] = {"p": p, "t": t, "flag": p >= t, "color": MODEL_COLORS["GNN"]}
        ctx["drivers"] = _bars(ex["drivers"], "delta")
        g = ex.get("graph", [])
        gm = max([abs(e["delta"]) for e in g] or [0])
        ctx["graph"] = [{**e, "w": abs(e["delta"]) / gm * 100 if gm >= 0.05 and abs(e["delta"]) >= 0.05 else 0}
                        for e in g]
        cf = ex.get("counterfactual")
        if cf:
            ctx["cf_path"] = [{"step": i + 1, "p": v, "name": (cf["changed"] + [""] * 10)[i].split(" (")[0],
                               "below": v < cf["threshold"]}
                              for i, v in enumerate(cf["path"][: len(cf["changed"])])]
    if alert.account:
        ctx["account_alerts"] = (Alert.objects.filter(dataset=alert.dataset, account=alert.account)
                                 .exclude(pk=alert.pk).count())
    return render(request, "detector/alert_detail.html", ctx)


# ------------------------------------------------------------------ staff sign-in and customer management
from django import forms as _forms  # noqa: E402
from django.conf import settings as _settings  # noqa: E402
from django.contrib.auth import login as _login, logout as _logout  # noqa: E402
from django.contrib.auth.forms import AuthenticationForm  # noqa: E402
from django.contrib.auth.models import User  # noqa: E402

from .customer import backfill_history, forget_enrolments  # noqa: E402
from .models import AccountTransaction, CustomerProfile  # noqa: E402


class StaffLoginForm(AuthenticationForm):
    def confirm_login_allowed(self, user):
        super().confirm_login_allowed(user)
        if not user.is_staff:
            raise _forms.ValidationError("This sign-in is for analysts. Customers sign in at /my/login/.",
                                         code="not_staff")


def staff_login(request):
    form = StaffLoginForm(request, data=request.POST or None)
    if request.method == "POST" and form.is_valid():
        _login(request, form.get_user())
        nxt = request.GET.get("next", "/")
        return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else "/")
    return render(request, "detector/staff_login.html", {"form": form})


@require_POST
def staff_logout(request):
    _logout(request)
    return redirect("staff_login")


class EnrolForm(_forms.Form):
    dataset = _forms.ModelChoiceField(queryset=Dataset.objects.filter(status=Dataset.Status.READY), empty_label=None)
    account_id = _forms.CharField(max_length=120, label="Account ID")
    name = _forms.CharField(max_length=120, required=False)
    email = _forms.EmailField(required=False)
    username = _forms.CharField(max_length=150)
    password = _forms.CharField(widget=_forms.PasswordInput, min_length=8, help_text="Give this to the customer securely.")

    def clean_username(self):
        u = self.cleaned_data["username"]
        if User.objects.filter(username__iexact=u).exists():
            raise _forms.ValidationError("That username is taken.")
        return u


def customers(request):
    form = EnrolForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        c = form.cleaned_data
        user = User.objects.create_user(c["username"], c["email"], c["password"])
        prof = CustomerProfile.objects.create(user=user, dataset=c["dataset"], account_id=c["account_id"].strip(),
                                              display_name=c["name"])
        forget_enrolments(prof.dataset_id)
        n = backfill_history(prof.dataset, prof.account_id)
        messages.success(request, f"Enrolled {user.username} on {prof.account_id} with {n} earlier transactions.")
        return redirect("customers")
    profs = CustomerProfile.objects.select_related("user", "dataset").order_by("-created_at")
    rows = []
    for p in profs:
        txs = AccountTransaction.objects.filter(dataset=p.dataset, account_id=p.account_id)
        rows.append({"p": p, "monitored": txs.exclude(status="history").count(),
                     "flagged": txs.filter(status="flagged").count(),
                     "responses": Alert.objects.filter(dataset=p.dataset, account=p.account_id)
                                          .exclude(customer_response="").count()})
    return render(request, "detector/customers.html", {"form": form, "rows": rows})


@require_POST
def customer_remove(request, pk):
    p = get_object_or_404(CustomerProfile, pk=pk)
    name, ds_id = p.user.username, p.dataset_id
    if not CustomerProfile.objects.filter(dataset_id=ds_id, account_id=p.account_id).exclude(pk=pk).exists():
        # last customer on the account: its transaction feed goes too
        AccountTransaction.objects.filter(dataset_id=ds_id, account_id=p.account_id).delete()
    p.user.delete()
    forget_enrolments(ds_id)
    messages.success(request, f"Removed {name}.")
    return redirect("customers")
