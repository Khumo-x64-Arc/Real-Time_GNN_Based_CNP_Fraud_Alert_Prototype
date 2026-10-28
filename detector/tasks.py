"""Background training in a thread (keeps the demo dependency-free: no Celery/Redis)."""
import logging
import shutil
import threading
import time

from django.db import close_old_connections, transaction
from django.utils import timezone

from .ml.engine import drop_engine
from .ml.pipeline import DatasetError, train_and_evaluate
from .models import Dataset, ModelResult

log = logging.getLogger(__name__)


def _run(dataset_id: int):
    close_old_connections()
    ds = Dataset.objects.get(pk=dataset_id)
    last = [0.0]

    def progress(msg, pct):
        if time.time() - last[0] > 0.5 or pct >= 90:
            Dataset.objects.filter(pk=dataset_id).update(progress=pct, progress_message=msg)
            last[0] = time.time()

    try:
        Dataset.objects.filter(pk=dataset_id).update(status=Dataset.Status.TRAINING, progress=1, error="",
                                                      progress_message="Starting")
        drop_engine(dataset_id)
        shutil.rmtree(ds.model_dir, ignore_errors=True)
        res = train_and_evaluate(ds.file.path, ds.model_dir, progress)
        with transaction.atomic():
            ds.model_results.all().delete()
            for name, m in res["models"].items():
                ModelResult.objects.create(dataset=ds, name=name, **{k: v for k, v in m.items()})
            Dataset.objects.filter(pk=dataset_id).update(
                status=Dataset.Status.READY, progress=100, progress_message="Done", results=res,
                schema=res["schema"], trained_at=timezone.now())
        log.info("Dataset %s trained", dataset_id)
    except DatasetError as e:
        Dataset.objects.filter(pk=dataset_id).update(status=Dataset.Status.UNUSABLE, error=str(e))
    except Exception as e:  # show the reason on the dataset page instead of hanging forever
        log.exception("Training failed for dataset %s", dataset_id)
        Dataset.objects.filter(pk=dataset_id).update(status=Dataset.Status.FAILED,
                                                      error=f"{type(e).__name__}: {e}")
    finally:
        close_old_connections()


def _run_oversampling(dataset_id: int):
    """Train the counterpart GNN (oversampling switched the other way) for a dataset trained before the
    comparison existed, and store the comparison in its results."""
    from .ml.pipeline import oversampling_comparison
    close_old_connections()
    ds = Dataset.objects.get(pk=dataset_id)

    def save(block):
        d = Dataset.objects.get(pk=dataset_id)
        Dataset.objects.filter(pk=dataset_id).update(results={**(d.results or {}), "oversampling": block})

    last = [0.0]

    def progress(msg, pct):
        if time.time() - last[0] > 1.0:
            save({"status": "running", "progress": pct, "message": msg})
            last[0] = time.time()
    try:
        save({"status": "running", "progress": 1, "message": "Starting"})
        block = oversampling_comparison(ds.file.path, ds.results or {}, progress)
        save(block)
    except Exception as e:
        log.exception("Oversampling comparison failed for dataset %s", dataset_id)
        save({"status": "failed", "error": f"{type(e).__name__}: {e}"})
    finally:
        close_old_connections()


def start_oversampling_comparison(dataset_id: int, background: bool = True):
    (threading.Thread(target=_run_oversampling, args=(dataset_id,), daemon=True).start() if background
     else _run_oversampling(dataset_id))


def start_training(dataset_id: int, background: bool = True):
    if background:
        threading.Thread(target=_run, args=(dataset_id,), daemon=True).start()
    else:
        _run(dataset_id)


# ------------------------------------------------------------------ semi-supervised GNN + DQN experiments
def _run_rl(pk: int):
    from .ml.rl_data import RLDataError
    from .ml.rl_pipeline import run_experiment
    from .ml.semisup import SemiSupError
    from .models import RLRun

    close_old_connections()
    run = RLRun.objects.get(pk=pk)
    try:
        RLRun.objects.filter(pk=pk).update(status=RLRun.Status.RUNNING, progress=1, error="")
        shutil.rmtree(run.run_dir, ignore_errors=True)
        res = run_experiment(run.kind, run.source, run.run_dir, progress=_progress_writer(RLRun, pk), **run.params)
        RLRun.objects.filter(pk=pk).update(status=RLRun.Status.READY, progress=100, progress_message="Done",
                                           results=res, finished_at=timezone.now())
    except (RLDataError, SemiSupError) as e:
        RLRun.objects.filter(pk=pk).update(status=RLRun.Status.FAILED, error=str(e))
    except Exception as e:
        log.exception("RL run %s failed", pk)
        RLRun.objects.filter(pk=pk).update(status=RLRun.Status.FAILED, error=f"{type(e).__name__}: {e}")
    finally:
        close_old_connections()


def start_rl(pk: int, background: bool = True):
    (threading.Thread(target=_run_rl, args=(pk,), daemon=True).start() if background else _run_rl(pk))


# ------------------------------------------------------------------ automated detection: GNN + RL agent
def _run_automation(pk: int):
    from .ml.fraud_automation import run_automation
    from .ml.transfer_target import FinetuneError
    from .models import AutomationRun

    close_old_connections()
    run = AutomationRun.objects.get(pk=pk)
    try:
        AutomationRun.objects.filter(pk=pk).update(status="running", progress=1, error="")
        params = dict(run.params)
        params.pop("transfer", None)   # older runs recorded whether they used transfer learning
        params.pop("rings", None)
        adv = params.pop("adversarial", None)
        params.pop("adversarial_setup", None)
        progress = _progress_writer(AutomationRun, pk)
        if adv and not run.params.get("adversarial_setup"):
            _plant_adversarial(run, adv, progress)
            run.refresh_from_db()
        res = run_automation(run.file.path, {}, run.run_dir, progress=progress, **params)
        if run.params.get("adversarial_setup"):
            res["adversarial_setup"] = run.params["adversarial_setup"]
        AutomationRun.objects.filter(pk=pk).update(status="ready", progress=100, progress_message="Done",
                                                   results=res, finished_at=timezone.now())
    except FinetuneError as e:
        AutomationRun.objects.filter(pk=pk).update(status="failed", error=str(e))
    except Exception as e:
        log.exception("Automation run %s failed", pk)
        AutomationRun.objects.filter(pk=pk).update(status="failed", error=f"{type(e).__name__}: {e}")
    finally:
        close_old_connections()


def _plant_adversarial(run, adv: dict, progress):
    """Replace the run's file with the attacked one (white-box attack included) and record how it was built."""
    import io
    from pathlib import Path

    import pandas as pd
    from django.core.files.base import ContentFile

    from .ml.adversarial import AdversarialError, build_adversarial_file
    from .ml.transfer_target import FinetuneError
    from .models import AutomationRun

    d = pd.read_csv(run.file.path, low_memory=False)
    try:
        out, info = build_adversarial_file(d, seed=adv.get("seed", 7), adversarial_training=adv.get("training", False),
                                           progress=lambda m: progress(m, 3))
    except AdversarialError as e:
        raise FinetuneError(str(e))
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    old = run.file.name
    run.file.save(Path(old).stem + "_adversarial.csv", ContentFile(buf.getvalue().encode()), save=False)
    params = {**run.params, "adversarial_setup": info}
    AutomationRun.objects.filter(pk=run.pk).update(file=run.file.name, params=params)


def start_automation(pk: int, background: bool = True):
    (threading.Thread(target=_run_automation, args=(pk,), daemon=True).start() if background else _run_automation(pk))
