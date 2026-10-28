from pathlib import Path

from django.core.files import File
from django.core.management.base import BaseCommand, CommandError

from detector.models import Dataset
from detector.tasks import start_training


class Command(BaseCommand):
    help = "Upload a CSV from disk, train the GNN on it and test it, in the foreground (handy for large files)."

    def add_arguments(self, parser):
        parser.add_argument("csv")
        parser.add_argument("--name", default=None)

    def handle(self, csv, name=None, **opts):
        p = Path(csv)
        if not p.exists():
            raise CommandError(f"{csv} not found")
        with p.open("rb") as fh:
            ds = Dataset.objects.create(name=name or p.stem, file=File(fh, name=p.name))
        self.stdout.write(f"Training dataset #{ds.pk} ({ds.name})…")
        start_training(ds.pk, background=False)
        ds.refresh_from_db()
        if ds.status != Dataset.Status.READY:
            raise CommandError(f"{ds.get_status_display()}: {ds.error}")
        for r in ds.model_results.all():
            self.stdout.write(f"  {r.name:<9} ROC-AUC {r.roc_auc:.4f}  PR-AUC {r.pr_auc:.4f}  F1 {r.f1:.4f}  "
                              f"P {r.precision:.3f}  R {r.recall:.3f}  TP {r.tp} FP {r.fp} FN {r.fn} TN {r.tn}")
        abl = ds.results.get("ablation") or []
        if abl:
            self.stdout.write("  Graph ablation (test PR-AUC, full graph):")
            for a in abl:
                self.stdout.write(f"    {a['label']:<38} {a['pr_auc']:.4f}  {a['d_pr_auc']:+.4f}")
        self.stdout.write(self.style.SUCCESS(f"Done. Open /datasets/{ds.pk}/"))
