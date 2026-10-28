"""Plant known fraud rings in a synthetic South African card file.

    python manage.py inject_rings credit_card_clean.csv credit_card_rings.csv [--seed 2026]

Adds six rings (shared-device account takeover, card testing, collusive merchant) across the training, validation
and test periods, keeps every derived field consistent, and writes <out>_ring_summary.csv beside the output. The
ground-truth columns fraud_ring_id and fraud_pattern are for evaluation only: no model feature reads them.
"""
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from detector.ml.fraud_rings import RingError, inject_rings


class Command(BaseCommand):
    help = "Add six known fraud rings to a synthetic South African card file."

    def add_arguments(self, parser):
        parser.add_argument("source")
        parser.add_argument("output")
        parser.add_argument("--seed", type=int, default=2026)

    def handle(self, source, output, seed, **opts):
        if not Path(source).exists():
            raise CommandError(f"{source} does not exist.")
        d = pd.read_csv(source, low_memory=False)
        try:
            out, summary = inject_rings(d, seed=seed)
        except RingError as e:
            raise CommandError(str(e))
        out.to_csv(output, index=False)
        sp = str(Path(output).with_name(Path(output).stem + "_ring_summary.csv"))
        summary.to_csv(sp)
        self.stdout.write(summary.to_string())
        new = len(out) - len(d)
        self.stdout.write(f"rows {len(d):,} -> {len(out):,} (+{new}); frauds {int(d['Class'].sum())} -> "
                          f"{int(out['Class'].sum())} ({out['Class'].mean():.2%})")
        self.stdout.write(self.style.SUCCESS(f"Wrote {output} and {sp}"))
