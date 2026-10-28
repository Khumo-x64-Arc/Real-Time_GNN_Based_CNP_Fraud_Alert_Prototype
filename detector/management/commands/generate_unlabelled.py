"""Write an unlabelled synthetic transaction CSV and, separately, its answer key."""
from pathlib import Path

from django.core.management.base import BaseCommand

from detector.ml.synthgen import generate_unlabelled


class Command(BaseCommand):
    help = "Generate an unlabelled synthetic transaction file plus a separate answer key."

    def add_arguments(self, parser):
        parser.add_argument("--rows", type=int, default=50_000)
        parser.add_argument("--accounts", type=int, default=3_000)
        parser.add_argument("--fraud-rate", type=float, default=0.012)
        parser.add_argument("--seed", type=int, default=2026)
        parser.add_argument("--out", default=".", help="Folder to write the two CSV files to")

    def handle(self, rows, accounts, fraud_rate, seed, out, **opts):
        df, key = generate_unlabelled(rows, accounts, fraud_rate, seed)
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        tag = f"{rows // 1000}k" if rows >= 1000 else str(rows)
        f1, f2 = out / f"synthetic_unlabelled_{tag}.csv", out / f"synthetic_answer_key_{tag}.csv"
        df.to_csv(f1, index=False)
        key.to_csv(f2, index=False)
        self.stdout.write(f"{len(df):,} transactions, no label column -> {f1}")
        self.stdout.write(f"Answer key ({int(key.is_fraud.sum()):,} hidden frauds) -> {f2}")
        self.stdout.write("Score the first file in the app; give the answer key only if you want it evaluated.")
