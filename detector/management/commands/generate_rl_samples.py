"""Write small synthetic look-alikes of the two benchmark files, for trying train_rl without the Kaggle data.

    python manage.py generate_rl_samples --out data/
    python manage.py train_rl --ulb data/creditcard_synthetic.csv --knn-min-sim 0.5 --episodes 60 --epochs 80
    python manage.py train_rl --ieee data/ieee_synthetic --episodes 60 --epochs 80

The synthetic European file is much less clustered than the real one, so link it at --knn-min-sim 0.5;
on the real creditcard.csv keep the default 0.9.
"""
from pathlib import Path

from django.core.management.base import BaseCommand

from detector.ml.rl_data import synth_ieee, synth_ulb


class Command(BaseCommand):
    help = "Write synthetic creditcard.csv and IEEE-CIS look-alikes for testing the RL pipeline."

    def add_arguments(self, parser):
        parser.add_argument("--out", default="data")
        parser.add_argument("--ulb-rows", type=int, default=60_000)
        parser.add_argument("--ieee-rows", type=int, default=80_000)

    def handle(self, out, ulb_rows, ieee_rows, **opts):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        df = synth_ulb(ulb_rows)
        df.to_csv(out / "creditcard_synthetic.csv", index=False)
        self.stdout.write(f"{out / 'creditcard_synthetic.csv'}: {len(df):,} rows, {int(df['Class'].sum())} frauds")
        tr, ident = synth_ieee(ieee_rows)
        folder = out / "ieee_synthetic"
        folder.mkdir(exist_ok=True)
        tr.to_csv(folder / "train_transaction.csv", index=False)
        ident.to_csv(folder / "train_identity.csv", index=False)
        self.stdout.write(f"{folder}/: {len(tr):,} transactions, {tr['isFraud'].mean():.1%} fraud")
