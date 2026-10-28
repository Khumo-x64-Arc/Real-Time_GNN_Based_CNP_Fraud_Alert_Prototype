"""Run a semi-supervised GNN + DQN experiment in the foreground.

    python manage.py train_rl --ulb path/to/creditcard.csv
    python manage.py train_rl --ieee path/to/ieee-fraud-detection.zip
    python manage.py train_rl --ulb creditcard.csv --label-rate 0.1 --episodes 400

--ulb takes creditcard.csv (or a zip holding it). --ieee takes the Kaggle zip, the folder it unpacks to,
or train_transaction.csv (train_identity.csv is picked up from the same folder).
"""
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from detector.models import RLRun
from detector.tasks import start_rl


class Command(BaseCommand):
    help = "Train the semi-supervised GNN and the DQN agent on the European Cardholder or IEEE-CIS data."

    def add_arguments(self, parser):
        src = parser.add_mutually_exclusive_group(required=True)
        src.add_argument("--ulb", help="European Cardholder creditcard.csv (or its zip)")
        src.add_argument("--ieee", help="IEEE-CIS Kaggle zip, its folder, or train_transaction.csv")
        parser.add_argument("--name", default=None)
        parser.add_argument("--label-rate", type=float, default=0.2,
                            help="Share of transactions whose label is ever revealed (default 0.2)")
        parser.add_argument("--rounds", type=int, default=2, help="Self-training rounds after the first (default 2)")
        parser.add_argument("--episodes", type=int, default=None,
                            help="DQN training episodes (default 300 for ULB, 600 for IEEE-CIS)")
        parser.add_argument("--agent-seeds", type=int, default=3,
                            help="Independently trained agents to average over (default 3)")
        parser.add_argument("--window-minutes", type=float, default=None,
                            help="Length of one decision window (default 15 for ULB, 360 for IEEE-CIS)")
        parser.add_argument("--delay", type=int, default=None,
                            help="Windows before a label arrives (default 2 for ULB, 4 for IEEE-CIS)")
        parser.add_argument("--fp-cost", type=float, default=1.0,
                            help="Cost of a false alarm; a caught fraud scores +1 and a missed one -1 (default 1)")
        parser.add_argument("--epochs", type=int, default=300, help="Maximum GNN epochs per round (default 300)")
        parser.add_argument("--max-rows", type=int, default=None, help="Use only the first N transactions (by time)")
        parser.add_argument("--knn-k", type=int, default=5, help="ULB: similar earlier transactions linked (default 5)")
        parser.add_argument("--knn-min-sim", type=float, default=0.9,
                            help="ULB: minimum cosine similarity for a link (default 0.9, as in FraudGNN-RL)")
        parser.add_argument("--no-reference", action="store_true", help="Skip the GNN trained on 100%% of labels")
        parser.add_argument("--seed", type=int, default=42)

    def handle(self, *args, **o):
        kind, src = ("ulb", o["ulb"]) if o["ulb"] else ("ieee", o["ieee"])
        if not Path(src).exists():
            raise CommandError(f"{src} not found.")
        params = {"label_rate": o["label_rate"], "rounds": o["rounds"], "episodes": o["episodes"],
                  "agent_seeds": o["agent_seeds"],
                  "window_minutes": o["window_minutes"], "delay": o["delay"], "fp_cost": o["fp_cost"],
                  "epochs": o["epochs"], "max_rows": o["max_rows"], "seed": o["seed"],
                  "full_reference": not o["no_reference"]}
        if kind == "ulb":
            params.update(knn_k=o["knn_k"], knn_min_sim=o["knn_min_sim"])
        name = o["name"] or f"{RLRun.Kind(kind).label} · {o['label_rate']:.0%} labels"
        run = RLRun.objects.create(name=name, kind=kind, source=str(Path(src).resolve()), params=params)
        self.stdout.write(f"Run #{run.pk}: {name}. The full IEEE-CIS file takes a while…")
        start_rl(run.pk, background=False)
        run.refresh_from_db()
        if run.status != RLRun.Status.READY:
            raise CommandError(f"{run.get_status_display()}: {run.error}")
        res = run.results
        self.stdout.write("Semi-supervised GNN (test period, scored one transaction at a time):")
        for k, v in res["semi_supervised"]["report"].items():
            self.stdout.write(f"  {k:<11} ROC-AUC {v['roc_auc']:.4f}  PR-AUC {v['pr_auc']:.4f}  F1 {v['f1']:.3f}  "
                              f"Recall@1% {v['recall_at_1pct']:.3f}")
        self.stdout.write("Decision policies on the test stream:")
        for v in res["policies"].values():
            sd = v.get("sd", {})
            pm = (lambda m, f: f"{v[m]:{f}}" + (f" ±{sd[m]:{f}}" if sd else ""))
            self.stdout.write(f"  {v['label']:<30} F1 {pm('f1', '.3f')}  PR-AUC {pm('pr_auc', '.4f')}  "
                              f"Recall@1% {pm('recall_at_1pct', '.3f')}  caught {v['tp']:.0f}  false alarms {v['fp']:.0f}  "
                              f"reward (all labels) {pm('reward_all', '.1f')}")
        self.stdout.write(self.style.SUCCESS(f"Done. Open /rl/runs/{run.pk}/"))
