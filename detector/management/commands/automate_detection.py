"""Automated fraud detection on a labelled card file: the GNN and the DQN agent decide every transaction.

    python manage.py automate_detection --target credit_card_clean.csv
    python manage.py automate_detection --target credit_card_clean.csv --inject-rings
    python manage.py automate_detection --target credit_card_clean.csv --adversarial [--adversarial-training]

The GNN is trained on 20% of the file's labels. The agent is trained on the validation period with 20% of labels arriving late, then runs the test period on
its own: it escalates, groups escalations into cases and rings, and logs what it chose not to escalate.
"""
from pathlib import Path

from django.core.files import File
from django.core.management.base import BaseCommand, CommandError

from detector.models import AutomationRun
from detector.tasks import start_automation


class Command(BaseCommand):
    help = "Run the GNN + RL agent end to end on a labelled card file and log every decision."

    def add_arguments(self, parser):
        parser.add_argument("--target", required=True, help="Target CSV in the synthetic South African schema")
        parser.add_argument("--name", default="")
        parser.add_argument("--episodes", type=int, default=300)
        parser.add_argument("--agents", type=int, default=3, help="Independently trained agents")
        parser.add_argument("--window-hours", type=float, default=2.0)
        parser.add_argument("--delay", type=int, default=3, help="Label delay in windows")
        parser.add_argument("--inject-rings", action="store_true",
                            help="Plant six known fraud rings in the target first and score the agent on them")
        parser.add_argument("--ring-seed", type=int, default=2026)
        parser.add_argument("--adversarial", action="store_true",
                            help="Plant adversarial frauds (incl. a white-box attack on the GNN) and score them")
        parser.add_argument("--adversarial-training", action="store_true",
                            help="Also plant the attacks in training; the attacker then re-optimises")

    def handle(self, target, name, episodes, agents, window_hours, delay, inject_rings, ring_seed, adversarial,
               adversarial_training, **opts):
        tp = Path(target)
        if not tp.exists():
            raise CommandError(f"{target} does not exist.")
        params = {"episodes": episodes, "agent_seeds": agents, "window_hours": window_hours, "delay": delay}
        content = None
        if inject_rings:
            import io

            import pandas as pd

            from detector.ml.fraud_rings import RingError
            from detector.ml.fraud_rings import inject_rings as plant
            try:
                out, summary = plant(pd.read_csv(tp, low_memory=False), seed=ring_seed)
            except RingError as e:
                raise CommandError(str(e))
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            content = buf.getvalue().encode()
            from detector.automation_views import _summary_records
            params["rings"] = _summary_records(summary)
            self.stdout.write(f"Planted {len(summary)} rings ({int(summary['transactions'].sum())} fraud transactions).")
        if adversarial or adversarial_training:
            params["adversarial"] = {"training": bool(adversarial_training), "seed": 7}
        tag = " with planted rings" if inject_rings else ""
        if adversarial or adversarial_training:
            tag += " with adversarial examples" + (" (adversarial training)" if adversarial_training else "")
        run = AutomationRun(name=name or f"Automated detection on {tp.stem}{tag}", params=params)
        if content is not None:
            from django.core.files.base import ContentFile
            run.file.save(tp.stem + "_rings.csv", ContentFile(content), save=False)
        else:
            with open(tp, "rb") as fh:
                run.file.save(tp.name, File(fh), save=False)
        run.save()
        self.stdout.write(f"Run #{run.pk}. Progress: /automation/{run.pk}/")
        start_automation(run.pk, background=False)
        run.refresh_from_db()
        if run.status != "ready":
            raise CommandError(f"Failed: {run.error}")
        a = run.results["automation"]
        self.stdout.write("Deployed the agent with the most validation reward.")
        for f in a["funnel"]:
            self.stdout.write(f"  {f['label']:<28} {f['n']:>6}  ({f['frauds']} frauds)")
        cs = a["case_stats"]
        self.stdout.write(f"  {cs['cases']} cases ({cs['rings']} rings across accounts), "
                          f"{cs['frauds_in_cases']} of {cs['tx_in_cases']} case transactions are fraud")
        for r in a.get("known_rings") or []:
            if r.get("scored"):
                self.stdout.write(f"  planted ring {r['ring']} ({r['pattern']}): {r['in_ring_case']} of {r['transactions']} "
                                  f"in a ring case, {r['escalated']} escalated -> {r['outcome']}")
        adv = a.get("adversarial")
        if adv:
            for p in run.results.get("adversarial_setup", {}).get("paths", []):
                self.stdout.write(f"  white-box attack against {p['against']}: {', '.join(p['edits']) or 'no edit'}")
            for g in adv["groups"]:
                self.stdout.write(f"  adversarial {g['group']} {g['label']:<26} GNN {g['gnn_alone']:>2}/{g['transactions']:<2} "
                                  f"agent {g['escalated']:>2}/{g['transactions']:<2} R{g['value'] - g['value_escalated']:,.0f} not escalated")
        self.stdout.write(self.style.SUCCESS(f"Decision log: {run.run_dir / 'decisions.csv'}"))
