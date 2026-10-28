"""Plant adversarial frauds (built to evade the GNN) in a synthetic South African card file.

    python manage.py inject_adversarial credit_card_clean.csv credit_card_adversarial.csv [--adversarial-training]

Five groups of 8 victim accounts: a naive takeover (control), full mimicry, low and slow, graph camouflage, and an
optimised group whose edits come from a white-box attack on the GNN trained on the file (it queries the GNN and
adds edits until it stops flagging them). With --adversarial-training the groups are also planted in the training
period and the optimised group is re-optimised against the retrained GNN. The ground-truth columns
adversarial_group, adversarial_attack and adversarial_period are for evaluation only.
"""
import json
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from detector.ml.adversarial import AdversarialError, build_adversarial_file


class Command(BaseCommand):
    help = "Add adversarial frauds (evasion attacks on the GNN) to a synthetic South African card file."

    def add_arguments(self, parser):
        parser.add_argument("source")
        parser.add_argument("output")
        parser.add_argument("--adversarial-training", action="store_true")
        parser.add_argument("--seed", type=int, default=7)

    def handle(self, source, output, adversarial_training, seed, **opts):
        if not Path(source).exists():
            raise CommandError(f"{source} does not exist.")
        d = pd.read_csv(source, low_memory=False)
        try:
            out, info = build_adversarial_file(d, seed=seed, adversarial_training=adversarial_training,
                                               progress=lambda m: self.stdout.write(m))
        except AdversarialError as e:
            raise CommandError(str(e))
        out.to_csv(output, index=False)
        sp = Path(output).with_name(Path(output).stem + "_attack.json")
        sp.write_text(json.dumps(info, indent=1))
        for g in info["groups"]:
            self.stdout.write(f"  {g['period']:<5} {g['group']} {g['label']:<26} {g['transactions']:>3} tx  R{g['value']:>10,.2f}")
        for p in info["paths"]:
            self.stdout.write(f"  white-box attack against {p['against']}: {', '.join(p['edits']) or 'no edit'}"
                              f" ({'evaded' if p['evaded'] else 'not fully evaded'})")
        self.stdout.write(self.style.SUCCESS(f"Wrote {output} and {sp}"))
