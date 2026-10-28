from django.contrib.auth.models import User
from django.core.management.base import BaseCommand, CommandError

from detector.customer import backfill_history
from detector.models import CustomerProfile, Dataset


class Command(BaseCommand):
    help = "Enrol a customer in the portal: create a login linked to one account in a dataset."

    def add_arguments(self, p):
        p.add_argument("--dataset", type=int, required=True, help="Dataset ID (see the Datasets page URL)")
        p.add_argument("--account", required=True, help="Account ID, e.g. ACC0034401")
        p.add_argument("--username", required=True)
        p.add_argument("--password", required=True)
        p.add_argument("--email", default="")
        p.add_argument("--name", default="")

    def handle(self, dataset, account, username, password, email, name, **_):
        try:
            ds = Dataset.objects.get(pk=dataset)
        except Dataset.DoesNotExist:
            raise CommandError(f"No dataset {dataset}")
        if User.objects.filter(username=username).exists():
            raise CommandError(f"Username {username} is taken")
        user = User.objects.create_user(username, email, password)
        CustomerProfile.objects.create(user=user, dataset=ds, account_id=account, display_name=name)
        n = backfill_history(ds, account)
        self.stdout.write(self.style.SUCCESS(f"Enrolled {username} on {account} ({n} earlier transactions). "
                                             f"Sign in at /my/login/"))
