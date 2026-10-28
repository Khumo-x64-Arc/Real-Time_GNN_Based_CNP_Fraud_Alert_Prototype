from pathlib import Path

from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.urls import reverse


class Dataset(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        TRAINING = "training", "Training"
        READY = "ready", "Ready"
        UNUSABLE = "unusable", "Can't be trained on"
        FAILED = "failed", "Training failed"

    name = models.CharField(max_length=200)
    file = models.FileField(upload_to="datasets/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.QUEUED)
    progress = models.PositiveSmallIntegerField(default=0)
    progress_message = models.CharField(max_length=200, blank=True)
    error = models.TextField(blank=True)
    schema = models.JSONField(default=dict, blank=True)
    columns = models.JSONField(default=list, blank=True)
    results = models.JSONField(default=dict, blank=True)
    trained_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse("dataset_detail", args=[self.pk])

    @property
    def model_dir(self) -> Path:
        return Path(settings.MEDIA_ROOT) / "models" / str(self.pk)

    @property
    def bundle_path(self) -> Path:
        return self.model_dir / "bundle.joblib"

    @property
    def is_gnn_only(self):
        """Trained by the GNN-only pipeline (older results also hold XGBoost, ensemble and LSTM-CNN)."""
        return str((self.results or {}).get("version", "")).startswith("gnn-only")

    @property
    def is_live_ready(self):
        return self.status == self.Status.READY and self.is_gnn_only and self.bundle_path.exists()

    @property
    def best_model(self):
        return self.model_results.filter(name__in=ModelResult.ORDER).order_by("-f1").first()

    @property
    def gnn_result(self):
        """The result the live monitor stands for: streaming if measured, else full graph."""
        rs = {r.name: r for r in self.model_results.all()}
        return rs.get("GNN live") or rs.get("GNN")


class ModelResult(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="model_results")
    name = models.CharField(max_length=40)
    roc_auc = models.FloatField()
    pr_auc = models.FloatField()
    f1 = models.FloatField()
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    threshold = models.FloatField()
    tp = models.IntegerField()
    fp = models.IntegerField()
    tn = models.IntegerField()
    fn = models.IntegerField()
    train_seconds = models.FloatField(default=0)
    roc_curve = models.JSONField(default=list)

    ORDER = ["GNN", "GNN live"]  # full graph, then streaming (how the live monitor scores)

    class Meta:
        unique_together = [("dataset", "name")]

    def __str__(self):
        return f"{self.dataset} · {self.name}"


class Alert(models.Model):
    class Level(models.TextChoices):
        FRAUD = "fraud", "Likely fraud"
        REVIEW = "review", "Needs review"

    class Review(models.TextChoices):
        OPEN = "open", "Open"
        CONFIRMED = "confirmed", "Confirmed fraud"
        DISMISSED = "dismissed", "False alarm"

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="alerts")
    transaction_id = models.CharField(max_length=120)
    level = models.CharField(max_length=10, choices=Level.choices)
    score = models.FloatField(help_text="GNN fraud probability")
    probabilities = models.JSONField(default=dict)
    signals = models.JSONField(default=list)
    explanation = models.JSONField(default=dict, blank=True)
    amount = models.FloatField(null=True, blank=True)
    account = models.CharField(max_length=120, blank=True)
    merchant = models.CharField(max_length=200, blank=True)
    country = models.CharField(max_length=60, blank=True)
    occurred_at = models.CharField(max_length=60, blank=True)
    actual_label = models.SmallIntegerField(null=True, blank=True)
    source = models.CharField(max_length=20, default="monitor")
    review = models.CharField(max_length=10, choices=Review.choices, default=Review.OPEN)
    emailed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class CustomerResponse(models.TextChoices):
        NONE = "", "No response"
        MINE = "mine", "Customer says it was them"
        NOT_MINE = "not_mine", "Customer doesn't recognise it"

    customer_response = models.CharField(max_length=10, choices=CustomerResponse.choices, blank=True, default="")
    customer_responded_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at", "-id"]
        unique_together = [("dataset", "transaction_id")]

    def __str__(self):
        return f"{self.get_level_display()} · {self.transaction_id}"


class CustomerProfile(models.Model):
    """Links a login to one account in one dataset. Customers only ever see that account."""

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="customer")
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="customers")
    account_id = models.CharField(max_length=120)
    display_name = models.CharField(max_length=120, blank=True)
    email_alerts = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} → {self.account_id}"

    @property
    def masked_account(self):
        tail = self.account_id[-4:]
        return f"••••{tail}"


class AccountTransaction(models.Model):
    """A transaction on an enrolled customer's account, as the customer sees it."""

    class Status(models.TextChoices):
        HISTORY = "history", "Earlier transaction"
        NORMAL = "normal", "Normal"
        REVIEW = "review", "Being checked"
        FLAGGED = "flagged", "Flagged as suspicious"

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="account_transactions")
    account_id = models.CharField(max_length=120, db_index=True)
    transaction_id = models.CharField(max_length=120)
    amount = models.FloatField(null=True, blank=True)
    merchant = models.CharField(max_length=200, blank=True)
    category = models.CharField(max_length=80, blank=True)
    country = models.CharField(max_length=60, blank=True)
    channel = models.CharField(max_length=40, blank=True)
    occurred_at = models.CharField(max_length=60, blank=True)
    status = models.CharField(max_length=10, choices=Status.choices)
    reasons = models.JSONField(default=list, blank=True)
    alert = models.ForeignKey(Alert, null=True, blank=True, on_delete=models.SET_NULL, related_name="account_transactions")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-id"]
        unique_together = [("dataset", "transaction_id")]
        indexes = [models.Index(fields=["dataset", "account_id", "id"])]

    def __str__(self):
        return f"{self.account_id} · {self.transaction_id}"


class RLRun(models.Model):
    """One semi-supervised GNN + DQN experiment on the European Cardholder or IEEE-CIS data."""

    class Kind(models.TextChoices):
        ULB = "ulb", "European Cardholder (ULB)"
        IEEE = "ieee", "IEEE-CIS"

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    name = models.CharField(max_length=200)
    kind = models.CharField(max_length=10, choices=Kind.choices)
    source = models.CharField(max_length=500, help_text="Path of the data file or folder on the server")
    params = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.QUEUED)
    progress = models.PositiveSmallIntegerField(default=0)
    progress_message = models.CharField(max_length=200, blank=True)
    error = models.TextField(blank=True)
    results = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse("rl_run", args=[self.pk])

    @property
    def run_dir(self) -> Path:
        return Path(settings.MEDIA_ROOT) / "rl" / str(self.pk)


class AutomationRun(models.Model):
    """Automated fraud detection on a labelled file: the GNN and the DQN agent decide every test transaction."""

    name = models.CharField(max_length=200)
    file = models.FileField(upload_to="automation_targets/")
    params = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=20, choices=RLRun.Status.choices, default=RLRun.Status.QUEUED)
    progress = models.PositiveSmallIntegerField(default=0)
    progress_message = models.CharField(max_length=200, blank=True)
    error = models.TextField(blank=True)
    results = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse("automation_run", args=[self.pk])

    @property
    def has_rings(self) -> bool:
        return bool(self.params.get("rings"))

    @property
    def has_adversarial(self) -> bool:
        return bool(self.params.get("adversarial"))

    @property
    def run_dir(self) -> Path:
        return Path(settings.MEDIA_ROOT) / "automation" / str(self.pk)
