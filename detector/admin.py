from django.contrib import admin

from .models import AccountTransaction, Alert, CustomerProfile, Dataset, ModelResult


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("name", "status", "uploaded_at", "trained_at")


@admin.register(ModelResult)
class ModelResultAdmin(admin.ModelAdmin):
    list_display = ("dataset", "name", "roc_auc", "f1", "precision", "recall", "tp", "fp", "tn", "fn")


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ("transaction_id", "dataset", "level", "score", "amount", "review", "created_at")
    list_filter = ("level", "review", "dataset")


@admin.register(CustomerProfile)
class CustomerProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "dataset", "account_id", "email_alerts", "created_at")


@admin.register(AccountTransaction)
class AccountTransactionAdmin(admin.ModelAdmin):
    list_display = ("transaction_id", "account_id", "amount", "status", "created_at")
    list_filter = ("status", "dataset")
