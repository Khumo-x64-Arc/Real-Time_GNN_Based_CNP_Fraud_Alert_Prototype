"""The customer portal: each customer sees only their own account."""
from functools import wraps

from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from .customer import backfill_history, can_self_enrol, forget_enrolments, verify_account
from .models import AccountTransaction, Alert, CustomerProfile, Dataset


SESSION_KEY = "as_customer"  # the customer being viewed when customer login is switched off


def login_required_for_customers() -> bool:
    return getattr(settings, "CUSTOMER_LOGIN_REQUIRED", True)


def customer_required(view):
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if not login_required_for_customers():
            prof = (CustomerProfile.objects.select_related("user", "dataset")
                    .filter(pk=request.session.get(SESSION_KEY)).first())
            if prof is None:
                return redirect("customer_login")   # the customer picker
            request.profile = prof
            return view(request, *args, **kwargs)
        if not request.user.is_authenticated:
            return redirect(f"/my/login/?next={request.path}")
        prof = getattr(request.user, "customer", None)
        if prof is None:
            if request.user.is_staff:
                messages.info(request, "Staff accounts don't have a customer dashboard. Enrol a customer on the Customers page.")
                return redirect("customers")
            logout(request)
            return redirect("customer_login")
        request.profile = prof
        return view(request, *args, **kwargs)
    return wrapper


# ------------------------------------------------------------------ sign in / up
class CustomerLoginForm(AuthenticationForm):
    def confirm_login_allowed(self, user):
        super().confirm_login_allowed(user)
        if not hasattr(user, "customer"):
            raise forms.ValidationError("This sign-in is for customers. Staff should use the analyst sign-in.",
                                        code="not_customer")


def customer_pick(request):
    """Login switched off: choose which enrolled customer's dashboard to open."""
    if request.method == "POST":
        prof = CustomerProfile.objects.filter(pk=request.POST.get("profile")).first()
        if prof:
            request.session[SESSION_KEY] = prof.pk
            return redirect("customer_dashboard")
        messages.error(request, "That customer no longer exists.")
    profiles = CustomerProfile.objects.select_related("user", "dataset").order_by("dataset__name", "account_id")
    return render(request, "detector/customer/pick.html", {"profiles": profiles})


def customer_login(request):
    if not login_required_for_customers():
        return customer_pick(request)
    if request.user.is_authenticated and hasattr(request.user, "customer"):
        return redirect("customer_dashboard")
    form = CustomerLoginForm(request, data=request.POST or None)
    if request.method == "POST" and form.is_valid():
        login(request, form.get_user())
        nxt = request.GET.get("next", "")
        return redirect(nxt if nxt.startswith("/my/") else "customer_dashboard")
    return render(request, "detector/customer/login.html", {"form": form})


class SignupForm(forms.Form):
    dataset = forms.ModelChoiceField(queryset=Dataset.objects.none(), label="Bank / dataset", empty_label=None)
    account_id = forms.CharField(label="Account number", max_length=120)
    card_last4 = forms.CharField(label="Last 4 digits of your card", min_length=4, max_length=4)
    name = forms.CharField(label="Your name", max_length=120, required=False)
    email = forms.EmailField(label="Email for alerts", required=False)
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput, min_length=8)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        ids = [d.pk for d in Dataset.objects.filter(status=Dataset.Status.READY) if can_self_enrol(d)]
        self.fields["dataset"].queryset = Dataset.objects.filter(pk__in=ids)

    def clean_username(self):
        u = self.cleaned_data["username"]
        if User.objects.filter(username__iexact=u).exists():
            raise forms.ValidationError("That username is taken.")
        return u

    def clean_card_last4(self):
        v = self.cleaned_data["card_last4"]
        if not v.isdigit():
            raise forms.ValidationError("Enter four digits.")
        return v

    def clean_password(self):
        pw = self.cleaned_data["password"]
        validate_password(pw)
        return pw

    def clean(self):
        c = super().clean()
        ds, acct, last4 = c.get("dataset"), c.get("account_id"), c.get("card_last4")
        if ds and acct and last4 and not verify_account(ds, acct.strip(), last4):
            # one message for both cases, so the form can't be used to discover valid account numbers
            raise forms.ValidationError("We couldn't match that account number and card. Check both and try again.")
        return c


def customer_signup(request):
    form = SignupForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        c = form.cleaned_data
        with transaction.atomic():
            user = User.objects.create_user(c["username"], c["email"], c["password"])
            prof = CustomerProfile.objects.create(user=user, dataset=c["dataset"], account_id=c["account_id"].strip(),
                                                  display_name=c["name"])
        forget_enrolments(prof.dataset_id)
        backfill_history(prof.dataset, prof.account_id)
        if login_required_for_customers():
            login(request, user)
        else:
            request.session[SESSION_KEY] = prof.pk
        messages.success(request, "You're enrolled. We'll alert you here the moment anything on your account looks unusual.")
        return redirect("customer_dashboard")
    return render(request, "detector/customer/signup.html", {"form": form,
                                                             "available": form.fields["dataset"].queryset.exists()})


@require_POST
def customer_logout(request):
    if not login_required_for_customers():
        request.session.pop(SESSION_KEY, None)   # back to the customer picker
        return redirect("customer_login")
    logout(request)
    return redirect("customer_login")


# ------------------------------------------------------------------ dashboard
def _tx_json(t: AccountTransaction):
    a = t.alert
    return {"id": t.pk, "tx": t.transaction_id, "amount": t.amount, "merchant": t.merchant, "category": t.category,
            "country": t.country, "channel": t.channel, "time": t.occurred_at, "status": t.status,
            "status_label": t.get_status_display(), "reasons": t.reasons,
            "alert_id": a.pk if a else None, "response": a.customer_response if a else ""}


def _attention(prof):
    qs = (AccountTransaction.objects.filter(dataset=prof.dataset, account_id=prof.account_id,
                                            status__in=["flagged", "review"], alert__isnull=False,
                                            alert__customer_response="")
          .select_related("alert").order_by("-id"))
    return list(qs[:20])


@customer_required
def customer_dashboard(request):
    prof = request.profile
    txs = list(AccountTransaction.objects.filter(dataset=prof.dataset, account_id=prof.account_id)
               .select_related("alert")[:60])
    stats = AccountTransaction.objects.filter(dataset=prof.dataset, account_id=prof.account_id)
    return render(request, "detector/customer/dashboard.html", {
        "prof": prof, "initial": {"transactions": [_tx_json(t) for t in reversed(txs)],
                                  "attention": [_tx_json(t) for t in _attention(prof)]},
        "last_id": txs[0].pk if txs else 0,
        "n_monitored": stats.exclude(status="history").count(),
        "n_flagged": stats.filter(status="flagged").count(),
        "login_off": not login_required_for_customers(),
    })


@customer_required
@require_GET
def customer_feed(request):
    """Polled every few seconds by the dashboard."""
    prof = request.profile
    try:
        after = int(request.GET.get("after", 0))
    except ValueError:
        after = 0
    base = AccountTransaction.objects.filter(dataset=prof.dataset, account_id=prof.account_id).select_related("alert")
    new = list(base.filter(pk__gt=after).order_by("id")[:50])
    return JsonResponse({
        "transactions": [_tx_json(t) for t in new],
        "attention": [_tx_json(t) for t in _attention(prof)],
        "n_monitored": base.exclude(status="history").count(),
        "n_flagged": base.filter(status="flagged").count(),
    })


@customer_required
@require_POST
def customer_respond(request, pk):
    prof = request.profile
    # the alert must belong to this customer's own account, or it's a 404 (not a 403, which would leak existence)
    alert = get_object_or_404(Alert, pk=pk, dataset=prof.dataset, account=prof.account_id)
    resp = request.POST.get("response")
    if resp not in {"mine", "not_mine"}:
        return JsonResponse({"error": "Unknown response"}, status=400)
    alert.customer_response = resp
    alert.customer_responded_at = timezone.now()
    alert.save(update_fields=["customer_response", "customer_responded_at"])
    msg = ("Thanks for confirming. We've noted that this was you." if resp == "mine" else
           "Thanks for telling us. Our fraud team has been notified and will contact you. "
           "If you still have the card, consider freezing it in your banking app.")
    if request.headers.get("x-requested-with") == "fetch":
        return JsonResponse({"ok": True, "message": msg})
    messages.success(request, msg)
    return redirect("customer_dashboard")


@customer_required
@require_POST
def customer_settings(request):
    prof = request.profile
    prof.email_alerts = request.POST.get("email_alerts") == "on"
    email = request.POST.get("email", "").strip()
    user = prof.user   # the customer's own user, also when login is switched off
    if email != user.email:
        try:
            forms.EmailField().clean(email) if email else None
            user.email = email
            user.save(update_fields=["email"])
        except forms.ValidationError:
            messages.error(request, "That email address doesn't look right.")
            return redirect("customer_dashboard")
    prof.save(update_fields=["email_alerts"])
    messages.success(request, "Alert settings saved.")
    return redirect("customer_dashboard")
