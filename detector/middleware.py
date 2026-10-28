"""Keep the analyst console and the customer portal apart.

Everything outside /my/ is the analyst console and needs a staff login;
customers who wander there are sent back to their own dashboard. Set
ANALYST_LOGIN_REQUIRED=0 to switch the check off for local experiments.
"""
from django.conf import settings
from django.shortcuts import redirect
from django.urls import reverse

PUBLIC_PREFIXES = ("/my/", "/static/", "/admin/", "/staff/login/", "/api/datasets/")


class ConsoleAccessMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path
        if getattr(settings, "ANALYST_LOGIN_REQUIRED", True) and not path.startswith(PUBLIC_PREFIXES):
            user = request.user
            if not user.is_authenticated:
                return redirect(f"{reverse('staff_login')}?next={path}")
            if not user.is_staff:
                return redirect("customer_dashboard")
        return self.get_response(request)
