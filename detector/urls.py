from django.urls import path

from . import customer_views as cv
from . import rl_views as rv
from . import automation_views as av
from . import views

urlpatterns = [
    path("", views.dataset_list, name="dataset_list"),
    path("upload/", views.upload, name="upload"),
    path("datasets/<int:pk>/", views.dataset_detail, name="dataset_detail"),
    path("datasets/<int:pk>/status.json", views.dataset_status, name="dataset_status"),
    path("datasets/<int:pk>/retrain/", views.dataset_retrain, name="dataset_retrain"),
    path("datasets/<int:pk>/oversampling/", views.dataset_oversampling, name="dataset_oversampling"),
    path("datasets/<int:pk>/delete/", views.dataset_delete, name="dataset_delete"),
    path("datasets/<int:pk>/graph/meta.json", views.graph_meta, name="graph_meta"),
    path("datasets/<int:pk>/graph/data.bin", views.graph_data, name="graph_data"),
    path("datasets/<int:pk>/graph/tx/<int:i>/", views.graph_tx, name="graph_tx"),
    path("datasets/<int:pk>/graph/edges.csv", views.graph_export, name="graph_export"),
    path("results/", views.results_overview, name="results"),
    path("results/export.csv", views.results_export, name="results_export"),
    path("automation/", av.automation_overview, name="automation"),
    path("automation/<int:pk>/", av.automation_run, name="automation_run"),
    path("automation/<int:pk>/status.json", av.automation_status, name="automation_status"),
    path("automation/<int:pk>/delete/", av.automation_delete, name="automation_delete"),
    path("automation/<int:pk>/decisions.csv", av.automation_decisions, name="automation_decisions"),
    path("automation/<int:pk>/file/", av.automation_file, name="automation_file"),
    path("rl/", rv.rl_overview, name="rl"),
    path("rl/runs/<int:pk>/", rv.rl_run, name="rl_run"),
    path("rl/runs/<int:pk>/status.json", rv.rl_status, name="rl_status"),
    path("rl/runs/<int:pk>/delete/", rv.rl_delete, name="rl_delete"),
    path("monitor/", views.monitor, name="monitor"),
    path("monitor/<int:pk>/", views.monitor, name="monitor_dataset"),
    path("monitor/<int:pk>/stream/", views.monitor_stream, name="monitor_stream"),
    path("alerts/", views.alert_list, name="alert_list"),
    path("alerts/clear/", views.alerts_clear, name="alerts_clear"),
    path("alerts/<int:pk>/", views.alert_detail, name="alert_detail"),
    path("alerts/<int:pk>/review/", views.alert_review, name="alert_review"),
    path("customers/", views.customers, name="customers"),
    path("customers/<int:pk>/remove/", views.customer_remove, name="customer_remove"),
    path("staff/login/", views.staff_login, name="staff_login"),
    path("staff/logout/", views.staff_logout, name="staff_logout"),
    # customer portal
    path("my/", cv.customer_dashboard, name="customer_dashboard"),
    path("my/login/", cv.customer_login, name="customer_login"),
    path("my/signup/", cv.customer_signup, name="customer_signup"),
    path("my/logout/", cv.customer_logout, name="customer_logout"),
    path("my/feed/", cv.customer_feed, name="customer_feed"),
    path("my/settings/", cv.customer_settings, name="customer_settings"),
    path("my/alerts/<int:pk>/respond/", cv.customer_respond, name="customer_respond"),
    path("api/alerts/latest/", views.api_latest_alerts, name="api_latest_alerts"),
    path("api/datasets/<int:pk>/score/", views.api_score, name="api_score"),
]
