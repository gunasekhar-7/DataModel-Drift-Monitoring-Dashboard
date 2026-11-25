"""Prometheus metrics for the Data & Model Drift Monitoring Platform."""

from datetime import datetime

from prometheus_client import Counter, Histogram, Gauge

# Request tracking
REQUEST_COUNT = Counter(
    "app_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "app_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

# Model tracking
ACTIVE_MODELS = Gauge(
    "app_active_models",
    "Number of active models",
    ["model_version"],
)

# Drift detection metrics
DETECTION_COUNT = Counter(
    "app_drift_detections_total",
    "Total number of drift detections",
    ["drift_type", "severity", "feature"],
)

DETECTION_DELAY = Histogram(
    "app_drift_detection_delay_seconds",
    "Delay between reference and current batch drift detection",
    ["drift_type"],
)

# Monitoring pipeline metrics
PIPELINE_RUN_COUNT = Counter(
    "app_pipeline_runs_total",
    "Total number of monitoring pipeline runs",
    ["status"],
)

PIPELINE_PROCESSING_TIME = Histogram(
    "app_pipeline_processing_seconds",
    "Monitoring pipeline processing time in seconds",
    ["batch_id"],
)

# Alert metrics
ALERT_COUNT = Counter(
    "app_alerts_total",
    "Total number of alerts",
    ["severity", "alert_type", "status"],
)

ALERT_LIFETIME = Histogram(
    "app_alert_lifetime_seconds",
    "Alert lifecycle duration in seconds",
    ["alert_id"],
)

# Retraining job metrics
RETRAINING_COUNT = Counter(
    "app_retraining_jobs_total",
    "Total number of retraining jobs",
    ["status", "trigger"],
)

RETRAINING_ELIGIBILITY = Counter(
    "app_retraining_eligibility_total",
    "Retraining eligibility evaluation results",
    ["eligible"],
)

# Database metrics
DB_CONNECTIONS = Gauge(
    "app_db_connections",
    "Number of database connections",
)

DB_QUERY_DURATION = Histogram(
    "app_db_query_duration_seconds",
    "Database query duration in seconds",
    ["query_type"],
)

# Version info
APP_VERSION = Gauge(
    "app_version",
    "Application version",
    ["version"],
)

APP_VERSION.set(float("0.1.0"))

# Startup time
START_TIME = datetime.now()


def get_uptime_seconds() -> float:
    """Get system uptime in seconds since startup."""
    return (datetime.now() - START_TIME).total_seconds()