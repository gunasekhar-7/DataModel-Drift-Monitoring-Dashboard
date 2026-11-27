"""Monitoring API routes for the Data & Model Drift Monitoring Platform."""

from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import Integer
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models import (
    Alert,
    BatchMetadata,
    DataQualityMetrics,
    DriftMetrics,
    ModelMetadata,
    PredictionHistory,
    RetrainingJob,
)
from app.schemas.schemas import HealthCheckResponse

# Router
monitoring_router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


@monitoring_router.get("/health", response_model=HealthCheckResponse)
async def monitoring_health_check(request: Request) -> HealthCheckResponse:
    """Monitoring service health check."""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=settings.app_version,
    )


@monitoring_router.get("/model-health")
async def get_model_health(
    model_version: str | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get overall model health status."""
    # Get active model
    query = db.query(ModelMetadata)
    if model_version:
        query = query.filter(ModelMetadata.version == model_version)
    else:
        query = query.filter(ModelMetadata.status == "production")

    model = query.order_by(ModelMetadata.training_date.desc()).first()

    if not model:
        return {
            "status": "no_model",
            "message": "No production model found",
            "model_version": None,
            "model_status": None,
            "latest_batch": None,
            "latest_monitoring": None,
            "warnings_count": 0,
            "critical_count": 0,
            "overall_status": "unknown"
        }

    # Get latest batch
    latest_batch = db.query(BatchMetadata).order_by(BatchMetadata.timestamp.desc()).first()

    # Get latest monitoring metrics
    latest_dq = db.query(DataQualityMetrics).filter(
        DataQualityMetrics.model_version == model.version
    ).order_by(DataQualityMetrics.timestamp.desc()).first()

    latest_drift = db.query(DriftMetrics).filter(
        DriftMetrics.model_version == model.version
    ).order_by(DriftMetrics.timestamp.desc()).first()

    # Count active alerts
    warnings = db.query(Alert).filter(
        Alert.model_version == model.version,
        Alert.severity == "warning",
        Alert.status == "active"
    ).count()

    critical = db.query(Alert).filter(
        Alert.model_version == model.version,
        Alert.severity == "critical",
        Alert.status == "active"
    ).count()

    # Determine overall status
    if critical > 0:
        overall_status: Literal["critical", "warning", "healthy"] = "critical"
    elif warnings > 0:
        overall_status = "warning"
    else:
        overall_status = "healthy"

    return {
        "status": "ok",
        "model_version": model.version,
        "model_status": model.status,
        "model_name": model.name,
        "training_date": model.training_date.isoformat() if model.training_date else None,
        "model_metrics": model.metrics,
        "latest_batch": {
            "batch_id": latest_batch.batch_id if latest_batch else None,
            "timestamp": latest_batch.timestamp.isoformat() if latest_batch else None,
            "scenario": latest_batch.scenario if latest_batch else None,
            "drift_severity": latest_batch.drift_severity if latest_batch else None,
        } if latest_batch else None,
        "latest_monitoring": {
            "data_quality_timestamp": latest_dq.timestamp.isoformat() if latest_dq else None,
            "drift_timestamp": latest_drift.timestamp.isoformat() if latest_drift else None,
        },
        "warnings_count": warnings,
        "critical_count": critical,
        "overall_status": overall_status,
    }


@monitoring_router.get("/data-quality")
async def get_data_quality_history(
    model_version: str | None = None,
    batch_id: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get data quality metrics history."""
    query = db.query(DataQualityMetrics)

    if model_version:
        query = query.filter(DataQualityMetrics.model_version == model_version)
    if batch_id:
        query = query.filter(DataQualityMetrics.batch_id == batch_id)

    metrics = query.order_by(DataQualityMetrics.timestamp.desc()).limit(limit).all()

    results = []
    for m in metrics:
        results.append({
            "batch_id": m.batch_id,
            "timestamp": m.timestamp.isoformat(),
            "model_version": m.model_version,
            "total_rows": m.total_rows,
            "total_columns": m.total_columns,
            "passed": m.passed,
            "missing_values_count": m.missing_values_count,
            "duplicate_rows_count": m.duplicate_rows_count,
            "infinite_values_count": m.infinite_values_count,
            "out_of_range_count": m.out_of_range_count,
            "unexpected_categorical_count": m.unexpected_categorical_count,
            "all_null_columns_count": m.all_null_columns_count,
            "invalid_records_count": m.invalid_records_count,
            "details": m.details,
        })

    return {
        "count": len(results),
        "results": results,
    }


@monitoring_router.get("/drift")
async def get_drift_history(
    model_version: str | None = None,
    batch_id: str | None = None,
    feature_name: str | None = None,
    test_name: str | None = None,
    severity: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get drift detection metrics history."""
    query = db.query(DriftMetrics)

    if model_version:
        query = query.filter(DriftMetrics.model_version == model_version)
    if batch_id:
        query = query.filter(DriftMetrics.batch_id == batch_id)
    if feature_name:
        query = query.filter(DriftMetrics.feature_name == feature_name)
    if test_name:
        query = query.filter(DriftMetrics.test_name == test_name)
    if severity:
        query = query.filter(DriftMetrics.severity == severity)

    metrics = query.order_by(DriftMetrics.timestamp.desc()).limit(limit).all()

    results = []
    for m in metrics:
        results.append({
            "id": m.id,
            "batch_id": m.batch_id,
            "reference_batch_id": m.reference_batch_id,
            "timestamp": m.timestamp.isoformat(),
            "model_version": m.model_version,
            "feature_name": m.feature_name,
            "test_name": m.test_name,
            "statistic": m.statistic,
            "p_value": m.p_value,
            "threshold": m.threshold,
            "severity": m.severity,
            "overall_drift": m.overall_drift,
            "details": m.details,
        })

    return {
        "count": len(results),
        "results": results,
    }


@monitoring_router.get("/drift/summary")
async def get_drift_summary(
    model_version: str | None = None,
    batch_id: str | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get drift summary by feature and test."""
    query = db.query(DriftMetrics)

    if model_version:
        query = query.filter(DriftMetrics.model_version == model_version)
    if batch_id:
        query = query.filter(DriftMetrics.batch_id == batch_id)

    metrics = query.all()

    # Aggregate by feature
    feature_summary: dict[str, dict[str, Any]] = {}
    test_summary: dict[str, dict[str, Any]] = {}

    for m in metrics:
        # By feature
        if m.feature_name not in feature_summary:
            feature_summary[m.feature_name] = {
                "feature_name": m.feature_name,
                "total_tests": 0,
                "drift_detected": 0,
                "warnings": 0,
                "no_drift": 0,
                "latest_severity": "no_drift",
                "latest_timestamp": None,
            }
        fs = feature_summary[m.feature_name]
        fs["total_tests"] += 1
        if m.severity == "drift_detected":
            fs["drift_detected"] += 1
        elif m.severity == "warning":
            fs["warnings"] += 1
        else:
            fs["no_drift"] += 1

        # Update latest
        if fs["latest_timestamp"] is None or m.timestamp > datetime.fromisoformat(fs["latest_timestamp"].replace("Z", "+00:00")):
            fs["latest_severity"] = m.severity
            fs["latest_timestamp"] = m.timestamp.isoformat()

        # By test
        if m.test_name not in test_summary:
            test_summary[m.test_name] = {
                "test_name": m.test_name,
                "total_tests": 0,
                "drift_detected": 0,
                "warnings": 0,
                "no_drift": 0,
            }
        ts = test_summary[m.test_name]
        ts["total_tests"] += 1
        if m.severity == "drift_detected":
            ts["drift_detected"] += 1
        elif m.severity == "warning":
            ts["warnings"] += 1
        else:
            ts["no_drift"] += 1

    return {
        "total_features_tested": len(feature_summary),
        "by_feature": list(feature_summary.values()),
        "by_test": list(test_summary.values()),
    }


@monitoring_router.get("/predictions")
async def get_prediction_distribution(
    model_version: str | None = None,
    batch_id: str | None = None,
    hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get prediction distribution over time."""
    from sqlalchemy import func

    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    query = db.query(PredictionHistory).filter(PredictionHistory.timestamp >= since)

    if model_version:
        query = query.filter(PredictionHistory.model_version == model_version)
    if batch_id:
        query = query.filter(PredictionHistory.batch_id == batch_id)

    # Aggregate by hour in the database instead of loading all rows
    hourly = query.with_entities(
        func.strftime("%Y-%m-%d %H:00:00", PredictionHistory.timestamp).label("hour"),
        func.count(PredictionHistory.id).label("total"),
        func.sum(func.cast(PredictionHistory.predicted_class == 0, Integer)).label("class_0"),
        func.sum(func.cast(PredictionHistory.predicted_class == 1, Integer)).label("class_1"),
        func.avg(PredictionHistory.probability).label("avg_prob"),
    ).group_by("hour").order_by("hour DESC").all()

    # Build hourly dict from grouped results
    hourly_counts: dict[str, dict[str, int | float]] = {}
    for row in hourly:
        hour_key = row.hour
        total = int(row.total) if row.total else 0
        c0 = int(row.class_0) if row.class_0 else 0
        c1 = int(row.class_1) if row.class_1 else 0
        avg_p = float(row.avg_prob) if row.avg_prob is not None else 0.0
        hourly_counts[hour_key] = {
            "total": total,
            "class_0": c0,
            "class_1": c1,
            "avg_probability": avg_p,
        }

    # Calculate averages per hour
    for hour_key in hourly_counts:
        if hourly_counts[hour_key]["total"] > 0:
            hourly_counts[hour_key]["avg_probability"] /= hourly_counts[hour_key]["total"]

    # Overall distribution from aggregated data
    total = sum(int(h["total"]) for h in hourly_counts.values())
    class_0 = sum(int(h["class_0"]) for h in hourly_counts.values())
    class_1 = sum(int(h["class_1"]) for h in hourly_counts.values())
    avg_prob = sum(float(h["avg_probability"]) * int(h["total"]) for h in hourly_counts.values()) / total if total > 0 else 0.0

    return {
        "period_hours": hours,
        "total_predictions": total,
        "class_distribution": {
            "class_0": class_0,
            "class_1": class_1,
            "class_0_pct": round(class_0 / total * 100, 2) if total > 0 else 0,
            "class_1_pct": round(class_1 / total * 100, 2) if total > 0 else 0,
        },
        "average_probability": round(avg_prob, 4),
        "hourly": [
            {"hour": k, **v} for k, v in sorted(hourly_counts.items())
        ],
    }


@monitoring_router.get("/performance")
async def get_model_performance(
    model_version: str | None = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get model performance metrics."""
    query = db.query(ModelMetadata)

    if model_version:
        query = query.filter(ModelMetadata.version == model_version)

    models = query.order_by(ModelMetadata.training_date.desc()).all()

    results = []
    for m in models:
        results.append({
            "model_version": m.version,
            "model_name": m.name,
            "status": m.status,
            "training_date": m.training_date.isoformat() if m.training_date else None,
            "metrics": m.metrics,
            "dataset_name": m.dataset_name,
            "dataset_version": m.dataset_version,
        })

    return {
        "count": len(results),
        "models": results,
    }


@monitoring_router.get("/alerts")
async def get_alerts(
    model_version: str | None = None,
    batch_id: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    alert_type: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get alerts history."""
    query = db.query(Alert)

    if model_version:
        query = query.filter(Alert.model_version == model_version)
    if batch_id:
        query = query.filter(Alert.batch_id == batch_id)
    if severity:
        query = query.filter(Alert.severity == severity)
    if status:
        query = query.filter(Alert.status == status)
    if alert_type:
        query = query.filter(Alert.alert_type == alert_type)

    alerts = query.order_by(Alert.timestamp.desc()).limit(limit).all()

    results = []
    for a in alerts:
        results.append({
            "alert_id": a.alert_id,
            "batch_id": a.batch_id,
            "model_version": a.model_version,
            "timestamp": a.timestamp.isoformat(),
            "alert_type": a.alert_type,
            "severity": a.severity,
            "feature_name": a.feature_name,
            "detection_method": a.detection_method,
            "measured_value": a.measured_value,
            "threshold_value": a.threshold_value,
            "message": a.message,
            "explanation": a.explanation,
            "recommended_action": a.recommended_action,
            "status": a.status,
            "acknowledged_at": a.acknowledged_at.isoformat() if a.acknowledged_at else None,
            "acknowledged_by": a.acknowledged_by,
            "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
            "resolved_by": a.resolved_by,
        })

    return {
        "count": len(results),
        "results": results,
    }


@monitoring_router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Acknowledge an alert."""
    alert = db.query(Alert).filter(Alert.alert_id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.status = "acknowledged"
    alert.acknowledged_at = datetime.now(timezone.utc)
    alert.acknowledged_by = acknowledged_by
    db.commit()

    return {"status": "acknowledged", "alert_id": alert_id}


@monitoring_router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    resolved_by: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Resolve an alert."""
    alert = db.query(Alert).filter(Alert.alert_id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.status = "resolved"
    alert.resolved_at = datetime.now(timezone.utc)
    alert.resolved_by = resolved_by
    db.commit()

    return {"status": "resolved", "alert_id": alert_id}


@monitoring_router.get("/retraining")
@monitoring_router.get("/retraining/jobs")
async def get_retraining_jobs(
    status: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Get retraining job history."""
    query = db.query(RetrainingJob)

    if status:
        query = query.filter(RetrainingJob.status == status)

    jobs = query.order_by(RetrainingJob.timestamp.desc()).limit(limit).all()

    results = []
    for j in jobs:
        results.append({
            "job_id": j.job_id,
            "trigger_batch_id": j.trigger_batch_id,
            "trigger_reason": j.trigger_reason,
            "timestamp": j.timestamp.isoformat(),
            "status": j.status,
            "mlflow_run_id": j.mlflow_run_id,
            "candidate_model_version": j.candidate_model_version,
            "candidate_metrics": j.candidate_metrics,
            "champion_model_version": j.champion_model_version,
            "champion_metrics": j.champion_metrics,
            "promotion_decision": j.promotion_decision,
            "promotion_reason": j.promotion_reason,
            "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            "error_message": j.error_message,
        })

    return {
        "count": len(results),
        "results": results,
    }


@monitoring_router.post("/retraining/trigger")
async def trigger_retraining(
    trigger_batch_id: str,
    trigger_reason: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Manually trigger a retraining job."""
    import uuid

    job_id = f"retrain-{uuid.uuid4().hex[:8]}"

    job = RetrainingJob(
        job_id=job_id,
        trigger_batch_id=trigger_batch_id,
        trigger_reason=trigger_reason,
        status="pending",
    )
    db.add(job)
    db.commit()

    return {
        "status": "triggered",
        "job_id": job_id,
        "message": "Retraining job queued. Will be picked up by the retraining orchestrator."
    }
