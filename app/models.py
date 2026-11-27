"""SQLAlchemy ORM models for the Data & Model Drift Monitoring Platform."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ModelMetadata(Base):
    """Model metadata stored in the database."""

    __tablename__ = "model_metadata"
    # No __table_args__ - schema is determined by DATABASE_URL
    # For SQLite: no schema prefix
    # For PostgreSQL: schema can be set via DATABASE_URL or metadata

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False, unique=True, index=True)
    training_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    metrics: Mapped[Any] = mapped_column(JSON, nullable=True)  # accuracy, precision, recall, f1, roc_auc
    preprocessing_config: Mapped[Any] = mapped_column(JSON, nullable=True)
    feature_schema: Mapped[Any] = mapped_column(JSON, nullable=True)
    artifact_path: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="staging", index=True)


class PredictionRequests(Base):
    """Individual prediction request records."""

    __tablename__ = "prediction_requests"
    # No __table_args__ - schema is determined by DATABASE_URL

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    request_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    features: Mapped[Any] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="completed", index=True)


class PredictionResults(Base):
    """Prediction results linked to prediction requests."""

    __tablename__ = "prediction_results"
    # No __table_args__ - schema is determined by DATABASE_URL

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    request_id: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True
    )  # FK via prediction_requests.request_id
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)  # probability of positive class
    prediction_probabilities: Mapped[Any] = mapped_column(JSON, nullable=True)  # full probability vector
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)


class BatchMetadata(Base):
    """Batch metadata for simulated production batches."""

    __tablename__ = "batch_metadata"
    # No __table_args__ - schema is determined by DATABASE_URL

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    batch_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    reference_batch_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    scenario: Mapped[str | None] = mapped_column(String(100), nullable=True)
    drift_severity: Mapped[str | None] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="generated", index=True)


class PredictionHistory(Base):
    """Append-only prediction history for monitoring and trend analysis."""

    __tablename__ = "prediction_history"
    # No __table_args__ - schema is determined by DATABASE_URL

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    batch_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True, index=True
    )  # FK via batch_metadata.batch_id
    request_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True, index=True
    )  # FK via prediction_requests.request_id
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)


class DataQualityMetrics(Base):
    """Data quality metrics for monitoring batches."""

    __tablename__ = "data_quality_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    batch_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    total_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    total_columns: Mapped[int] = mapped_column(Integer, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    missing_values_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicate_rows_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    infinite_values_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    out_of_range_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unexpected_categorical_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    all_null_columns_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    invalid_records_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    details: Mapped[Any] = mapped_column(JSON, nullable=True)  # Full validation results

    __table_args__ = (Index("ix_dq_batch_timestamp", "batch_id", "timestamp"),)


class DriftMetrics(Base):
    """Drift detection metrics for monitoring batches."""

    __tablename__ = "drift_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    batch_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    reference_batch_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    feature_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    test_name: Mapped[str] = mapped_column(String(50), nullable=False)  # ks_test, psi, wasserstein
    statistic: Mapped[float] = mapped_column(Float, nullable=False)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # no_drift, warning, drift_detected
    overall_drift: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    details: Mapped[Any] = mapped_column(JSON, nullable=True)

    __table_args__ = (Index("ix_drift_batch_feature", "batch_id", "feature_name"),)


class Alert(Base):
    """Alert records for drift and data quality issues."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    alert_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    batch_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)  # data_quality, drift, performance
    severity: Mapped[str] = mapped_column(String(20), nullable=False, index=True)  # warning, critical
    feature_name: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    detection_method: Mapped[str | None] = mapped_column(String(50), nullable=True)
    measured_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    recommended_action: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active", index=True)  # active, acknowledged, resolved
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(100), nullable=True)

    __table_args__ = (Index("ix_alert_batch_severity", "batch_id", "severity"),)


class RetrainingJob(Base):
    """Retraining job tracking."""

    __tablename__ = "retraining_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    trigger_batch_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    trigger_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)  # pending, running, completed, failed
    mlflow_run_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    candidate_model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    candidate_metrics: Mapped[Any] = mapped_column(JSON, nullable=True)
    champion_model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    champion_metrics: Mapped[Any] = mapped_column(JSON, nullable=True)
    promotion_decision: Mapped[str | None] = mapped_column(String(20), nullable=True)  # promoted, rejected
    promotion_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_retraining_status", "status"),)
