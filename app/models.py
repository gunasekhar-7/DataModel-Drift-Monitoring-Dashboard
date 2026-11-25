"""SQLAlchemy ORM models for the Data & Model Drift Monitoring Platform."""

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)

from app.core.database import Base


class ModelMetadata(Base):
    """Model metadata stored in the database."""

    __tablename__ = "model_metadata"
    # No __table_args__ - schema is determined by DATABASE_URL
    # For SQLite: no schema prefix
    # For PostgreSQL: schema can be set via DATABASE_URL or metadata

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False, unique=True, index=True)
    training_date = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    dataset_name = Column(String(255), nullable=False)
    dataset_version = Column(String(100), nullable=True)
    metrics = Column(JSON, nullable=True)  # accuracy, precision, recall, f1, roc_auc
    preprocessing_config = Column(JSON, nullable=True)
    feature_schema = Column(JSON, nullable=True)
    artifact_path = Column(String(500), nullable=False)
    status = Column(String(50), nullable=False, default="staging", index=True)


class PredictionRequests(Base):
    """Individual prediction request records."""

    __tablename__ = "prediction_requests"
    # No __table_args__ - schema is determined by DATABASE_URL

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(100), unique=True, nullable=False, index=True)
    timestamp = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    model_version = Column(String(50), nullable=False, index=True)
    features = Column(JSON, nullable=False)
    status = Column(String(50), nullable=False, default="completed", index=True)


class PredictionResults(Base):
    """Prediction results linked to prediction requests."""

    __tablename__ = "prediction_results"
    # No __table_args__ - schema is determined by DATABASE_URL

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(
        String(100), nullable=False, index=True
    )  # FK via prediction_requests.request_id
    predicted_class = Column(Integer, nullable=False)
    probability = Column(Float, nullable=True)  # probability of positive class
    prediction_probabilities = Column(JSON, nullable=True)  # full probability vector
    processing_time_ms = Column(Integer, nullable=True)


class BatchMetadata(Base):
    """Batch metadata for simulated production batches."""

    __tablename__ = "batch_metadata"
    # No __table_args__ - schema is determined by DATABASE_URL

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String(100), unique=True, nullable=False, index=True)
    timestamp = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    reference_batch_id = Column(String(100), nullable=True)
    row_count = Column(Integer, nullable=False, default=0)
    scenario = Column(String(100), nullable=True)
    drift_severity = Column(String(50), nullable=True)
    status = Column(String(50), nullable=False, default="generated", index=True)


class PredictionHistory(Base):
    """Append-only prediction history for monitoring and trend analysis."""

    __tablename__ = "prediction_history"
    # No __table_args__ - schema is determined by DATABASE_URL

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(
        String(100), nullable=True, index=True
    )  # FK via batch_metadata.batch_id
    request_id = Column(
        String(100), nullable=True, index=True
    )  # FK via prediction_requests.request_id
    timestamp = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    model_version = Column(String(50), nullable=False, index=True)
    predicted_class = Column(Integer, nullable=False)
    probability = Column(Float, nullable=True)


class DataQualityMetrics(Base):
    """Data quality metrics for monitoring batches."""

    __tablename__ = "data_quality_metrics"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    model_version = Column(String(50), nullable=True, index=True)
    total_rows = Column(Integer, nullable=False)
    total_columns = Column(Integer, nullable=False)
    passed = Column(Boolean, nullable=False, default=True)
    missing_values_count = Column(Integer, nullable=False, default=0)
    duplicate_rows_count = Column(Integer, nullable=False, default=0)
    infinite_values_count = Column(Integer, nullable=False, default=0)
    out_of_range_count = Column(Integer, nullable=False, default=0)
    unexpected_categorical_count = Column(Integer, nullable=False, default=0)
    all_null_columns_count = Column(Integer, nullable=False, default=0)
    invalid_records_count = Column(Integer, nullable=False, default=0)
    details = Column(JSON, nullable=True)  # Full validation results

    __table_args__ = (Index("ix_dq_batch_timestamp", "batch_id", "timestamp"),)


class DriftMetrics(Base):
    """Drift detection metrics for monitoring batches."""

    __tablename__ = "drift_metrics"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String(100), nullable=False, index=True)
    reference_batch_id = Column(String(100), nullable=True, index=True)
    timestamp = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    model_version = Column(String(50), nullable=True, index=True)
    feature_name = Column(String(100), nullable=False, index=True)
    test_name = Column(String(50), nullable=False)  # ks_test, psi, wasserstein
    statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=True)
    threshold = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False)  # no_drift, warning, drift_detected
    overall_drift = Column(Boolean, nullable=False, default=False)
    details = Column(JSON, nullable=True)

    __table_args__ = (Index("ix_drift_batch_feature", "batch_id", "feature_name"),)


class Alert(Base):
    """Alert records for drift and data quality issues."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(50), unique=True, nullable=False, index=True)
    batch_id = Column(String(100), nullable=True, index=True)
    model_version = Column(String(50), nullable=True, index=True)
    timestamp = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    alert_type = Column(String(50), nullable=False, index=True)  # data_quality, drift, performance
    severity = Column(String(20), nullable=False, index=True)  # warning, critical
    feature_name = Column(String(100), nullable=True, index=True)
    detection_method = Column(String(50), nullable=True)
    measured_value = Column(Float, nullable=True)
    threshold_value = Column(Float, nullable=True)
    message = Column(Text, nullable=False)
    explanation = Column(Text, nullable=True)
    recommended_action = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="active", index=True)  # active, acknowledged, resolved
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    acknowledged_by = Column(String(100), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by = Column(String(100), nullable=True)

    __table_args__ = (Index("ix_alert_batch_severity", "batch_id", "severity"),)


class RetrainingJob(Base):
    """Retraining job tracking."""

    __tablename__ = "retraining_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(50), unique=True, nullable=False, index=True)
    trigger_batch_id = Column(String(100), nullable=True, index=True)
    trigger_reason = Column(String(100), nullable=True)
    timestamp = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    status = Column(String(20), nullable=False, default="pending", index=True)  # pending, running, completed, failed
    mlflow_run_id = Column(String(100), nullable=True)
    candidate_model_version = Column(String(50), nullable=True)
    candidate_metrics = Column(JSON, nullable=True)
    champion_model_version = Column(String(50), nullable=True)
    champion_metrics = Column(JSON, nullable=True)
    promotion_decision = Column(String(20), nullable=True)  # promoted, rejected
    promotion_reason = Column(Text, nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)

    __table_args__ = (Index("ix_retraining_status", "status"),)