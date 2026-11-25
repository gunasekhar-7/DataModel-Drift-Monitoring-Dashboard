"""Monitoring pipeline that orchestrates data quality, drift detection, and alerting."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import (
    BatchMetadata,
    DataQualityMetrics,
    DriftMetrics,
    ModelMetadata,
)
from app.monitoring.alerting import AlertContext, AlertEngine, create_alert_engine
from app.monitoring.data_quality import create_engine_from_reference
from app.monitoring.drift.numerical import NumericalDriftDetector
from app.monitoring.notifications import (
    NotificationService,
    create_notification_service,
)

logger = logging.getLogger(__name__)


@dataclass
class MonitoringResult:
    """Result of monitoring pipeline execution."""
    batch_id: str
    timestamp: datetime
    data_quality_report: Any | None = None
    drift_report: Any | None = None
    alerts_generated: list[str] = None
    success: bool = True
    error: str | None = None


class MonitoringPipeline:
    """Main monitoring pipeline orchestrator."""
    
    def __init__(
        self,
        db: Session,
        reference_data: pd.DataFrame | None = None,
        reference_batch_id: str | None = None,
        notification_service: NotificationService | None = None,
        alert_engine: AlertEngine | None = None,
    ):
        """Initialize the monitoring pipeline.
        
        Args:
            db: Database session
            reference_data: Reference dataset for drift detection (loads from default if None)
            reference_batch_id: Reference batch ID
            notification_service: Notification service for alerts
            alert_engine: Alert engine (created if None)
        """
        self.db = db
        self.reference_batch_id = reference_batch_id
        
        # Load reference data if not provided
        if reference_data is not None:
            self.reference_data = reference_data
        else:
            self.reference_data = self._load_reference_data()
        
        # Initialize engines
        self.dq_engine = create_engine_from_reference(self.reference_data)
        self.drift_detector = NumericalDriftDetector()
        self.notification_service = notification_service or create_notification_service()
        self.alert_engine = alert_engine or create_alert_engine(
            db=db, notification_service=self.notification_service
        )
        
        # Numeric features for drift detection
        self.numeric_features = [
            col for col in self.reference_data.columns
            if pd.api.types.is_numeric_dtype(self.reference_data[col])
        ]
    
    def _load_reference_data(self) -> pd.DataFrame:
        """Load reference dataset."""
        try:
            path = settings.simulator_reference_data_path
            if not path or not pd.io.common.file_exists(path):
                path = "data/raw/Telco-Customer-Churn.csv"
            return pd.read_csv(path)
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
            # Return empty dataframe as fallback
            return pd.DataFrame()
    
    def process_batch(
        self,
        batch_df: pd.DataFrame,
        batch_id: str | None = None,
        model_version: str | None = None,
        scenario: str | None = None,
    ) -> MonitoringResult:
        """Process a production batch through the full monitoring pipeline.
        
        Args:
            batch_df: Production batch DataFrame
            batch_id: Batch identifier (generated if None)
            model_version: Model version used for predictions
            scenario: Scenario name if this is a simulated batch
            
        Returns:
            MonitoringResult with all monitoring outputs
        """
        if batch_id is None:
            batch_id = f"batch-{uuid.uuid4().hex[:8]}"
        
        timestamp = datetime.now(timezone.utc)
        
        logger.info(f"Processing batch {batch_id} with {len(batch_df)} rows")
        
        result = MonitoringResult(
            batch_id=batch_id,
            timestamp=timestamp,
            alerts_generated=[],
        )
        
        try:
            # 1. Store batch metadata
            self._store_batch_metadata(batch_id, batch_df, timestamp, scenario)
            
            # 2. Run data quality checks
            logger.info(f"Batch {batch_id}: Running data quality checks...")
            dq_report = self.dq_engine.validate(batch_df, batch_id)
            result.data_quality_report = dq_report
            self._store_data_quality_metrics(dq_report, model_version)
            
            # 3. Run drift detection
            logger.info(f"Batch {batch_id}: Running drift detection...")
            drift_report = self.drift_detector.detect_batch(
                reference_df=self.reference_data,
                current_df=batch_df,
                feature_names=self.numeric_features,
                batch_id=batch_id,
                reference_batch_id=self.reference_batch_id,
            )
            result.drift_report = drift_report
            self._store_drift_metrics(drift_report, model_version)
            
            # 4. Get model version if not provided
            if not model_version:
                model_version = self._get_latest_model_version()
            
            # 5. Generate alerts
            logger.info(f"Batch {batch_id}: Evaluating alerts...")
            alerts = self._evaluate_alerts(batch_id, model_version, timestamp, dq_report, drift_report)
            result.alerts_generated = [a.alert_id for a in alerts]
            
            logger.info(f"Batch {batch_id}: Monitoring complete. {len(alerts)} alerts generated.")
            
        except Exception as e:
            logger.error(f"Batch {batch_id}: Monitoring failed: {e}")
            result.success = False
            result.error = str(e)
        
        return result
    
    def _store_batch_metadata(
        self,
        batch_id: str,
        batch_df: pd.DataFrame,
        timestamp: datetime,
        scenario: str | None,
    ) -> None:
        """Store batch metadata in database."""
        batch_meta = BatchMetadata(
            batch_id=batch_id,
            timestamp=timestamp,
            reference_batch_id=self.reference_batch_id,
            row_count=len(batch_df),
            scenario=scenario,
            drift_severity=None,  # Will be updated after drift detection
            status="processed",
        )
        self.db.add(batch_meta)
        self.db.commit()
    
    def _store_data_quality_metrics(
        self,
        dq_report: Any,
        model_version: str | None,
    ) -> None:
        """Store data quality metrics in database."""
        # Count issues by type
        missing_count = sum(
            r.affected_rows for r in dq_report.results
            if r.validation_type.value == "missing_values"
        )
        duplicate_count = sum(
            r.affected_rows for r in dq_report.results
            if r.validation_type.value == "duplicates"
        )
        infinite_count = sum(
            r.affected_rows for r in dq_report.results
            if r.validation_type.value == "infinite_values"
        )
        oob_count = sum(
            r.affected_rows for r in dq_report.results
            if r.validation_type.value == "out_of_range"
        )
        unexpected_cat_count = sum(
            r.affected_rows for r in dq_report.results
            if r.validation_type.value == "unexpected_categorical"
        )
        all_null_count = sum(
            r.affected_rows for r in dq_report.results
            if r.validation_type.value == "all_null_columns"
        )
        invalid_count = sum(
            r.affected_rows for r in dq_report.results
            if r.validation_type.value == "invalid_records"
        )
        
        dq_metrics = DataQualityMetrics(
            batch_id=dq_report.batch_id,
            timestamp=datetime.fromisoformat(dq_report.timestamp.replace("Z", "+00:00")),
            model_version=model_version,
            total_rows=dq_report.total_rows,
            total_columns=dq_report.total_columns,
            passed=dq_report.passed,
            missing_values_count=missing_count,
            duplicate_rows_count=duplicate_count,
            infinite_values_count=infinite_count,
            out_of_range_count=oob_count,
            unexpected_categorical_count=unexpected_cat_count,
            all_null_columns_count=all_null_count,
            invalid_records_count=invalid_count,
            details=dq_report.to_dict(),
        )
        self.db.add(dq_metrics)
        self.db.commit()
    
    def _store_drift_metrics(
        self,
        drift_report: Any,
        model_version: str | None,
    ) -> None:
        """Store drift metrics in database."""
        for dm in drift_report.results:
            drift_metric = DriftMetrics(
                batch_id=drift_report.batch_id,
                reference_batch_id=drift_report.reference_batch_id,
                timestamp=datetime.now(timezone.utc),
                model_version=model_version,
                feature_name=dm.feature_name,
                test_name=dm.test_name,
                statistic=dm.statistic,
                p_value=dm.p_value,
                threshold=dm.threshold,
                severity=dm.severity.value,
                overall_drift=drift_report.overall_drift,
                details=dm.details,
            )
            self.db.add(drift_metric)
        
        # Update batch metadata with drift severity
        batch_meta = self.db.query(BatchMetadata).filter(
            BatchMetadata.batch_id == drift_report.batch_id
        ).first()
        
        if batch_meta:
            if drift_report.overall_drift:
                # Determine severity from results
                critical = any(r.severity.value == "drift_detected" for r in drift_report.results)
                batch_meta.drift_severity = "critical" if critical else "warning"
            else:
                batch_meta.drift_severity = "none"
        
        self.db.commit()
    
    def _get_latest_model_version(self) -> str | None:
        """Get the latest production model version."""
        model = self.db.query(ModelMetadata).filter(
            ModelMetadata.status == "production"
        ).order_by(ModelMetadata.training_date.desc()).first()
        
        return model.version if model else None
    
    def _evaluate_alerts(
        self,
        batch_id: str,
        model_version: str,
        timestamp: datetime,
        dq_report: Any,
        drift_report: Any,
    ) -> list[Any]:
        """Evaluate and generate alerts."""
        # Get stored metrics
        dq_metrics = self.db.query(DataQualityMetrics).filter(
            DataQualityMetrics.batch_id == batch_id
        ).first()
        
        drift_metrics = self.db.query(DriftMetrics).filter(
            DriftMetrics.batch_id == batch_id
        ).all()
        
        model_meta = None
        if model_version:
            model_meta = self.db.query(ModelMetadata).filter(
                ModelMetadata.version == model_version
            ).first()
        
        # Create alert context
        context = AlertContext(
            batch_id=batch_id,
            model_version=model_version,
            timestamp=timestamp,
            data_quality_metrics=dq_metrics,
            drift_metrics=drift_metrics,
            model_metadata=model_meta,
        )
        
        # Generate alerts
        alerts = self.alert_engine.evaluate(context)
        
        return alerts


def create_monitoring_pipeline(
    db: Session,
    reference_data: pd.DataFrame | None = None,
    notification_service: NotificationService | None = None,
) -> MonitoringPipeline:
    """Factory function to create MonitoringPipeline instance."""
    return MonitoringPipeline(
        db=db,
        reference_data=reference_data,
        notification_service=notification_service,
    )