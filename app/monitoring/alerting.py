"""Alerting engine for the drift monitoring platform."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from app.models import Alert, DataQualityMetrics, DriftMetrics, ModelMetadata


class AlertType(str, Enum):
    """Types of alerts."""
    DATA_QUALITY = "data_quality"
    DRIFT = "drift"
    PERFORMANCE = "performance"
    RETRAINING = "retraining"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert lifecycle status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class AlertRule:
    """Configuration for an alert rule."""
    name: str
    alert_type: AlertType
    severity: AlertSeverity
    condition: str  # Human-readable condition description
    threshold: float
    feature_specific: bool = False
    feature_names: list[str] = field(default_factory=list)
    cooldown_minutes: int = 60  # Minimum time between duplicate alerts
    enabled: bool = True


@dataclass
class AlertContext:
    """Context information for alert evaluation."""
    batch_id: str
    model_version: str
    timestamp: datetime
    data_quality_metrics: DataQualityMetrics | None = None
    drift_metrics: list[DriftMetrics] = field(default_factory=list)
    model_metadata: ModelMetadata | None = None


class AlertEngine:
    """Engine for evaluating monitoring data and generating alerts."""
    
    DEFAULT_RULES = [
        # Data Quality Rules
        AlertRule(
            name="high_missing_values",
            alert_type=AlertType.DATA_QUALITY,
            severity=AlertSeverity.WARNING,
            condition="Missing values exceed 10% of rows",
            threshold=0.10,
            feature_specific=True,
        ),
        AlertRule(
            name="critical_missing_values",
            alert_type=AlertType.DATA_QUALITY,
            severity=AlertSeverity.CRITICAL,
            condition="Missing values exceed 25% of rows",
            threshold=0.25,
            feature_specific=True,
        ),
        AlertRule(
            name="high_duplicate_rows",
            alert_type=AlertType.DATA_QUALITY,
            severity=AlertSeverity.WARNING,
            condition="Duplicate rows exceed 5% of batch",
            threshold=0.05,
            feature_specific=False,
        ),
        AlertRule(
            name="infinite_values_detected",
            alert_type=AlertType.DATA_QUALITY,
            severity=AlertSeverity.CRITICAL,
            condition="Infinite values detected in numeric columns",
            threshold=0,
            feature_specific=True,
        ),
        AlertRule(
            name="out_of_range_values",
            alert_type=AlertType.DATA_QUALITY,
            severity=AlertSeverity.WARNING,
            condition="Out-of-range values exceed 5% of rows",
            threshold=0.05,
            feature_specific=True,
        ),
        AlertRule(
            name="unexpected_categorical_values",
            alert_type=AlertType.DATA_QUALITY,
            severity=AlertSeverity.WARNING,
            condition="Unexpected categorical values detected",
            threshold=0.01,
            feature_specific=True,
        ),
        AlertRule(
            name="all_null_columns",
            alert_type=AlertType.DATA_QUALITY,
            severity=AlertSeverity.CRITICAL,
            condition="Columns with all null values detected",
            threshold=0,
            feature_specific=True,
        ),
        
        # Drift Rules
        AlertRule(
            name="drift_detected_ks",
            alert_type=AlertType.DRIFT,
            severity=AlertSeverity.WARNING,
            condition="KS test p-value below 0.05",
            threshold=0.05,
            feature_specific=True,
        ),
        AlertRule(
            name="drift_detected_psi_warning",
            alert_type=AlertType.DRIFT,
            severity=AlertSeverity.WARNING,
            condition="PSI between 0.1 and 0.25 (moderate drift)",
            threshold=0.1,
            feature_specific=True,
        ),
        AlertRule(
            name="drift_detected_psi_critical",
            alert_type=AlertType.DRIFT,
            severity=AlertSeverity.CRITICAL,
            condition="PSI >= 0.25 (significant drift)",
            threshold=0.25,
            feature_specific=True,
        ),
        AlertRule(
            name="drift_detected_wasserstein",
            alert_type=AlertType.DRIFT,
            severity=AlertSeverity.WARNING,
            condition="Wasserstein distance exceeds threshold",
            threshold=1.0,  # Will be auto-scaled
            feature_specific=True,
        ),
        
        # Aggregate drift rules
        AlertRule(
            name="multiple_features_drifting",
            alert_type=AlertType.DRIFT,
            severity=AlertSeverity.CRITICAL,
            condition="3+ features showing drift simultaneously",
            threshold=3,
            feature_specific=False,
        ),
    ]
    
    def __init__(
        self,
        db: Session,
        rules: list[AlertRule] | None = None,
        notification_service: Any | None = None,
    ):
        """Initialize the alert engine.
        
        Args:
            db: Database session
            rules: Custom alert rules (uses defaults if None)
            notification_service: Optional notification service for sending alerts
        """
        self.db = db
        self.rules = rules or self.DEFAULT_RULES
        self.notification_service = notification_service
    
    def evaluate(self, context: AlertContext) -> list[Alert]:
        """Evaluate monitoring context and generate alerts.
        
        Args:
            context: Monitoring context with batch data and metrics
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        # Evaluate data quality rules
        if context.data_quality_metrics:
            dq_alerts = self._evaluate_data_quality(context)
            alerts.extend(dq_alerts)
        
        # Evaluate drift rules
        if context.drift_metrics:
            drift_alerts = self._evaluate_drift(context)
            alerts.extend(drift_alerts)
        
        # Filter out None (suppressed duplicate) alerts
        alerts = [a for a in alerts if a is not None]

        # Persist alerts
        for alert in alerts:
            self._persist_alert(alert)

        # Send notifications
        if self.notification_service and alerts:
            self.notification_service.send_alerts(alerts)

        return alerts
    
    def _evaluate_data_quality(self, context: AlertContext) -> list[Alert]:
        """Evaluate data quality metrics against rules."""
        alerts = []
        dq = context.data_quality_metrics
        
        if not dq:
            return alerts
        
        total_rows = dq.total_rows
        if total_rows == 0:
            return alerts
        
        # Check missing values per column (from details)
        if dq.details:
            for result in dq.details.get("results", []):
                if result.get("validation_type") == "missing_values":
                    col = result.get("column")
                    pct = result.get("affected_percentage", 0) / 100
                    
                    # Check warning threshold
                    if pct >= 0.10:
                        severity = AlertSeverity.CRITICAL if pct >= 0.25 else AlertSeverity.WARNING
                        alerts.append(self._create_alert(
                            context=context,
                            rule_name=f"missing_values_{col}",
                            alert_type=AlertType.DATA_QUALITY,
                            severity=severity,
                            feature_name=col,
                            detection_method="data_quality_validation",
                            measured_value=pct,
                            threshold_value=0.25 if severity == AlertSeverity.CRITICAL else 0.10,
                            message=f"Column '{col}' has {pct:.1%} missing values",
                            explanation=f"Missing value percentage ({pct:.1%}) exceeds threshold",
                            recommended_action="Investigate data pipeline for source of missing values. Consider imputation strategy.",
                        ))
                
                elif result.get("validation_type") == "duplicates":
                    pct = result.get("affected_percentage", 0) / 100
                    if pct >= 0.05:
                        alerts.append(self._create_alert(
                            context=context,
                            rule_name="duplicate_rows",
                            alert_type=AlertType.DATA_QUALITY,
                            severity=AlertSeverity.WARNING,
                            feature_name=None,
                            detection_method="data_quality_validation",
                            measured_value=pct,
                            threshold_value=0.05,
                            message=f"Batch contains {pct:.1%} duplicate rows",
                            explanation="Duplicate rows may indicate data pipeline issues or double-counting",
                            recommended_action="Review data ingestion pipeline for deduplication logic",
                        ))
                
                elif result.get("validation_type") == "infinite_values":
                    col = result.get("column")
                    count = result.get("affected_rows", 0)
                    if count > 0:
                        alerts.append(self._create_alert(
                            context=context,
                            rule_name=f"infinite_values_{col}",
                            alert_type=AlertType.DATA_QUALITY,
                            severity=AlertSeverity.CRITICAL,
                            feature_name=col,
                            detection_method="data_quality_validation",
                            measured_value=count,
                            threshold_value=0,
                            message=f"Column '{col}' contains {count} infinite values",
                            explanation="Infinite values will break model inference and downstream processing",
                            recommended_action="Fix data source or add validation to replace infinite values",
                        ))
                
                elif result.get("validation_type") == "out_of_range":
                    col = result.get("column")
                    pct = result.get("affected_percentage", 0) / 100
                    if pct >= 0.05:
                        alerts.append(self._create_alert(
                            context=context,
                            rule_name=f"out_of_range_{col}",
                            alert_type=AlertType.DATA_QUALITY,
                            severity=AlertSeverity.WARNING,
                            feature_name=col,
                            detection_method="data_quality_validation",
                            measured_value=pct,
                            threshold_value=0.05,
                            message=f"Column '{col}' has {pct:.1%} out-of-range values",
                            explanation="Values fall outside expected range from reference data",
                            recommended_action="Investigate data source for sensor errors or schema changes",
                        ))
                
                elif result.get("validation_type") == "unexpected_categorical":
                    col = result.get("column")
                    pct = result.get("affected_percentage", 0) / 100
                    unexpected = result.get("details", {}).get("unexpected_categories", [])
                    if pct >= 0.01:
                        alerts.append(self._create_alert(
                            context=context,
                            rule_name=f"unexpected_categorical_{col}",
                            alert_type=AlertType.DATA_QUALITY,
                            severity=AlertSeverity.WARNING,
                            feature_name=col,
                            detection_method="data_quality_validation",
                            measured_value=pct,
                            threshold_value=0.01,
                            message=f"Column '{col}' has unexpected categories: {unexpected}",
                            explanation=f"New categorical values not seen in training data ({pct:.1%} of rows)",
                            recommended_action="Update categorical encoder or investigate data source changes",
                        ))
                
                elif result.get("validation_type") == "all_null_columns":
                    col = result.get("column")
                    alerts.append(self._create_alert(
                        context=context,
                        rule_name=f"all_null_{col}",
                        alert_type=AlertType.DATA_QUALITY,
                        severity=AlertSeverity.CRITICAL,
                        feature_name=col,
                        detection_method="data_quality_validation",
                        measured_value=1.0,
                        threshold_value=0,
                        message=f"Column '{col}' is entirely null",
                        explanation="Column contains no usable data for model inference",
                        recommended_action="Remove column from pipeline or investigate data source",
                    ))
        
        return alerts
    
    def _evaluate_drift(self, context: AlertContext) -> list[Alert]:
        """Evaluate drift metrics against rules."""
        alerts = []
        drift_metrics = context.drift_metrics
        
        if not drift_metrics:
            return alerts
        
        # Group by feature
        by_feature: dict[str, list[DriftMetrics]] = {}
        for dm in drift_metrics:
            if dm.feature_name not in by_feature:
                by_feature[dm.feature_name] = []
            by_feature[dm.feature_name].append(dm)
        
        drift_detected_features = set()
        
        for feature_name, metrics in by_feature.items():
            for dm in metrics:
                # KS test
                if dm.test_name == "kolmogorov_smirnov":
                    if dm.p_value is not None and dm.p_value < 0.05:
                        severity = AlertSeverity.CRITICAL if dm.p_value < 0.001 else AlertSeverity.WARNING
                        alerts.append(self._create_alert(
                            context=context,
                            rule_name=f"drift_ks_{feature_name}",
                            alert_type=AlertType.DRIFT,
                            severity=severity,
                            feature_name=feature_name,
                            detection_method="kolmogorov_smirnov",
                            measured_value=dm.p_value,
                            threshold_value=0.05,
                            message=f"KS test detects distribution drift in '{feature_name}' (p={dm.p_value:.4f})",
                            explanation=f"Feature distribution has significantly changed from reference (p-value: {dm.p_value:.4f})",
                            recommended_action="Investigate feature drift cause. Consider retraining if persistent.",
                        ))
                        drift_detected_features.add(feature_name)
                
                # PSI
                elif dm.test_name == "population_stability_index":
                    if dm.statistic >= 0.25:
                        alerts.append(self._create_alert(
                            context=context,
                            rule_name=f"drift_psi_critical_{feature_name}",
                            alert_type=AlertType.DRIFT,
                            severity=AlertSeverity.CRITICAL,
                            feature_name=feature_name,
                            detection_method="population_stability_index",
                            measured_value=dm.statistic,
                            threshold_value=0.25,
                            message=f"PSI indicates significant drift in '{feature_name}' (PSI={dm.statistic:.3f})",
                            explanation="Population Stability Index shows major distribution shift (PSI >= 0.25)",
                            recommended_action="Immediate investigation required. Model retraining likely needed.",
                        ))
                        drift_detected_features.add(feature_name)
                    elif dm.statistic >= 0.1:
                        alerts.append(self._create_alert(
                            context=context,
                            rule_name=f"drift_psi_warning_{feature_name}",
                            alert_type=AlertType.DRIFT,
                            severity=AlertSeverity.WARNING,
                            feature_name=feature_name,
                            detection_method="population_stability_index",
                            measured_value=dm.statistic,
                            threshold_value=0.1,
                            message=f"PSI indicates moderate drift in '{feature_name}' (PSI={dm.statistic:.3f})",
                            explanation="Population Stability Index shows moderate distribution shift (0.1 <= PSI < 0.25)",
                            recommended_action="Monitor closely. Investigate if trend continues.",
                        ))
                        drift_detected_features.add(feature_name)
                
                # Wasserstein
                elif dm.test_name == "wasserstein_distance":
                    if dm.statistic > dm.threshold:
                        alerts.append(self._create_alert(
                            context=context,
                            rule_name=f"drift_wasserstein_{feature_name}",
                            alert_type=AlertType.DRIFT,
                            severity=AlertSeverity.WARNING,
                            feature_name=feature_name,
                            detection_method="wasserstein_distance",
                            measured_value=dm.statistic,
                            threshold_value=dm.threshold,
                            message=f"Wasserstein distance indicates drift in '{feature_name}' (distance={dm.statistic:.3f})",
                            explanation="Earth Mover's Distance exceeds threshold, indicating distribution shift",
                            recommended_action="Review feature distribution changes. Consider retraining.",
                        ))
                        drift_detected_features.add(feature_name)
        
        # Aggregate rule: multiple features drifting
        if len(drift_detected_features) >= 3:
            alerts.append(self._create_alert(
                context=context,
                rule_name="multiple_features_drifting",
                alert_type=AlertType.DRIFT,
                severity=AlertSeverity.CRITICAL,
                feature_name=None,
                detection_method="aggregate_drift_detection",
                measured_value=len(drift_detected_features),
                threshold_value=3,
                message=f"Multiple features ({len(drift_detected_features)}) showing simultaneous drift: {', '.join(sorted(drift_detected_features))}",
                explanation="Coordinated drift across multiple features suggests systematic data change",
                recommended_action="Urgent investigation. Schedule model retraining with updated data.",
            ))
        
        return alerts
    
    def _create_alert(
        self,
        context: AlertContext,
        rule_name: str,
        alert_type: AlertType,
        severity: AlertSeverity,
        feature_name: str | None,
        detection_method: str,
        measured_value: float,
        threshold_value: float,
        message: str,
        explanation: str,
        recommended_action: str,
    ) -> Alert:
        """Create an alert object."""
        alert_id = f"alt-{uuid.uuid4().hex[:10]}"
        
        # Check for duplicate alerts (cooldown)
        existing = self.db.query(Alert).filter(
            Alert.batch_id == context.batch_id,
            Alert.feature_name == feature_name,
            Alert.detection_method == detection_method,
            Alert.status.in_([AlertStatus.ACTIVE.value, AlertStatus.ACKNOWLEDGED.value]),
        ).first()
        
        if existing:
            # Check cooldown
            rule = next(
                (r for r in self.rules if r.name == rule_name or rule_name.startswith(r.name)),
                None,
            )
            cooldown_minutes = rule.cooldown_minutes if rule else 60
            existing_ts = existing.timestamp
            if existing_ts.tzinfo is None:
                existing_ts = existing_ts.replace(tzinfo=timezone.utc)
            now_ts = datetime.now(timezone.utc)
            cooldown_delta = now_ts - existing_ts
            if cooldown_delta.total_seconds() < cooldown_minutes * 60:
                return None  # Suppress duplicate
        
        # Ensure alert timestamp is timezone-aware
        alert_ts = context.timestamp
        if alert_ts.tzinfo is None:
            alert_ts = alert_ts.replace(tzinfo=timezone.utc)

        alert = Alert(
            alert_id=alert_id,
            batch_id=context.batch_id,
            model_version=context.model_version,
            timestamp=alert_ts,
            alert_type=alert_type.value,
            severity=severity.value,
            feature_name=feature_name,
            detection_method=detection_method,
            measured_value=measured_value,
            threshold_value=threshold_value,
            message=message,
            explanation=explanation,
            recommended_action=recommended_action,
            status=AlertStatus.ACTIVE.value,
        )
        
        return alert
    
    def _persist_alert(self, alert: Alert | None) -> None:
        """Persist alert to database."""
        if alert is None:
            return
        try:
            self.db.add(alert)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            # Log error but don't fail
            import logging
            logging.getLogger(__name__).error(f"Failed to persist alert: {e}")


def create_alert_engine(
    db: Session,
    notification_service: Any | None = None,
    custom_rules: list[AlertRule] | None = None,
) -> AlertEngine:
    """Factory function to create an AlertEngine instance."""
    return AlertEngine(db=db, rules=custom_rules, notification_service=notification_service)