"""Automated retraining orchestrator for the drift monitoring platform."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import (
    Alert,
    ModelMetadata,
    PredictionHistory,
    RetrainingJob,
)
from app.monitoring.alerting import AlertEngine
from app.monitoring.mlflow_integration import MLflowManager, create_mlflow_manager

logger = logging.getLogger(__name__)


class RetrainingTrigger(str, Enum):
    """Types of retraining triggers."""
    SUSTAINED_CRITICAL_DRIFT = "sustained_critical_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SUFFICIENT_NEW_DATA = "sufficient_new_data"
    MANUAL = "manual"
    SCHEDULED = "scheduled"


class RetrainingStatus(str, Enum):
    """Retraining job statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RetrainingEligibility:
    """Result of retraining eligibility check."""
    eligible: bool
    trigger: RetrainingTrigger | None
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrainingConfig:
    """Configuration for retraining orchestrator."""
    # Eligibility thresholds
    critical_drift_count_threshold: int = 3  # Number of critical drift alerts in window
    drift_window_hours: int = 24  # Time window for sustained drift
    performance_degradation_threshold: float = 0.05  # 5% drop in key metrics
    min_new_labeled_samples: int = 1000  # Minimum new labeled samples
    data_window_days: int = 30  # Days of new data to consider

    # Cooldown
    retraining_cooldown_hours: int = 48  # Minimum time between retraining jobs

    # Training config
    training_script: str = "scripts/train_model.py"
    model_name: str = "baseline_churn_model"
    mlflow_experiment: str = "drift_monitoring"

    # Promotion thresholds
    min_accuracy: float = 0.60
    min_roc_auc: float = 0.55
    max_metric_regression: float = 0.02  # Max allowed regression vs champion


class RetrainingOrchestrator:
    """Orchestrates automated model retraining based on monitoring signals."""

    def __init__(
        self,
        db: Session,
        config: RetrainingConfig | None = None,
        mlflow_manager: MLflowManager | None = None,
        alert_engine: AlertEngine | None = None,
    ):
        """Initialize the retraining orchestrator.

        Args:
            db: Database session
            config: Retraining configuration
            mlflow_manager: MLflow manager for experiment tracking
            alert_engine: Alert engine for checking alerts
        """
        self.db = db
        self.config = config or RetrainingConfig()
        self.mlflow_manager = mlflow_manager or create_mlflow_manager(
            experiment_name=self.config.mlflow_experiment
        )
        self.alert_engine = alert_engine

    def check_eligibility(self) -> RetrainingEligibility:
        """Check if retraining should be triggered.

        Returns:
            RetrainingEligibility with decision and reasoning
        """
        # Check cooldown
        if not self._check_cooldown():
            return RetrainingEligibility(
                eligible=False,
                trigger=None,
                reason="Retraining cooldown period active",
                details={"cooldown_hours": self.config.retraining_cooldown_hours}
            )

        # Check sustained critical drift
        drift_check = self._check_sustained_critical_drift()
        if drift_check.eligible:
            return drift_check

        # Check performance degradation
        perf_check = self._check_performance_degradation()
        if perf_check.eligible:
            return perf_check

        # Check sufficient new labeled data
        data_check = self._check_sufficient_new_data()
        if data_check.eligible:
            return data_check

        return RetrainingEligibility(
            eligible=False,
            trigger=None,
            reason="No retraining triggers met",
            details={}
        )

    def _check_cooldown(self) -> bool:
        """Check if enough time has passed since last retraining."""
        last_job = self.db.query(RetrainingJob).filter(
            RetrainingJob.status.in_([RetrainingStatus.COMPLETED.value, RetrainingStatus.RUNNING.value])
        ).order_by(RetrainingJob.timestamp.desc()).first()

        if not last_job:
            return True

        last_ts = last_job.timestamp
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        elapsed = datetime.now(timezone.utc) - last_ts
        return elapsed >= timedelta(hours=self.config.retraining_cooldown_hours)

    def _check_sustained_critical_drift(self) -> RetrainingEligibility:
        """Check for sustained critical drift alerts."""
        since = datetime.now(timezone.utc) - timedelta(hours=self.config.drift_window_hours)

        critical_alerts = self.db.query(Alert).filter(
            Alert.timestamp >= since,
            Alert.severity == "critical",
            Alert.alert_type == "drift",
            Alert.status.in_(["active", "acknowledged"])
        ).count()

        if critical_alerts >= self.config.critical_drift_count_threshold:
            # Get affected features
            alerts = self.db.query(Alert).filter(
                Alert.timestamp >= since,
                Alert.severity == "critical",
                Alert.alert_type == "drift",
            ).all()

            features = list(set(a.feature_name for a in alerts if a.feature_name))

            return RetrainingEligibility(
                eligible=True,
                trigger=RetrainingTrigger.SUSTAINED_CRITICAL_DRIFT,
                reason=f"Sustained critical drift detected: {critical_alerts} critical drift alerts in {self.config.drift_window_hours}h",
                details={
                    "critical_alert_count": critical_alerts,
                    "window_hours": self.config.drift_window_hours,
                    "affected_features": features,
                    "threshold": self.config.critical_drift_count_threshold,
                }
            )

        return RetrainingEligibility(eligible=False, trigger=None, reason="")

    def _check_performance_degradation(self) -> RetrainingEligibility:
        """Check for actual model performance degradation (requires labeled data)."""
        # This requires ground truth labels which may not be available
        # For now, return not eligible - would need label feedback loop
        return RetrainingEligibility(
            eligible=False,
            trigger=None,
            reason="Performance degradation check requires labeled feedback data (not implemented)",
            details={}
        )

    def _check_sufficient_new_data(self) -> RetrainingEligibility:
        """Check if enough new labeled data is available for retraining."""
        since = datetime.now(timezone.utc) - timedelta(days=self.config.data_window_days)

        # Count prediction history entries (proxy for new data)
        new_data_count = self.db.query(func.count(PredictionHistory.id)).filter(
            PredictionHistory.timestamp >= since
        ).scalar() or 0

        if new_data_count >= self.config.min_new_labeled_samples:
            return RetrainingEligibility(
                eligible=True,
                trigger=RetrainingTrigger.SUFFICIENT_NEW_DATA,
                reason=f"Sufficient new data accumulated: {new_data_count} samples in {self.config.data_window_days} days",
                details={
                    "new_samples": new_data_count,
                    "window_days": self.config.data_window_days,
                    "threshold": self.config.min_new_labeled_samples,
                }
            )

        return RetrainingEligibility(
            eligible=False,
            trigger=None,
            reason="",
            details={"new_samples": new_data_count, "threshold": self.config.min_new_labeled_samples}
        )

    def execute_retraining(
        self,
        job_id: str,
        trigger_batch_id: str | None = None,
        trigger_reason: str = "automated",
    ) -> dict[str, Any]:
        """Execute the retraining pipeline.

        Args:
            job_id: Retraining job ID
            trigger_batch_id: Batch that triggered retraining
            trigger_reason: Reason for retraining

        Returns:
            Dictionary with execution results
        """
        # Update job status
        job = self.db.query(RetrainingJob).filter(RetrainingJob.job_id == job_id).first()
        if not job:
            return {"success": False, "error": "Job not found"}

        job.status = RetrainingStatus.RUNNING.value
        self.db.commit()

        try:
            # Step 1: Prepare training data
            logger.info(f"Job {job_id}: Preparing training data...")
            training_data_path = self._prepare_training_data(job_id)

            # Step 2: Run training
            logger.info(f"Job {job_id}: Starting model training...")
            run_id, model_version = self._run_training(job_id, training_data_path)

            # Step 3: Get candidate metrics
            candidate_metrics = self._get_candidate_metrics(run_id)
            job.candidate_metrics = candidate_metrics
            job.candidate_model_version = model_version
            job.mlflow_run_id = run_id

            # Step 4: Get champion metrics
            champion_info = self.mlflow_manager.get_latest_model(
                self.config.model_name, "Production"
            )
            if champion_info:
                champion_run = self.mlflow_manager.client.get_run(champion_info["run_id"])
                job.champion_model_version = champion_info.get("version_string")
                job.champion_metrics = champion_run.data.metrics

            self.db.commit()

            # Step 5: Validate candidate
            logger.info(f"Job {job_id}: Validating candidate model...")
            validation_result = self._validate_candidate(job)

            if validation_result["passed"]:
                # Step 6: Promote
                logger.info(f"Job {job_id}: Promoting candidate model...")
                promotion_result = self._promote_candidate(job)
                job.promotion_decision = "promoted" if promotion_result else "rejected"
                job.promotion_reason = "Candidate meets all promotion criteria"
            else:
                job.promotion_decision = "rejected"
                job.promotion_reason = validation_result["reason"]

            job.status = RetrainingStatus.COMPLETED.value
            job.completed_at = datetime.now(timezone.utc)
            self.db.commit()

            logger.info(f"Job {job_id}: Retraining completed. Promotion: {job.promotion_decision}")
            return {
                "success": True,
                "job_id": job_id,
                "mlflow_run_id": run_id,
                "candidate_version": model_version,
                "promotion_decision": job.promotion_decision,
                "promotion_reason": job.promotion_reason,
            }

        except Exception as e:
            logger.error(f"Job {job_id}: Retraining failed: {e}")
            job.status = RetrainingStatus.FAILED.value
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)
            self.db.commit()
            return {"success": False, "error": str(e)}

    def _prepare_training_data(self, job_id: str) -> str:
        """Prepare combined training dataset from historical + new data.

        Returns:
            Path to prepared training data CSV
        """
        # In a real implementation, this would:
        # 1. Load historical training data
        # 2. Add new labeled data from prediction history (if labels available)
        # 3. Save combined dataset

        # For now, return the original training data path
        return "data/raw/Telco-Customer-Churn.csv"

    def _run_training(self, job_id: str, data_path: str) -> tuple[str, str]:
        """Run the training script and return MLflow run ID and model version.

        Returns:
            Tuple of (mlflow_run_id, model_version)
        """
        model_version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # Run training script as subprocess
        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = self.mlflow_manager.tracking_uri or "./mlruns"
        env["MLFLOW_EXPERIMENT"] = self.config.mlflow_experiment

        cmd = [
            "python", self.config.training_script,
            "--model-version", model_version,
            "--data-path", data_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if result.returncode != 0:
            raise RuntimeError(f"Training failed: {result.stderr}")

        # Parse output for MLflow run ID (in practice, would use MLflow API directly)
        # For now, find the latest run
        runs = self.mlflow_manager.client.search_runs(
            experiment_ids=[self.mlflow_manager.experiment_id],
            filter_string=f"tags.model_version = '{model_version}'",
            max_results=1,
            order_by=["start_time DESC"]
        )

        if runs:
            return runs[0].info.run_id, model_version
        else:
            raise RuntimeError("Training completed but no MLflow run found")

    def _get_candidate_metrics(self, run_id: str) -> dict[str, float]:
        """Get metrics from candidate model run."""
        run = self.mlflow_manager.client.get_run(run_id)
        return run.data.metrics

    def _validate_candidate(self, job: RetrainingJob) -> dict[str, Any]:
        """Validate candidate model against promotion criteria."""
        candidate_metrics = job.candidate_metrics or {}

        # Check minimum thresholds
        failed_checks = []

        accuracy = candidate_metrics.get("accuracy", 0)
        if accuracy < self.config.min_accuracy:
            failed_checks.append(f"accuracy {accuracy:.4f} < {self.config.min_accuracy}")

        roc_auc = candidate_metrics.get("roc_auc", 0)
        if roc_auc < self.config.min_roc_auc:
            failed_checks.append(f"roc_auc {roc_auc:.4f} < {self.config.min_roc_auc}")

        # Compare with champion if available
        if job.champion_metrics:
            comparison = self.mlflow_manager.compare_models(
                candidate_run_id=job.mlflow_run_id,
                champion_model_name=self.config.model_name,
                champion_stage="Production",
                metric_thresholds={
                    "accuracy": self.config.max_metric_regression,
                    "roc_auc": self.config.max_metric_regression,
                    "f1": self.config.max_metric_regression,
                }
            )
            if not comparison.passed:
                failed_checks.append(f"champion_comparison: {comparison.reason}")

        passed = len(failed_checks) == 0
        reason = "All validation checks passed" if passed else f"Failed checks: {', '.join(failed_checks)}"

        return {"passed": passed, "reason": reason, "failed_checks": failed_checks}

    def _promote_candidate(self, job: RetrainingJob) -> bool:
        """Promote candidate model to production."""
        if not job.candidate_model_version:
            return False

        # Get MLflow version number for candidate
        mlflow_version = self.mlflow_manager.get_model_version(
            self.config.model_name, job.candidate_model_version
        )

        if not mlflow_version:
            logger.error(f"Could not find MLflow version for {job.candidate_model_version}")
            return False

        # Promote to production
        success = self.mlflow_manager.promote_model(
            model_name=self.config.model_name,
            version=mlflow_version,
            stage="Production",
            archive_existing=True,
        )

        if success:
            # Update model metadata in our database
            model_meta = self.db.query(ModelMetadata).filter(
                ModelMetadata.version == job.candidate_model_version
            ).first()
            if model_meta:
                model_meta.status = "production"
                self.db.commit()

        return success


def create_retraining_orchestrator(
    db: Session,
    config: RetrainingConfig | None = None,
) -> RetrainingOrchestrator:
    """Factory function to create RetrainingOrchestrator instance."""
    return RetrainingOrchestrator(db=db, config=config)
