"""MLflow integration for experiment tracking and model management."""

from __future__ import annotations

import logging
import os

# Enable local file store in MLflow 2.8+ without warnings/exceptions
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


@dataclass
class ModelComparisonResult:
    """Result of champion vs candidate model comparison."""
    candidate_version: str
    champion_version: str
    candidate_metrics: dict[str, float]
    champion_metrics: dict[str, float]
    metric_differences: dict[str, float]
    passed: bool
    failed_metrics: list[str]
    reason: str


class MLflowManager:
    """Manages MLflow integration for experiment tracking and model registry."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "drift_monitoring",
        registry_uri: str | None = None,
    ):
        """Initialize MLflow manager.

        Args:
            tracking_uri: MLflow tracking server URI (defaults to local ./mlruns)
            experiment_name: Name of the MLflow experiment
            registry_uri: MLflow model registry URI (for remote registry)
        """
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
        self.experiment_name = experiment_name
        self.registry_uri = registry_uri or os.getenv("MLFLOW_REGISTRY_URI")

        # Set tracking URI
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created MLflow experiment: {experiment_name} (ID: {self.experiment_id})")
            else:
                self.experiment_id = self.experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {self.experiment_id})")
            self.client = MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.registry_uri)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Could not fully initialize MLflow tracking server: {e}")
            self.experiment = None
            self.experiment_id = "0"
            try:
                self.client = MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.registry_uri)
            except Exception:
                self.client = None

    @contextmanager
    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None):
        """Context manager for MLflow run.

        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run

        Yields:
            Active MLflow run
        """
        run = mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id)
        try:
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            yield run
        finally:
            mlflow.end_run()

    def log_training_run(
        self,
        model: Any,
        model_name: str,
        version: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifacts: dict[str, str],
        tags: dict[str, str] | None = None,
        run_name: str | None = None,
    ) -> str:
        """Log a complete training run to MLflow.

        Args:
            model: Trained model object
            model_name: Name of the model
            version: Model version string
            params: Training parameters
            metrics: Evaluation metrics
            artifacts: Dictionary of artifact names to file paths
            tags: Optional tags
            run_name: Optional run name

        Returns:
            MLflow run ID
        """
        run_name = run_name or f"{model_name}_{version}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        default_tags = {
            "model_name": model_name,
            "model_version": version,
            "training_date": datetime.now(timezone.utc).isoformat(),
            "framework": "xgboost" if "xgb" in type(model).__name__.lower() else "sklearn",
        }
        if tags:
            default_tags.update(tags)

        with self.start_run(run_name=run_name, tags=default_tags) as run:
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            # Log model
            if "xgb" in type(model).__name__.lower():
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")

            # Log additional artifacts
            for artifact_name, artifact_path in artifacts.items():
                if os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path, artifact_name)

            run_id = run.info.run_id
            logger.info(f"Logged training run: {run_id} for {model_name} v{version}")
            return run_id

    def register_model(
        self,
        run_id: str,
        model_name: str,
        version: str,
        stage: str = "Staging",
    ) -> str:
        """Register a model in MLflow Model Registry.

        Args:
            run_id: MLflow run ID containing the model
            model_name: Name for the registered model
            version: Version string (used as alias/tag)
            stage: Initial stage (Staging, Production, Archived)

        Returns:
            Registered model version
        """
        model_uri = f"runs:/{run_id}/model"

        # Register model
        mv = mlflow.register_model(model_uri, model_name)

        # Add version tag
        self.client.set_model_version_tag(
            name=model_name,
            version=mv.version,
            key="version_string",
            value=version,
        )

        # Transition to stage
        self.client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage=stage,
            archive_existing_versions=False,
        )

        logger.info(f"Registered model {model_name} version {mv.version} in stage {stage}")
        return mv.version

    def get_model_version(self, model_name: str, version: str) -> int | None:
        """Get MLflow model version number for a version string.

        Args:
            model_name: Registered model name
            version: Version string (e.g., "v1.0.0")

        Returns:
            MLflow version number or None if not found
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            for mv in versions:
                if mv.tags.get("version_string") == version:
                    return int(mv.version)
        except Exception as e:
            logger.warning(f"Error searching model versions: {e}")
        return None

    def get_latest_model(self, model_name: str, stage: str = "Production") -> dict | None:
        """Get latest model version in a stage.

        Args:
            model_name: Registered model name
            stage: Stage to search (Production, Staging, etc.)

        Returns:
            Dictionary with model info or None
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if versions:
                mv = versions[0]
                return {
                    "version": mv.version,
                    "version_string": mv.tags.get("version_string"),
                    "run_id": mv.run_id,
                    "stage": mv.current_stage,
                    "creation_timestamp": mv.creation_timestamp,
                }
        except Exception as e:
            logger.warning(f"Error getting latest model: {e}")
        return None

    def compare_models(
        self,
        candidate_run_id: str,
        champion_model_name: str,
        champion_stage: str = "Production",
        metric_thresholds: dict[str, float] | None = None,
    ) -> ModelComparisonResult:
        """Compare candidate model against champion.

        Args:
            candidate_run_id: MLflow run ID of candidate model
            champion_model_name: Name of registered champion model
            champion_stage: Stage of champion model to compare against
            metric_thresholds: Minimum required metric values (metric -> min_value)

        Returns:
            ModelComparisonResult with comparison details
        """
        # Get candidate metrics
        candidate_run = self.client.get_run(candidate_run_id)
        candidate_metrics = candidate_run.data.metrics

        # Get champion model
        champion_info = self.get_latest_model(champion_model_name, champion_stage)
        if not champion_info:
            return ModelComparisonResult(
                candidate_version="unknown",
                champion_version="none",
                candidate_metrics=candidate_metrics,
                champion_metrics={},
                metric_differences={},
                passed=False,
                failed_metrics=["champion_not_found"],
                reason=f"No champion model found in {champion_stage} stage",
            )

        # Get champion metrics
        champion_run = self.client.get_run(champion_info["run_id"])
        champion_metrics = champion_run.data.metrics

        # Get version strings
        candidate_version = candidate_run.data.tags.get("model_version", "unknown")
        champion_version = champion_info.get("version_string", "unknown")

        # Compare metrics
        metric_differences = {}
        failed_metrics = []

        for metric_name, candidate_value in candidate_metrics.items():
            if metric_name in champion_metrics:
                champion_value = champion_metrics[metric_name]
                diff = candidate_value - champion_value
                metric_differences[metric_name] = diff

                # Check if candidate is worse (for metrics where higher is better)
                higher_is_better = metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]
                if higher_is_better and diff < 0:
                    # Allow small regression within threshold
                    threshold = metric_thresholds.get(metric_name, 0.0) if metric_thresholds else 0.0
                    if diff < -threshold:
                        failed_metrics.append(metric_name)
                elif not higher_is_better and diff > 0:
                    # For metrics where lower is better (e.g., loss)
                    threshold = metric_thresholds.get(metric_name, 0.0) if metric_thresholds else 0.0
                    if diff > threshold:
                        failed_metrics.append(metric_name)

        passed = len(failed_metrics) == 0

        reason = "All metrics meet or exceed champion" if passed else f"Failed metrics: {', '.join(failed_metrics)}"

        return ModelComparisonResult(
            candidate_version=candidate_version,
            champion_version=champion_version,
            candidate_metrics=candidate_metrics,
            champion_metrics=champion_metrics,
            metric_differences=metric_differences,
            passed=passed,
            failed_metrics=failed_metrics,
            reason=reason,
        )

    def promote_model(
        self,
        model_name: str,
        version: int,
        stage: str = "Production",
        archive_existing: bool = True,
    ) -> bool:
        """Promote a model version to a stage.

        Args:
            model_name: Registered model name
            version: Model version number
            stage: Target stage (Production, Staging, Archived)
            archive_existing: Whether to archive existing versions in target stage

        Returns:
            True if successful
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=str(version),
                stage=stage,
                archive_existing_versions=archive_existing,
            )
            logger.info(f"Promoted {model_name} v{version} to {stage}")
            return True
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False

    def load_model(self, model_name: str, stage: str = "Production") -> Any:
        """Load a model from MLflow registry.

        Args:
            model_name: Registered model name
            stage: Stage to load from

        Returns:
            Loaded model object
        """
        model_uri = f"models:/{model_name}/{stage}"
        return mlflow.pyfunc.load_model(model_uri)

    def get_run_artifacts(self, run_id: str) -> list[str]:
        """List artifacts for a run.

        Args:
            run_id: MLflow run ID

        Returns:
            List of artifact paths
        """
        try:
            artifacts = self.client.list_artifacts(run_id)
            return [a.path for a in artifacts]
        except Exception as e:
            logger.warning(f"Error listing artifacts: {e}")
            return []


def create_mlflow_manager(
    tracking_uri: str | None = None,
    experiment_name: str = "drift_monitoring",
) -> MLflowManager:
    """Factory function to create MLflowManager instance."""
    return MLflowManager(tracking_uri=tracking_uri, experiment_name=experiment_name)
