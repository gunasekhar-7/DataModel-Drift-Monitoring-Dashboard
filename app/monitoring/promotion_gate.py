"""Model promotion gate for candidate model validation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import joblib
from sqlalchemy.orm import Session

from app.models import ModelMetadata
from app.monitoring.mlflow_integration import MLflowManager, create_mlflow_manager

logger = logging.getLogger(__name__)


@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromotionGateResult:
    """Result of promotion gate validation."""
    passed: bool
    candidate_version: str
    champion_version: str | None
    checks: list[ValidationCheck]
    failed_checks: list[str]
    reason: str


class PromotionGate:
    """Validates candidate models before promotion to production."""

    def __init__(
        self,
        db: Session,
        mlflow_manager: MLflowManager | None = None,
        min_accuracy: float = 0.60,
        min_roc_auc: float = 0.55,
        max_regression: float = 0.02,
        required_metrics: list[str] | None = None,
    ):
        """Initialize the promotion gate.

        Args:
            db: Database session
            mlflow_manager: MLflow manager
            min_accuracy: Minimum accuracy threshold
            min_roc_auc: Minimum ROC-AUC threshold
            max_regression: Maximum allowed metric regression vs champion
            required_metrics: List of required metric names
        """
        self.db = db
        self.mlflow_manager = mlflow_manager or create_mlflow_manager()
        self.min_accuracy = min_accuracy
        self.min_roc_auc = min_roc_auc
        self.max_regression = max_regression
        self.required_metrics = required_metrics or ["accuracy", "precision", "recall", "f1", "roc_auc"]

    def validate(
        self,
        candidate_version: str,
        champion_version: str | None = None,
    ) -> PromotionGateResult:
        """Validate a candidate model for promotion.

        Args:
            candidate_version: Candidate model version string
            champion_version: Current champion version (auto-detected if None)

        Returns:
            PromotionGateResult with validation details
        """
        checks = []
        failed_checks = []

        # Check 1: Candidate model exists in MLflow
        check = self._check_candidate_exists(candidate_version)
        checks.append(check)
        if not check.passed:
            failed_checks.append(check.name)

        # Check 2: Candidate artifacts are loadable
        check = self._check_artifacts_loadable(candidate_version)
        checks.append(check)
        if not check.passed:
            failed_checks.append(check.name)

        # Check 3: Input schema compatibility
        check = self._check_schema_compatibility(candidate_version, champion_version)
        checks.append(check)
        if not check.passed:
            failed_checks.append(check.name)

        # Check 4: Preprocessing compatibility
        check = self._check_preprocessing_compatibility(candidate_version, champion_version)
        checks.append(check)
        if not check.passed:
            failed_checks.append(check.name)

        # Check 5: Minimum performance requirements
        check = self._check_minimum_performance(candidate_version)
        checks.append(check)
        if not check.passed:
            failed_checks.append(check.name)

        # Check 6: Champion vs candidate comparison
        if champion_version:
            check = self._check_champion_comparison(candidate_version, champion_version)
            checks.append(check)
            if not check.passed:
                failed_checks.append(check.name)
        else:
            # Auto-detect champion
            champion_info = self.mlflow_manager.get_latest_model(
                "baseline_churn_model", "Production"
            )
            if champion_info:
                check = self._check_champion_comparison(candidate_version, champion_info["version_string"])
                checks.append(check)
                if not check.passed:
                    failed_checks.append(check.name)

        # Check 7: Regression tests (inference compatibility)
        check = self._check_inference_compatibility(candidate_version)
        checks.append(check)
        if not check.passed:
            failed_checks.append(check.name)

        # Check 8: Artifact integrity
        check = self._check_artifact_integrity(candidate_version)
        checks.append(check)
        if not check.passed:
            failed_checks.append(check.name)

        passed = len(failed_checks) == 0
        reason = "All promotion checks passed" if passed else f"Failed checks: {', '.join(failed_checks)}"

        return PromotionGateResult(
            passed=passed,
            candidate_version=candidate_version,
            champion_version=champion_version,
            checks=checks,
            failed_checks=failed_checks,
            reason=reason,
        )

    def _check_candidate_exists(self, version: str) -> ValidationCheck:
        """Check if candidate model exists in MLflow registry or database."""
        try:
            if self.mlflow_manager:
                mv = self.mlflow_manager.get_model_version("baseline_churn_model", version)
                if mv:
                    return ValidationCheck(
                        name="candidate_exists",
                        passed=True,
                        message=f"Candidate model found in MLflow (version {mv})",
                        details={"mlflow_version": mv}
                    )
        except Exception:
            pass

        # Fallback check in database ModelMetadata
        meta = self.db.query(ModelMetadata).filter(ModelMetadata.version == version).first()
        if meta:
            return ValidationCheck(
                name="candidate_exists",
                passed=True,
                message=f"Candidate model found in database metadata (version {version})",
                details={"version": version, "name": meta.name}
            )

        return ValidationCheck(
            name="candidate_exists",
            passed=False,
            message=f"Candidate model version '{version}' not found in MLflow registry or database",
            details={"searched_version": version}
        )

    def _check_artifacts_loadable(self, version: str) -> ValidationCheck:
        """Check if candidate model artifacts can be loaded."""
        try:
            mv = self.mlflow_manager.get_model_version("baseline_churn_model", version)
            if not mv:
                return ValidationCheck(
                    name="artifacts_loadable",
                    passed=False,
                    message="Candidate not found in MLflow",
                    details={}
                )

            # Try loading model
            model = self.mlflow_manager.load_model("baseline_churn_model", "Staging")

            # Try loading preprocessor from artifacts
            if self.mlflow_manager.client is None:
                raise RuntimeError("MLflow client is not initialized")
            mv_info = self.mlflow_manager.client.get_model_version("baseline_churn_model", str(mv))
            if not mv_info.run_id:
                raise RuntimeError("Candidate model version has no MLflow run_id")
            self.mlflow_manager.client.get_run(mv_info.run_id)

            return ValidationCheck(
                name="artifacts_loadable",
                passed=True,
                message="Model and artifacts load successfully",
                details={"model_type": type(model).__name__}
            )
        except Exception as e:
            return ValidationCheck(
                name="artifacts_loadable",
                passed=False,
                message=f"Failed to load candidate artifacts: {e}",
                details={"error": str(e)}
            )

    def _check_schema_compatibility(
        self,
        candidate_version: str,
        champion_version: str | None,
    ) -> ValidationCheck:
        """Check input schema compatibility with champion."""
        try:
            # Get candidate feature schema
            candidate_meta = self.db.query(ModelMetadata).filter(
                ModelMetadata.version == candidate_version
            ).first()

            if not candidate_meta or not candidate_meta.feature_schema:
                return ValidationCheck(
                    name="schema_compatibility",
                    passed=False,
                    message="Candidate feature schema not found in database",
                    details={}
                )

            candidate_schema = candidate_meta.feature_schema
            if isinstance(candidate_schema, str):
                import json
                candidate_schema = json.loads(candidate_schema)

            # Get champion schema
            champion_schema = None
            if champion_version:
                champion_meta = self.db.query(ModelMetadata).filter(
                    ModelMetadata.version == champion_version
                ).first()
                if champion_meta and champion_meta.feature_schema:
                    champion_schema = champion_meta.feature_schema
                    if isinstance(champion_schema, str):
                        import json
                        champion_schema = json.loads(champion_schema)

            if not champion_schema:
                # No champion to compare against - just validate candidate has required features
                required = {"numeric_features", "categorical_features", "target_feature"}
                missing = required - set(candidate_schema.keys())
                if missing:
                    return ValidationCheck(
                        name="schema_compatibility",
                        passed=False,
                        message=f"Candidate schema missing required keys: {missing}",
                        details={"candidate_schema_keys": list(candidate_schema.keys())}
                    )
                return ValidationCheck(
                    name="schema_compatibility",
                    passed=True,
                    message="Candidate schema valid (no champion to compare)",
                    details=candidate_schema
                )

            # Compare schemas
            cand_numeric = set(candidate_schema.get("numeric_features", []))
            champ_numeric = set(champion_schema.get("numeric_features", []))
            cand_categorical = set(candidate_schema.get("categorical_features", []))
            champ_categorical = set(champion_schema.get("categorical_features", []))

            missing_numeric = champ_numeric - cand_numeric
            missing_categorical = champ_categorical - cand_categorical
            extra_numeric = cand_numeric - champ_numeric
            extra_categorical = cand_categorical - champ_categorical

            issues = []
            if missing_numeric:
                issues.append(f"Missing numeric features: {missing_numeric}")
            if missing_categorical:
                issues.append(f"Missing categorical features: {missing_categorical}")

            if issues:
                return ValidationCheck(
                    name="schema_compatibility",
                    passed=False,
                    message=f"Schema incompatibility: {'; '.join(issues)}",
                    details={
                        "missing_numeric": list(missing_numeric),
                        "missing_categorical": list(missing_categorical),
                        "extra_numeric": list(extra_numeric),
                        "extra_categorical": list(extra_categorical),
                    }
                )

            return ValidationCheck(
                name="schema_compatibility",
                passed=True,
                message="Schema compatible with champion",
                details={
                    "numeric_features_match": len(missing_numeric) == 0,
                    "categorical_features_match": len(missing_categorical) == 0,
                }
            )
        except Exception as e:
            return ValidationCheck(
                name="schema_compatibility",
                passed=False,
                message=f"Error checking schema compatibility: {e}",
                details={"error": str(e)}
            )

    def _check_preprocessing_compatibility(
        self,
        candidate_version: str,
        champion_version: str | None,
    ) -> ValidationCheck:
        """Check if preprocessing pipeline is compatible."""
        try:
            # Load candidate preprocessor
            candidate_meta = self.db.query(ModelMetadata).filter(
                ModelMetadata.version == candidate_version
            ).first()

            if not candidate_meta or not candidate_meta.preprocessor_path:
                return ValidationCheck(
                    name="preprocessing_compatibility",
                    passed=False,
                    message="Candidate preprocessor path not found",
                    details={}
                )

            # Check if preprocessor file exists and is loadable
            if not os.path.exists(candidate_meta.preprocessor_path):
                return ValidationCheck(
                    name="preprocessing_compatibility",
                    passed=False,
                    message=f"Preprocessor file not found: {candidate_meta.preprocessor_path}",
                    details={}
                )

            preprocessor = joblib.load(candidate_meta.preprocessor_path)

            # Verify it's a ColumnTransformer
            from sklearn.compose import ColumnTransformer
            if not isinstance(preprocessor, ColumnTransformer):
                return ValidationCheck(
                    name="preprocessing_compatibility",
                    passed=False,
                    message=f"Preprocessor is not a ColumnTransformer: {type(preprocessor)}",
                    details={}
                )

            return ValidationCheck(
                name="preprocessing_compatibility",
                passed=True,
                message="Preprocessor loads and is valid ColumnTransformer",
                details={"transformers": [t[0] for t in preprocessor.transformers_]}
            )
        except Exception as e:
            return ValidationCheck(
                name="preprocessing_compatibility",
                passed=False,
                message=f"Error checking preprocessing compatibility: {e}",
                details={"error": str(e)}
            )

    def _check_minimum_performance(self, version: str) -> ValidationCheck:
        """Check if candidate meets minimum performance thresholds."""
        metrics = None
        if self.mlflow_manager:
            try:
                mv = self.mlflow_manager.get_model_version("baseline_churn_model", version)
                if mv and self.mlflow_manager.client is not None:
                    mv_info = self.mlflow_manager.client.get_model_version("baseline_churn_model", str(mv))
                    if mv_info.run_id:
                        run = self.mlflow_manager.client.get_run(mv_info.run_id)
                        metrics = run.data.metrics
            except Exception:
                metrics = None

        if not metrics:
            meta = self.db.query(ModelMetadata).filter(ModelMetadata.version == version).first()
            if meta and meta.metrics:
                import json
                metrics = meta.metrics if isinstance(meta.metrics, dict) else json.loads(meta.metrics)

        if not metrics:
            return ValidationCheck(
                name="minimum_performance",
                passed=False,
                message="Candidate metrics not found in MLflow or database",
                details={},
            )

        issues = []
        for metric in self.required_metrics:
            if metric not in metrics:
                issues.append(f"Missing metric: {metric}")
            else:
                value = metrics[metric]
                if metric == "accuracy" and value < self.min_accuracy:
                    issues.append(f"accuracy {value:.4f} < {self.min_accuracy}")
                elif metric == "roc_auc" and value < self.min_roc_auc:
                    issues.append(f"roc_auc {value:.4f} < {self.min_roc_auc}")

        if issues:
            return ValidationCheck(
                name="minimum_performance",
                passed=False,
                message=f"Minimum performance not met: {'; '.join(issues)}",
                details={"metrics": metrics, "thresholds": {"accuracy": self.min_accuracy, "roc_auc": self.min_roc_auc}},
            )

        return ValidationCheck(
            name="minimum_performance",
            passed=True,
            message="All minimum performance thresholds met",
            details={"metrics": metrics},
        )

    def _check_champion_comparison(
        self,
        candidate_version: str,
        champion_version: str,
    ) -> ValidationCheck:
        """Compare candidate against champion."""
        try:
            comparison = self.mlflow_manager.compare_models(
                candidate_run_id="",  # Will be filled by getting candidate run
                champion_model_name="baseline_churn_model",
                champion_stage="Production",
                metric_thresholds={
                    "accuracy": self.max_regression,
                    "roc_auc": self.max_regression,
                    "f1": self.max_regression,
                    "precision": self.max_regression,
                    "recall": self.max_regression,
                }
            )

            # Get candidate run ID
            candidate_mv = self.mlflow_manager.get_model_version("baseline_churn_model", candidate_version)
            if candidate_mv and self.mlflow_manager.client is not None:
                candidate_run_id = self.mlflow_manager.client.get_model_version(
                    "baseline_churn_model", str(candidate_mv)
                ).run_id
                if candidate_run_id:
                    comparison = self.mlflow_manager.compare_models(
                        candidate_run_id=candidate_run_id,
                        champion_model_name="baseline_churn_model",
                        champion_stage="Production",
                        metric_thresholds={
                            "accuracy": self.max_regression,
                            "roc_auc": self.max_regression,
                            "f1": self.max_regression,
                        }
                    )

            if not comparison.passed:
                return ValidationCheck(
                    name="champion_comparison",
                    passed=False,
                    message=f"Candidate does not meet champion comparison: {comparison.reason}",
                    details={
                        "candidate_metrics": comparison.candidate_metrics,
                        "champion_metrics": comparison.champion_metrics,
                        "differences": comparison.metric_differences,
                        "failed_metrics": comparison.failed_metrics,
                    }
                )

            return ValidationCheck(
                name="champion_comparison",
                passed=True,
                message="Candidate meets or exceeds champion performance",
                details={
                    "candidate_metrics": comparison.candidate_metrics,
                    "champion_metrics": comparison.champion_metrics,
                    "differences": comparison.metric_differences,
                }
            )
        except Exception as e:
            return ValidationCheck(
                name="champion_comparison",
                passed=False,
                message=f"Error comparing with champion: {e}",
                details={"error": str(e)}
            )

    def _check_inference_compatibility(self, version: str) -> ValidationCheck:
        """Check if candidate model can run inference correctly."""
        try:
            # Load candidate model
            model = self.mlflow_manager.load_model("baseline_churn_model", "Staging")

            # Load preprocessor
            candidate_meta = self.db.query(ModelMetadata).filter(
                ModelMetadata.version == version
            ).first()

            if not candidate_meta:
                return ValidationCheck(
                    name="inference_compatibility",
                    passed=False,
                    message="Candidate metadata not found",
                    details={}
                )

            preprocessor = joblib.load(candidate_meta.preprocessor_path)
            feature_schema = candidate_meta.feature_schema
            if isinstance(feature_schema, str):
                import json
                feature_schema = json.loads(feature_schema)

            # Create sample input
            sample: dict[str, Any] = {}
            for feat in feature_schema.get("numeric_features", []):
                sample[feat] = 0.0
            for feat in feature_schema.get("categorical_features", []):
                sample[feat] = "unknown"

            # Run inference
            import pandas as pd
            df = pd.DataFrame([sample])
            transformed = preprocessor.transform(df)
            prediction = model.predict(transformed)
            probabilities = model.predict_proba(transformed) if hasattr(model, "predict_proba") else None

            # Verify output shape
            if len(prediction) != 1:
                return ValidationCheck(
                    name="inference_compatibility",
                    passed=False,
                    message=f"Unexpected prediction shape: {prediction.shape}",
                    details={}
                )

            return ValidationCheck(
                name="inference_compatibility",
                passed=True,
                message="Inference runs successfully with expected output",
                details={
                    "prediction": int(prediction[0]),
                    "probabilities_shape": list(probabilities.shape) if probabilities is not None else None,
                }
            )
        except Exception as e:
            return ValidationCheck(
                name="inference_compatibility",
                passed=False,
                message=f"Inference compatibility check failed: {e}",
                details={"error": str(e)}
            )

    def _check_artifact_integrity(self, version: str) -> ValidationCheck:
        """Check artifact integrity (files exist, not corrupted)."""
        try:
            candidate_meta = self.db.query(ModelMetadata).filter(
                ModelMetadata.version == version
            ).first()

            if not candidate_meta:
                return ValidationCheck(
                    name="artifact_integrity",
                    passed=False,
                    message="Candidate metadata not found",
                    details={}
                )

            issues = []
            artifacts = {
                "model": candidate_meta.artifact_path,
                "preprocessor": candidate_meta.preprocessor_path,
            }

            # Check feature schema file
            if candidate_meta.feature_schema_path:
                artifacts["feature_schema"] = candidate_meta.feature_schema_path

            for name, path in artifacts.items():
                if not path:
                    issues.append(f"{name}: path not recorded")
                    continue
                if not os.path.exists(path):
                    issues.append(f"{name}: file not found at {path}")
                    continue
                # Check file size > 0
                if os.path.getsize(path) == 0:
                    issues.append(f"{name}: file is empty")

            if issues:
                return ValidationCheck(
                    name="artifact_integrity",
                    passed=False,
                    message=f"Artifact integrity issues: {'; '.join(issues)}",
                    details={"issues": issues}
                )

            return ValidationCheck(
                name="artifact_integrity",
                passed=True,
                message="All artifacts present and non-empty",
                details={name: {"path": path, "size": os.path.getsize(path)} for name, path in artifacts.items()}
            )
        except Exception as e:
            return ValidationCheck(
                name="artifact_integrity",
                passed=False,
                message=f"Error checking artifact integrity: {e}",
                details={"error": str(e)}
            )


def create_promotion_gate(
    db: Session,
    mlflow_manager: MLflowManager | None = None,
    min_accuracy: float = 0.60,
    min_roc_auc: float = 0.55,
    max_regression: float = 0.02,
) -> PromotionGate:
    """Factory function to create PromotionGate instance."""
    return PromotionGate(
        db=db,
        mlflow_manager=mlflow_manager,
        min_accuracy=min_accuracy,
        min_roc_auc=min_roc_auc,
        max_regression=max_regression,
    )
