"""Data Quality Engine for drift monitoring platform.

Provides reusable validators for data quality checks with structured results
including severity levels and feature-level explanations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class Severity(Enum):
    """Severity levels for data quality issues."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ValidationType(Enum):
    """Types of data quality validations."""

    MISSING_VALUES = "missing_values"
    DUPLICATES = "duplicates"
    DATA_TYPES = "data_types"
    SCHEMA_COMPATIBILITY = "schema_compatibility"
    OUT_OF_RANGE = "out_of_range"
    UNEXPECTED_CATEGORICAL = "unexpected_categorical"
    INFINITE_VALUES = "infinite_values"
    EMPTY_BATCH = "empty_batch"
    ALL_NULL_COLUMNS = "all_null_columns"
    INVALID_RECORDS = "invalid_records"


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    validation_type: ValidationType
    column: str | None
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    affected_rows: int = 0
    affected_percentage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "validation_type": self.validation_type.value,
            "column": self.column,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "affected_rows": self.affected_rows,
            "affected_percentage": round(self.affected_percentage, 4),
        }


@dataclass
class DataQualityReport:
    """Aggregated data quality report for a batch."""

    batch_id: str
    timestamp: str
    total_rows: int
    total_columns: int
    results: list[ValidationResult] = field(default_factory=list)
    passed: bool = True

    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result and update overall pass status."""
        self.results.append(result)
        if result.severity in (Severity.WARNING, Severity.CRITICAL):
            self.passed = False

    def get_results_by_type(
        self, validation_type: ValidationType
    ) -> list[ValidationResult]:
        """Filter results by validation type."""
        return [r for r in self.results if r.validation_type == validation_type]

    def get_results_by_severity(self, severity: Severity) -> list[ValidationResult]:
        """Filter results by severity."""
        return [r for r in self.results if r.severity == severity]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "batch_id": self.batch_id,
            "timestamp": self.timestamp,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total_checks": len(self.results),
                "critical_count": len(self.get_results_by_severity(Severity.CRITICAL)),
                "warning_count": len(self.get_results_by_severity(Severity.WARNING)),
                "info_count": len(self.get_results_by_severity(Severity.INFO)),
            },
        }


class DataQualityEngine:
    """Main engine for running data quality validations."""

    def __init__(
        self,
        reference_schema: dict[str, Any] | None = None,
        reference_stats: dict[str, Any] | None = None,
        categorical_domains: dict[str, list] | None = None,
        numeric_ranges: dict[str, tuple[float, float]] | None = None,
        missing_threshold: float = 0.1,
        duplicate_threshold: float = 0.01,
        out_of_range_threshold: float = 0.05,
        unexpected_categorical_threshold: float = 0.01,
    ):
        """Initialize the data quality engine.

        Args:
            reference_schema: Expected schema with column names and types
            reference_stats: Reference statistics for range checks
            categorical_domains: Expected categories per categorical column
            numeric_ranges: Expected (min, max) per numeric column
            missing_threshold: Threshold for missing value warnings (fraction)
            duplicate_threshold: Threshold for duplicate row warnings (fraction)
            out_of_range_threshold: Threshold for out-of-range warnings (fraction)
            unexpected_categorical_threshold: Threshold for unexpected categories (fraction)
        """
        self.reference_schema = reference_schema or {}
        self.reference_stats = reference_stats or {}
        self.categorical_domains = categorical_domains or {}
        self.numeric_ranges = numeric_ranges or {}
        self.missing_threshold = missing_threshold
        self.duplicate_threshold = duplicate_threshold
        self.out_of_range_threshold = out_of_range_threshold
        self.unexpected_categorical_threshold = unexpected_categorical_threshold

    def validate(self, df: pd.DataFrame, batch_id: str) -> DataQualityReport:
        """Run all validation checks on a DataFrame.

        Args:
            df: Input DataFrame to validate
            batch_id: Identifier for the batch

        Returns:
            DataQualityReport with all validation results
        """
        from datetime import datetime, timezone

        report = DataQualityReport(
            batch_id=batch_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_rows=len(df),
            total_columns=len(df.columns),
        )

        if len(df) == 0:
            report.add_result(
                ValidationResult(
                    validation_type=ValidationType.EMPTY_BATCH,
                    column=None,
                    severity=Severity.CRITICAL,
                    message="Batch contains zero rows",
                    details={"row_count": 0},
                    affected_rows=0,
                    affected_percentage=100.0,
                )
            )
            return report

        self._check_empty_batch(df, report)
        self._check_missing_values(df, report)
        self._check_duplicates(df, report)
        self._check_data_types(df, report)
        self._check_schema_compatibility(df, report)
        self._check_infinite_values(df, report)
        self._check_all_null_columns(df, report)
        self._check_out_of_range(df, report)
        self._check_unexpected_categorical(df, report)
        self._check_invalid_records(df, report)

        return report

    def _check_empty_batch(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check if batch is empty."""
        if len(df) == 0:
            report.add_result(
                ValidationResult(
                    validation_type=ValidationType.EMPTY_BATCH,
                    column=None,
                    severity=Severity.CRITICAL,
                    message="Batch contains zero rows",
                    details={"row_count": 0},
                    affected_rows=0,
                    affected_percentage=100.0,
                )
            )

    def _check_missing_values(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check for missing values in each column."""
        total_rows = len(df)
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / total_rows) * 100 if total_rows > 0 else 0

            if missing_count > 0:
                severity = (
                    Severity.CRITICAL
                    if missing_pct > self.missing_threshold * 100
                    else Severity.WARNING
                )
                report.add_result(
                    ValidationResult(
                        validation_type=ValidationType.MISSING_VALUES,
                        column=col,
                        severity=severity,
                        message=f"Column '{col}' has {missing_count} missing values ({missing_pct:.2f}%)",
                        details={
                            "missing_count": int(missing_count),
                            "missing_percentage": round(missing_pct, 4),
                            "threshold_percentage": self.missing_threshold * 100,
                        },
                        affected_rows=int(missing_count),
                        affected_percentage=missing_pct,
                    )
                )

    def _check_duplicates(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check for duplicate rows."""
        total_rows = len(df)
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0

        if duplicate_count > 0:
            severity = (
                Severity.CRITICAL
                if duplicate_pct > self.duplicate_threshold * 100
                else Severity.WARNING
            )
            report.add_result(
                ValidationResult(
                    validation_type=ValidationType.DUPLICATES,
                    column=None,
                    severity=severity,
                    message=f"Found {duplicate_count} duplicate rows ({duplicate_pct:.2f}%)",
                    details={
                        "duplicate_count": int(duplicate_count),
                        "duplicate_percentage": round(duplicate_pct, 4),
                        "threshold_percentage": self.duplicate_threshold * 100,
                    },
                    affected_rows=int(duplicate_count),
                    affected_percentage=duplicate_pct,
                )
            )

    def _check_data_types(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check data types against reference schema."""
        if not self.reference_schema:
            return

        for col, expected_type in self.reference_schema.items():
            if col not in df.columns:
                continue

            actual_type = str(df[col].dtype)
            type_match = self._types_compatible(actual_type, expected_type)

            if not type_match:
                report.add_result(
                    ValidationResult(
                        validation_type=ValidationType.DATA_TYPES,
                        column=col,
                        severity=Severity.WARNING,
                        message=f"Column '{col}' has type '{actual_type}', expected '{expected_type}'",
                        details={
                            "actual_type": actual_type,
                            "expected_type": expected_type,
                        },
                        affected_rows=len(df),
                        affected_percentage=100.0,
                    )
                )

    def _types_compatible(self, actual: str, expected: str) -> bool:
        """Check if actual dtype is compatible with expected type."""
        numeric_types = {"int64", "int32", "float64", "float32", "int", "float"}
        object_types = {"object", "string"}
        bool_types = {"bool"}

        actual_lower = actual.lower()
        expected_lower = expected.lower()

        if expected_lower in numeric_types:
            return actual_lower in numeric_types
        if expected_lower in object_types:
            return actual_lower in object_types
        if expected_lower in bool_types:
            return actual_lower in bool_types

        return actual_lower == expected_lower

    def _check_schema_compatibility(
        self, df: pd.DataFrame, report: DataQualityReport
    ) -> None:
        """Check schema compatibility with reference."""
        if not self.reference_schema:
            return

        expected_columns = set(self.reference_schema.keys())
        actual_columns = set(df.columns)

        missing_columns = expected_columns - actual_columns
        extra_columns = actual_columns - expected_columns

        if missing_columns:
            report.add_result(
                ValidationResult(
                    validation_type=ValidationType.SCHEMA_COMPATIBILITY,
                    column=None,
                    severity=Severity.CRITICAL,
                    message=f"Missing expected columns: {sorted(missing_columns)}",
                    details={
                        "missing_columns": sorted(missing_columns),
                        "expected_columns": sorted(expected_columns),
                        "actual_columns": sorted(actual_columns),
                    },
                    affected_rows=len(df),
                    affected_percentage=100.0,
                )
            )

        if extra_columns:
            report.add_result(
                ValidationResult(
                    validation_type=ValidationType.SCHEMA_COMPATIBILITY,
                    column=None,
                    severity=Severity.WARNING,
                    message=f"Unexpected extra columns: {sorted(extra_columns)}",
                    details={
                        "extra_columns": sorted(extra_columns),
                        "expected_columns": sorted(expected_columns),
                        "actual_columns": sorted(actual_columns),
                    },
                    affected_rows=len(df),
                    affected_percentage=100.0,
                )
            )

    def _check_infinite_values(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check for infinite values in numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            inf_pct = (inf_count / len(df)) * 100 if len(df) > 0 else 0

            if inf_count > 0:
                col_data = df[col]
                pos_inf = int(np.sum(np.isinf(col_data) & (col_data > 0)))
                neg_inf = int(np.sum(np.isinf(col_data) & (col_data < 0)))
                report.add_result(
                    ValidationResult(
                        validation_type=ValidationType.INFINITE_VALUES,
                        column=col,
                        severity=Severity.CRITICAL,
                        message=f"Column '{col}' contains {inf_count} infinite values ({inf_pct:.2f}%)",
                        details={
                            "infinite_count": int(inf_count),
                            "infinite_percentage": round(inf_pct, 4),
                            "positive_inf": pos_inf,
                            "negative_inf": neg_inf,
                        },
                        affected_rows=int(inf_count),
                        affected_percentage=inf_pct,
                    )
                )

    def _check_all_null_columns(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check for columns that are entirely null."""
        for col in df.columns:
            if df[col].isna().all():
                report.add_result(
                    ValidationResult(
                        validation_type=ValidationType.ALL_NULL_COLUMNS,
                        column=col,
                        severity=Severity.CRITICAL,
                        message=f"Column '{col}' is entirely null (all values missing)",
                        details={
                            "column": col,
                            "total_rows": len(df),
                        },
                        affected_rows=len(df),
                        affected_percentage=100.0,
                    )
                )

    def _check_out_of_range(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check for out-of-range values in numeric columns."""
        if not self.numeric_ranges:
            if self.reference_stats:
                for col, stats in self.reference_stats.items():
                    if col in df.columns and col in df.select_dtypes(include=[np.number]).columns:
                        min_val = stats.get("min", -np.inf)
                        max_val = stats.get("max", np.inf)
                        self.numeric_ranges[col] = (min_val, max_val)

        for col, (min_val, max_val) in self.numeric_ranges.items():
            if col not in df.columns:
                continue

            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            out_of_range = ((col_data < min_val) | (col_data > max_val)).sum()
            out_of_range_pct = (out_of_range / len(col_data)) * 100

            if out_of_range > 0:
                severity = (
                    Severity.CRITICAL
                    if out_of_range_pct > self.out_of_range_threshold * 100
                    else Severity.WARNING
                )
                report.add_result(
                    ValidationResult(
                        validation_type=ValidationType.OUT_OF_RANGE,
                        column=col,
                        severity=severity,
                        message=f"Column '{col}' has {out_of_range} out-of-range values ({out_of_range_pct:.2f}%) "
                        f"[expected range: {min_val} - {max_val}]",
                        details={
                            "out_of_range_count": int(out_of_range),
                            "out_of_range_percentage": round(out_of_range_pct, 4),
                            "expected_min": min_val,
                            "expected_max": max_val,
                            "actual_min": float(col_data.min()),
                            "actual_max": float(col_data.max()),
                            "threshold_percentage": self.out_of_range_threshold * 100,
                        },
                        affected_rows=int(out_of_range),
                        affected_percentage=out_of_range_pct,
                    )
                )

    def _check_unexpected_categorical(
        self, df: pd.DataFrame, report: DataQualityReport
    ) -> None:
        """Check for unexpected categorical values."""
        for col, expected_categories in self.categorical_domains.items():
            if col not in df.columns:
                continue

            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            actual_categories = set(col_data.unique())
            expected_set = set(expected_categories)
            unexpected = actual_categories - expected_set

            if unexpected:
                unexpected_count = col_data.isin(unexpected).sum()
                unexpected_pct = (unexpected_count / len(col_data)) * 100

                severity = (
                    Severity.CRITICAL
                    if unexpected_pct > self.unexpected_categorical_threshold * 100
                    else Severity.WARNING
                )
                report.add_result(
                    ValidationResult(
                        validation_type=ValidationType.UNEXPECTED_CATEGORICAL,
                        column=col,
                        severity=severity,
                        message=f"Column '{col}' has {len(unexpected)} unexpected categories: {sorted(unexpected)} "
                        f"({unexpected_pct:.2f}% of rows)",
                        details={
                            "unexpected_categories": sorted(unexpected),
                            "expected_categories": sorted(expected_categories),
                            "unexpected_count": int(unexpected_count),
                            "unexpected_percentage": round(unexpected_pct, 4),
                            "threshold_percentage": self.unexpected_categorical_threshold * 100,
                        },
                        affected_rows=int(unexpected_count),
                        affected_percentage=unexpected_pct,
                    )
                )

    def _check_invalid_records(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check for completely invalid records (all NaN or all invalid)."""
        invalid_mask = df.isna().all(axis=1)
        invalid_count = invalid_mask.sum()
        invalid_pct = (invalid_count / len(df)) * 100 if len(df) > 0 else 0

        if invalid_count > 0:
            report.add_result(
                ValidationResult(
                    validation_type=ValidationType.INVALID_RECORDS,
                    column=None,
                    severity=Severity.WARNING,
                    message=f"Found {invalid_count} completely invalid records (all NaN) ({invalid_pct:.2f}%)",
                    details={
                        "invalid_count": int(invalid_count),
                        "invalid_percentage": round(invalid_pct, 4),
                    },
                    affected_rows=int(invalid_count),
                    affected_percentage=invalid_pct,
                )
            )


def create_engine_from_reference(
    reference_df: pd.DataFrame,
    categorical_domains: dict[str, list] | None = None,
    missing_threshold: float = 0.1,
    duplicate_threshold: float = 0.01,
    out_of_range_threshold: float = 0.05,
    unexpected_categorical_threshold: float = 0.01,
) -> DataQualityEngine:
    """Create a DataQualityEngine configured from a reference DataFrame.

    Args:
        reference_df: Reference DataFrame to derive schema and statistics from
        categorical_domains: Optional explicit categorical domains
        missing_threshold: Threshold for missing value warnings
        duplicate_threshold: Threshold for duplicate warnings
        out_of_range_threshold: Threshold for out-of-range warnings
        unexpected_categorical_threshold: Threshold for unexpected categorical warnings

    Returns:
        Configured DataQualityEngine instance
    """
    schema: dict[str, Any] = {
        str(col): str(dtype) for col, dtype in reference_df.dtypes.items()
    }

    stats: dict[str, dict[str, float]] = {}
    for col in reference_df.select_dtypes(include=[np.number]).columns:
        col_data = reference_df[col].dropna()
        if len(col_data) > 0:
            stats[str(col)] = {
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
            }

    domains: dict[str, list[Any]] = categorical_domains or {}
    for col in reference_df.select_dtypes(include=["object", "category"]).columns:
        col_key = str(col)
        if col_key not in domains:
            unique_vals = reference_df[col].dropna().unique()
            domains[col_key] = sorted([str(v) for v in unique_vals])

    numeric_ranges = {
        col: (stats[col]["min"], stats[col]["max"]) for col in stats
    }

    return DataQualityEngine(
        reference_schema=schema,
        reference_stats=stats,
        categorical_domains=domains,
        numeric_ranges=numeric_ranges,
        missing_threshold=missing_threshold,
        duplicate_threshold=duplicate_threshold,
        out_of_range_threshold=out_of_range_threshold,
        unexpected_categorical_threshold=unexpected_categorical_threshold,
    )
