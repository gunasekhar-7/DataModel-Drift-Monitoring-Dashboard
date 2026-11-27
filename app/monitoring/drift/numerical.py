"""Numerical drift detection module.

Implements statistical drift detection for numerical features:
- Kolmogorov-Smirnov (KS) test
- Population Stability Index (PSI) - hand-coded, verified
- Wasserstein Distance (Earth Mover's Distance)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance


class DriftSeverity(Enum):
    """Severity levels for drift detection."""

    NO_DRIFT = "no_drift"
    WARNING = "warning"
    DRIFT_DETECTED = "drift_detected"


@dataclass
class NumericalDriftResult:
    """Result of numerical drift detection for a single feature."""

    feature_name: str
    test_name: str
    statistic: float
    p_value: float | None
    threshold: float
    severity: DriftSeverity
    details: dict[str, Any] | None = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "feature_name": self.feature_name,
            "test_name": self.test_name,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value) if self.p_value is not None else None,
            "threshold": float(self.threshold),
            "severity": self.severity.value,
            "details": self.details,
        }


@dataclass
class NumericalDriftReport:
    """Aggregated numerical drift report for a batch."""

    batch_id: str
    reference_batch_id: str | None
    features_tested: list[str]
    results: list[NumericalDriftResult]
    overall_drift: bool = False

    def add_result(self, result: NumericalDriftResult) -> None:
        """Add a drift result."""
        self.results.append(result)
        if result.severity == DriftSeverity.DRIFT_DETECTED:
            self.overall_drift = True

    def get_results_by_feature(self, feature_name: str) -> list[NumericalDriftResult]:
        """Get all results for a specific feature."""
        return [r for r in self.results if r.feature_name == feature_name]

    def get_results_by_test(self, test_name: str) -> list[NumericalDriftResult]:
        """Get all results for a specific test."""
        return [r for r in self.results if r.test_name == test_name]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "batch_id": self.batch_id,
            "reference_batch_id": self.reference_batch_id,
            "features_tested": self.features_tested,
            "overall_drift": self.overall_drift,
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total_tests": len(self.results),
                "drift_detected_count": len(
                    [r for r in self.results if r.severity == DriftSeverity.DRIFT_DETECTED]
                ),
                "warning_count": len(
                    [r for r in self.results if r.severity == DriftSeverity.WARNING]
                ),
                "no_drift_count": len(
                    [r for r in self.results if r.severity == DriftSeverity.NO_DRIFT]
                ),
            },
        }


class NumericalDriftDetector:
    """Detector for numerical feature drift using multiple statistical tests."""

    def __init__(
        self,
        ks_threshold: float = 0.05,
        psi_threshold: float = 0.1,
        wasserstein_threshold: float | None = None,
        psi_bins: int = 10,
        min_samples: int = 30,
    ):
        """Initialize the numerical drift detector.

        Args:
            ks_threshold: P-value threshold for KS test (drift if p < threshold)
            psi_threshold: PSI threshold (drift if PSI > threshold)
            wasserstein_threshold: Wasserstein distance threshold (auto if None)
            psi_bins: Number of bins for PSI calculation
            min_samples: Minimum samples required for valid test
        """
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.psi_bins = psi_bins
        self.min_samples = min_samples

    def detect(
        self,
        reference: np.ndarray | pd.Series,
        current: np.ndarray | pd.Series,
        feature_name: str,
        batch_id: str,
        reference_batch_id: str | None = None,
    ) -> NumericalDriftReport:
        """Run all drift detection tests on a numerical feature.

        Args:
            reference: Reference (baseline) data
            current: Current (production) data
            feature_name: Name of the feature
            batch_id: Current batch identifier
            reference_batch_id: Reference batch identifier

        Returns:
            NumericalDriftReport with all test results
        """
        ref_data = self._prepare_data(reference)
        cur_data = self._prepare_data(current)

        report = NumericalDriftReport(
            batch_id=batch_id,
            reference_batch_id=reference_batch_id,
            features_tested=[feature_name],
            results=[],
        )

        if len(ref_data) < self.min_samples or len(cur_data) < self.min_samples:
            report.add_result(
                NumericalDriftResult(
                    feature_name=feature_name,
                    test_name="insufficient_samples",
                    statistic=0.0,
                    p_value=None,
                    threshold=0.0,
                    severity=DriftSeverity.WARNING,
                    details={
                        "reference_samples": len(ref_data),
                        "current_samples": len(cur_data),
                        "min_samples_required": self.min_samples,
                    },
                )
            )
            return report

        ks_result = self._ks_test(ref_data, cur_data, feature_name)
        report.add_result(ks_result)

        psi_result = self._psi_test(ref_data, cur_data, feature_name)
        report.add_result(psi_result)

        ws_result = self._wasserstein_test(ref_data, cur_data, feature_name)
        report.add_result(ws_result)

        return report

    def detect_batch(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_names: list[str],
        batch_id: str,
        reference_batch_id: str | None = None,
    ) -> NumericalDriftReport:
        """Run drift detection on multiple numerical features.

        Args:
            reference_df: Reference DataFrame
            current_df: Current DataFrame
            feature_names: List of numerical feature names to test
            batch_id: Current batch identifier
            reference_batch_id: Reference batch identifier

        Returns:
            Aggregated NumericalDriftReport
        """
        report = NumericalDriftReport(
            batch_id=batch_id,
            reference_batch_id=reference_batch_id,
            features_tested=feature_names,
            results=[],
        )

        for feature in feature_names:
            if feature not in reference_df.columns or feature not in current_df.columns:
                report.add_result(
                    NumericalDriftResult(
                        feature_name=feature,
                        test_name="missing_feature",
                        statistic=0.0,
                        p_value=None,
                        threshold=0.0,
                        severity=DriftSeverity.WARNING,
                        details={"error": f"Feature '{feature}' not found in one or both DataFrames"},
                    )
                )
                continue

            ref_data = reference_df[feature].dropna()
            cur_data = current_df[feature].dropna()

            if len(ref_data) == 0 or len(cur_data) == 0:
                report.add_result(
                    NumericalDriftResult(
                        feature_name=feature,
                        test_name="empty_data",
                        statistic=0.0,
                        p_value=None,
                        threshold=0.0,
                        severity=DriftSeverity.WARNING,
                        details={
                            "reference_samples": len(ref_data),
                            "current_samples": len(cur_data),
                        },
                    )
                )
                continue

            feature_report = self.detect(
                ref_data, cur_data, feature, batch_id, reference_batch_id
            )
            report.results.extend(feature_report.results)
            if feature_report.overall_drift:
                report.overall_drift = True

        return report

    def _prepare_data(self, data: np.ndarray | pd.Series) -> np.ndarray:
        """Prepare data array by removing NaN and inf values."""
        raw = data.to_numpy() if isinstance(data, pd.Series) else data
        arr = np.asarray(raw, dtype=float)
        arr = arr[~np.isnan(arr)]
        arr = arr[~np.isinf(arr)]
        return arr

    def _ks_test(
        self, ref: np.ndarray, cur: np.ndarray, feature_name: str
    ) -> NumericalDriftResult:
        """Perform two-sample Kolmogorov-Smirnov test.

        Tests the null hypothesis that both samples are drawn from the same distribution.
        """
        try:
            statistic, p_value = stats.ks_2samp(ref, cur, alternative="two-sided", mode="auto")
        except Exception as e:
            warnings.warn(f"KS test failed for {feature_name}: {e}")
            statistic, p_value = 0.0, 1.0

        severity = (
            DriftSeverity.DRIFT_DETECTED
            if p_value < self.ks_threshold
            else DriftSeverity.NO_DRIFT
        )

        return NumericalDriftResult(
            feature_name=feature_name,
            test_name="kolmogorov_smirnov",
            statistic=float(statistic),
            p_value=float(p_value),
            threshold=self.ks_threshold,
            severity=severity,
            details={
                "test": "Two-sample KS test",
                "null_hypothesis": "Both samples from same distribution",
                "alternative": "two-sided",
                "reference_samples": len(ref),
                "current_samples": len(cur),
            },
        )

    def _psi_test(
        self, ref: np.ndarray, cur: np.ndarray, feature_name: str
    ) -> NumericalDriftResult:
        """Calculate Population Stability Index (PSI).

        PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
        where bins are created from reference data quantiles.

        Standard thresholds:
        - PSI < 0.1: No significant drift
        - 0.1 <= PSI < 0.25: Moderate drift (warning)
        - PSI >= 0.25: Significant drift (alert)
        """
        try:
            psi_value = self._calculate_psi(ref, cur, self.psi_bins)
        except Exception as e:
            warnings.warn(f"PSI calculation failed for {feature_name}: {e}")
            psi_value = 0.0

        if psi_value < 0.1:
            severity = DriftSeverity.NO_DRIFT
        elif psi_value < 0.25:
            severity = DriftSeverity.WARNING
        else:
            severity = DriftSeverity.DRIFT_DETECTED

        return NumericalDriftResult(
            feature_name=feature_name,
            test_name="population_stability_index",
            statistic=float(psi_value),
            p_value=None,
            threshold=self.psi_threshold,
            severity=severity,
            details={
                "test": "Population Stability Index (PSI)",
                "bins": self.psi_bins,
                "reference_samples": len(ref),
                "current_samples": len(cur),
                "interpretation": self._psi_interpretation(psi_value),
            },
        )

    def _calculate_psi(
        self, expected: np.ndarray, actual: np.ndarray, bins: int = 10
    ) -> float:
        """Calculate PSI between expected (reference) and actual (current) distributions.

        Uses quantile-based binning on expected data for stable bins.
        """
        expected = np.asarray(expected, dtype=float)
        actual = np.asarray(actual, dtype=float)

        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]

        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        quantiles = np.linspace(0, 1, bins + 1)
        bin_edges = np.quantile(expected, quantiles)
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) < 2:
            return 0.0

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)

        expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
        actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

        psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

        return psi

    def _psi_interpretation(self, psi: float) -> str:
        """Interpret PSI value."""
        if psi < 0.1:
            return "No significant population shift"
        elif psi < 0.25:
            return "Moderate population shift - investigate"
        else:
            return "Significant population shift - action required"

    def _wasserstein_test(
        self, ref: np.ndarray, cur: np.ndarray, feature_name: str
    ) -> NumericalDriftResult:
        """Calculate Wasserstein Distance (Earth Mover's Distance).

        Measures the minimum "work" to transform one distribution into another.
        Lower values indicate more similar distributions.
        """
        try:
            distance = wasserstein_distance(ref, cur)
        except Exception as e:
            warnings.warn(f"Wasserstein distance failed for {feature_name}: {e}")
            distance = 0.0

        threshold = self.wasserstein_threshold
        if threshold is None:
            ref_range = ref.max() - ref.min() if len(ref) > 0 else 1.0
            threshold = 0.1 * ref_range if ref_range > 0 else 0.1

        severity = (
            DriftSeverity.DRIFT_DETECTED
            if distance > threshold
            else DriftSeverity.NO_DRIFT
        )

        return NumericalDriftResult(
            feature_name=feature_name,
            test_name="wasserstein_distance",
            statistic=float(distance),
            p_value=None,
            threshold=float(threshold),
            severity=severity,
            details={
                "test": "Wasserstein Distance (Earth Mover's Distance)",
                "reference_samples": len(ref),
                "current_samples": len(cur),
                "reference_range": float(ref.max() - ref.min()) if len(ref) > 0 else 0.0,
                "auto_threshold": self.wasserstein_threshold is None,
            },
        )


def verify_psi_implementation() -> dict[str, Any]:
    """Verify PSI implementation against known test cases.

    Returns:
        Dictionary with verification results
    """
    detector = NumericalDriftDetector()

    np.random.seed(42)

    ref = np.random.normal(0, 1, 1000)
    cur_same = np.random.normal(0, 1, 1000)
    cur_shifted = np.random.normal(1, 1, 1000)
    cur_wider = np.random.normal(0, 2, 1000)

    psi_same = detector._calculate_psi(ref, cur_same)
    psi_shifted = detector._calculate_psi(ref, cur_shifted)
    psi_wider = detector._calculate_psi(ref, cur_wider)

    return {
        "identical_distributions": {
            "psi": psi_same,
            "expected": "< 0.1",
            "pass": psi_same < 0.1,
        },
        "mean_shift_1_std": {
            "psi": psi_shifted,
            "expected": "> 0.1 (typically 0.2-0.5)",
            "pass": psi_shifted > 0.1,
        },
        "variance_doubled": {
            "psi": psi_wider,
            "expected": "> 0.1",
            "pass": psi_wider > 0.1,
        },
    }
