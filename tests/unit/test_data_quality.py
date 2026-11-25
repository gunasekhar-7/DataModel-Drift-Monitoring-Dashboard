"""Tests for Data Quality Engine."""

import sys

sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import pytest

from app.monitoring.data_quality import (
    Severity,
    ValidationType,
    create_engine_from_reference,
)


class TestDataQualityEngine:
    """Tests for DataQualityEngine."""

    @pytest.fixture
    def reference_df(self):
        """Create a reference DataFrame for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'numeric_1': np.random.normal(100, 15, 1000),
            'numeric_2': np.random.uniform(0, 100, 1000),
            'categorical_1': np.random.choice(['A', 'B', 'C'], 1000),
            'categorical_2': np.random.choice(['X', 'Y'], 1000),
        })

    @pytest.fixture
    def engine(self, reference_df):
        """Create engine from reference data."""
        return create_engine_from_reference(reference_df)

    def test_engine_creation(self, reference_df):
        """Test engine creation from reference DataFrame."""
        engine = create_engine_from_reference(reference_df)
        assert engine is not None
        assert len(engine.reference_schema) == 4
        assert len(engine.reference_stats) == 2
        assert len(engine.categorical_domains) == 2
        assert len(engine.numeric_ranges) == 2

    def test_clean_data_passes(self, engine, reference_df):
        """Test that clean reference data passes validation."""
        test_df = reference_df.iloc[:100].copy()
        report = engine.validate(test_df, "test_batch_clean")
        assert report.passed is True
        assert len(report.results) == 0

    def test_missing_values_detection(self, engine, reference_df):
        """Test missing values detection."""
        test_df = reference_df.iloc[:100].copy()
        test_df.loc[0:10, 'numeric_1'] = np.nan
        report = engine.validate(test_df, "test_batch_missing")
        assert report.passed is False
        missing_results = report.get_results_by_type(ValidationType.MISSING_VALUES)
        assert len(missing_results) == 1
        assert missing_results[0].column == 'numeric_1'
        assert missing_results[0].affected_rows == 11

    def test_duplicate_detection(self, engine, reference_df):
        """Test duplicate row detection."""
        test_df = pd.concat([reference_df.iloc[:50], reference_df.iloc[:10]], ignore_index=True)
        report = engine.validate(test_df, "test_batch_dup")
        assert report.passed is False
        dup_results = report.get_results_by_type(ValidationType.DUPLICATES)
        assert len(dup_results) == 1
        assert dup_results[0].affected_rows == 10

    def test_infinite_values_detection(self, engine, reference_df):
        """Test infinite values detection."""
        test_df = reference_df.iloc[:100].copy()
        test_df.loc[0, 'numeric_1'] = np.inf
        test_df.loc[1, 'numeric_1'] = -np.inf
        report = engine.validate(test_df, "test_batch_inf")
        assert report.passed is False
        inf_results = report.get_results_by_type(ValidationType.INFINITE_VALUES)
        assert len(inf_results) == 1
        assert inf_results[0].column == 'numeric_1'
        assert inf_results[0].details['positive_inf'] == 1
        assert inf_results[0].details['negative_inf'] == 1

    def test_all_null_column_detection(self, engine, reference_df):
        """Test all-null column detection."""
        test_df = reference_df.iloc[:100].copy()
        test_df['all_null'] = np.nan
        report = engine.validate(test_df, "test_batch_null")
        assert report.passed is False
        null_results = report.get_results_by_type(ValidationType.ALL_NULL_COLUMNS)
        assert len(null_results) == 1
        assert null_results[0].column == 'all_null'

    def test_out_of_range_detection(self, engine, reference_df):
        """Test out-of-range values detection."""
        test_df = reference_df.iloc[:100].copy()
        # numeric_1 range is roughly 55-145 (mean 100, std 15)
        test_df.loc[0:5, 'numeric_1'] = 1000  # Way out of range
        report = engine.validate(test_df, "test_batch_oor")
        assert report.passed is False
        oor_results = report.get_results_by_type(ValidationType.OUT_OF_RANGE)
        assert len(oor_results) == 1
        assert oor_results[0].column == 'numeric_1'
        assert oor_results[0].affected_rows == 6

    def test_unexpected_categorical_detection(self, engine, reference_df):
        """Test unexpected categorical values detection."""
        test_df = reference_df.iloc[:100].copy()
        test_df.loc[0:3, 'categorical_1'] = 'UNEXPECTED_VALUE'
        report = engine.validate(test_df, "test_batch_unexp")
        assert report.passed is False
        unexp_results = report.get_results_by_type(ValidationType.UNEXPECTED_CATEGORICAL)
        assert len(unexp_results) == 1
        assert unexp_results[0].column == 'categorical_1'
        assert 'UNEXPECTED_VALUE' in unexp_results[0].details['unexpected_categories']

    def test_schema_compatibility_missing_column(self, engine, reference_df):
        """Test schema compatibility - missing expected column."""
        test_df = reference_df.iloc[:100].drop(columns=['categorical_1'])
        report = engine.validate(test_df, "test_batch_missing_col")
        assert report.passed is False
        schema_results = report.get_results_by_type(ValidationType.SCHEMA_COMPATIBILITY)
        assert len(schema_results) == 1
        assert schema_results[0].severity == Severity.CRITICAL
        assert 'categorical_1' in schema_results[0].details['missing_columns']

    def test_schema_compatibility_extra_column(self, engine, reference_df):
        """Test schema compatibility - extra unexpected column."""
        test_df = reference_df.iloc[:100].copy()
        test_df['extra_col'] = 'test'
        report = engine.validate(test_df, "test_batch_extra_col")
        assert report.passed is False
        schema_results = report.get_results_by_type(ValidationType.SCHEMA_COMPATIBILITY)
        assert len(schema_results) == 1
        assert schema_results[0].severity == Severity.WARNING
        assert 'extra_col' in schema_results[0].details['extra_columns']

    def test_empty_batch_detection(self, engine, reference_df):
        """Test empty batch detection."""
        empty_df = pd.DataFrame(columns=reference_df.columns)
        report = engine.validate(empty_df, "test_batch_empty")
        assert report.passed is False
        empty_results = report.get_results_by_type(ValidationType.EMPTY_BATCH)
        assert len(empty_results) == 1
        assert empty_results[0].severity == Severity.CRITICAL

    def test_invalid_records_detection(self, engine, reference_df):
        """Test completely invalid records detection."""
        test_df = reference_df.iloc[:100].copy()
        # Add a row with all NaN
        test_df.loc[100] = [np.nan, np.nan, np.nan, np.nan]
        report = engine.validate(test_df, "test_batch_invalid")
        invalid_results = report.get_results_by_type(ValidationType.INVALID_RECORDS)
        assert len(invalid_results) == 1
        assert invalid_results[0].affected_rows == 1

    def test_report_serialization(self, engine, reference_df):
        """Test report serialization to dictionary."""
        test_df = reference_df.iloc[:100].copy()
        test_df.loc[0, 'numeric_1'] = np.nan
        report = engine.validate(test_df, "test_batch_serialization")
        report_dict = report.to_dict()
        assert 'batch_id' in report_dict
        assert 'timestamp' in report_dict
        assert 'total_rows' in report_dict
        assert 'passed' in report_dict
        assert 'results' in report_dict
        assert 'summary' in report_dict
        assert len(report_dict['results']) == 1


class TestNumericalDriftDetector:
    """Tests for NumericalDriftDetector."""

    @pytest.fixture
    def reference_data(self):
        """Create reference data for drift tests."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.uniform(0, 10, 1000),
        })

    @pytest.fixture
    def detector(self):
        """Create drift detector."""
        from app.monitoring.drift.numerical import NumericalDriftDetector
        return NumericalDriftDetector(
            ks_threshold=0.05,
            psi_threshold=0.1,
            wasserstein_threshold=None,
            psi_bins=10,
            min_samples=30
        )

    def test_no_drift_same_distribution(self, detector, reference_data):
        """Test that same distribution shows no drift."""
        ref = reference_data.iloc[:500]
        cur = reference_data.iloc[500:1000]
        report = detector.detect_batch(ref, cur, ['feature_1', 'feature_2'], "batch_same", "ref")
        assert report.overall_drift is False
        for result in report.results:
            assert result.severity.value == 'no_drift'

    def test_ks_test_detects_mean_shift(self, detector, reference_data):
        """Test KS test detects mean shift."""
        ref = reference_data.iloc[:500]
        cur = reference_data.iloc[500:1000].copy()
        cur['feature_1'] = cur['feature_1'] + 2.0  # Shift by 2 std devs
        report = detector.detect_batch(ref, cur, ['feature_1'], "batch_shift", "ref")
        assert report.overall_drift is True
        ks_result = [r for r in report.results if r.test_name == 'kolmogorov_smirnov'][0]
        assert ks_result.severity.value == 'drift_detected'
        assert ks_result.p_value is not None
        assert ks_result.p_value < 0.05

    def test_psi_detects_distribution_shift(self, detector, reference_data):
        """Test PSI detects distribution shift."""
        ref = reference_data.iloc[:500]
        cur = reference_data.iloc[500:1000].copy()
        cur['feature_1'] = cur['feature_1'] + 1.5  # Shift by 1.5 std devs
        report = detector.detect_batch(ref, cur, ['feature_1'], "batch_shift", "ref")
        psi_result = [r for r in report.results if r.test_name == 'population_stability_index'][0]
        assert psi_result.severity.value == 'drift_detected'
        assert psi_result.statistic > 0.1

    def test_wasserstein_detects_shift(self, detector, reference_data):
        """Test Wasserstein distance detects shift."""
        ref = reference_data.iloc[:500]
        cur = reference_data.iloc[500:1000].copy()
        cur['feature_1'] = cur['feature_1'] + 1.0
        report = detector.detect_batch(ref, cur, ['feature_1'], "batch_shift", "ref")
        ws_result = [r for r in report.results if r.test_name == 'wasserstein_distance'][0]
        assert ws_result.severity.value == 'drift_detected'
        assert ws_result.statistic > 0

    def test_insufficient_samples_warning(self, detector, reference_data):
        """Test warning for insufficient samples."""
        ref = reference_data.iloc[:10]  # Too few samples
        cur = reference_data.iloc[:10]
        report = detector.detect_batch(ref, cur, ['feature_1'], "batch_small", "ref")
        assert report.overall_drift is False
        warning_results = [r for r in report.results if r.severity.value == 'warning']
        assert len(warning_results) >= 1
        assert warning_results[0].test_name == 'insufficient_samples'

    def test_missing_feature_warning(self, detector, reference_data):
        """Test warning for missing feature."""
        ref = reference_data.iloc[:100]
        cur = reference_data.iloc[:100]
        report = detector.detect_batch(ref, cur, ['nonexistent_feature'], "batch_missing", "ref")
        assert report.overall_drift is False
        warning_results = [r for r in report.results if r.severity.value == 'warning']
        assert len(warning_results) >= 1
        assert warning_results[0].test_name == 'missing_feature'

    def test_report_serialization(self, detector, reference_data):
        """Test drift report serialization."""
        ref = reference_data.iloc[:500]
        cur = reference_data.iloc[500:1000].copy()
        cur['feature_1'] = cur['feature_1'] + 2.0
        report = detector.detect_batch(ref, cur, ['feature_1'], "batch_test", "ref")
        report_dict = report.to_dict()
        assert 'batch_id' in report_dict
        assert 'reference_batch_id' in report_dict
        assert 'features_tested' in report_dict
        assert 'overall_drift' in report_dict
        assert 'results' in report_dict
        assert 'summary' in report_dict