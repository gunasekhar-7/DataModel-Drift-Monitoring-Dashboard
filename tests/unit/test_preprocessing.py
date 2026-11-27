"""Tests for data preprocessing."""

import sys

sys.path.insert(0, '.')


from scripts.train_model import (
    create_preprocessing_pipeline,
    engineer_features,
    inspect_data,
    load_telco_data,
)


def test_load_telco_data():
    """Test that the Telco Churn dataset can be loaded."""
    df = load_telco_data()
    assert df is not None
    assert len(df) > 0
    assert 'churn' in df.columns


def test_inspect_data():
    """Test EDA inspection returns expected structure."""
    df = load_telco_data()
    profile = inspect_data(df)
    assert 'shape' in profile
    assert 'columns' in profile
    assert 'missing_values' in profile
    assert 'class_distribution' in profile


def test_engineer_features():
    """Test feature engineering produces expected results."""
    df = load_telco_data()
    X, y, feature_info = engineer_features(df)

    assert len(X) == len(df)
    assert len(y) == len(df)
    assert 'numeric_features' in feature_info
    assert 'categorical_features' in feature_info
    assert 'target_feature' in feature_info
    assert feature_info['target_feature'] == 'churn'

    # Check that y is binary
    assert set(y.unique()) <= {0, 1}


def test_create_preprocessing_pipeline():
    """Test that the preprocessing pipeline can be created."""
    df = load_telco_data()
    X, y, feature_info = engineer_features(df)

    numeric_features = feature_info['numeric_features']
    categorical_features = feature_info['categorical_features']

    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    assert preprocessor is not None
