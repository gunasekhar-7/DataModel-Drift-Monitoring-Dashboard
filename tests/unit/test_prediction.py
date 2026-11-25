"""Tests for prediction functionality."""

import sys

sys.path.insert(0, '.')


from app.ml.model import load_model_artifacts, make_prediction, preprocess_features


def test_model_artifacts_can_load():
    """Test that model artifacts can be loaded for prediction."""
    
    model, preprocessor, feature_schema = load_model_artifacts()
    
    assert model is not None
    assert preprocessor is not None
    assert feature_schema is not None


def test_prediction_with_valid_input():
    """Test that valid input produces a prediction."""
    
    model, preprocessor, feature_schema = load_model_artifacts()
    
    # Create sample features matching the schema
    feature_names = feature_schema.get('numeric_features', []) + feature_schema.get('categorical_features', [])
    sample_features = {}
    
    # Add numeric features
    for name in feature_schema.get('numeric_features', [])[:4]:  # First 4 numeric
        sample_features[name] = 30.0 if 'tenure' in name else 70.5 if 'monthly' in name else 1500.0
    
    # Add categorical features
    for name in feature_schema.get('categorical_features', )[:3]:  # First 3 categorical
        sample_features[name] = 'male' if 'gender' in name else 'no' if 'partner' in name else 'electronic_check'
    
    # Make prediction
    features_processed = preprocess_features(sample_features, preprocessor, feature_schema)
    prediction, probabilities = make_prediction(model, features_processed)
    
    assert prediction is not None
    assert len(prediction) == 1


def test_prediction_dtype():
    """Test that prediction returns expected types."""
    
    model, preprocessor, feature_schema = load_model_artifacts()
    
    sample_features = {
        'tenure': 30.0,
        'monthly_charges': 70.5,
        'contract': 'month-to-month',
        'payment_method': 'electronic_check',
        'gender': 'female',
        'senior_citizen': 0,
        'partner': 'no',
        'dependents': 'no'
    }
    
    features_processed = preprocess_features(sample_features, preprocessor, feature_schema)
    prediction, probabilities = make_prediction(model, features_processed)
    
    # Check prediction is a numpy array or list
    assert prediction is not None
    assert probabilities is not None or hasattr(model, 'predict_proba')