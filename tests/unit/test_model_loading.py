"""Tests for model loading."""

import sys

sys.path.insert(0, '.')

import os

import joblib

from scripts.train_model import train_baseline_pipeline


def test_model_artifacts_exist():
    """Test that model artifacts exist after training."""
    # Train the model if artifacts don't exist
    results = train_baseline_pipeline()

    # Check that artifacts were persisted
    artifact_paths = results['artifact_paths']

    assert os.path.exists(artifact_paths['model'])
    assert os.path.exists(artifact_paths['preprocessor'])
    assert os.path.exists(artifact_paths['features'])
    assert os.path.exists(artifact_paths['metadata'])


def test_model_can_load():
    """Test that the trained model can be loaded independently."""

    artifact_paths = train_baseline_pipeline()['artifact_paths']

    model = joblib.load(artifact_paths['model'])
    preprocessor = joblib.load(artifact_paths['preprocessor'])

    assert model is not None
    assert preprocessor is not None
