"""Model loading and prediction utilities for the ML module."""

from functools import lru_cache
import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.core.config import settings


@lru_cache(maxsize=1)
def load_model_artifacts() -> tuple[Any, Any, dict[str, Any]]:
    """Load the trained model and preprocessing pipeline.

    Cached in-memory to prevent expensive disk reloads on every request.

    Returns:
        Tuple of (model, preprocessor, feature_schema)
    """
    model_path = settings.model_artifact_path
    preprocessor_path = settings.preprocessor_artifact_path
    features_path = settings.feature_schema_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(
            f"Preprocessor artifact not found at {preprocessor_path}"
        )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    if os.path.exists(features_path):
        with open(features_path, "r", encoding="utf-8") as f:
            feature_schema = json.load(f)
    else:
        feature_schema = {"numeric_features": [], "categorical_features": []}

    return model, preprocessor, feature_schema


def clear_model_cache() -> None:
    """Clear cached model artifacts after retraining/promotion."""
    load_model_artifacts.cache_clear()


def preprocess_features(
    features: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame,
    preprocessor: Any,
    feature_schema: dict[str, Any],
) -> Any:
    """Apply preprocessing pipeline to input features with deterministic ordering.

    Args:
        features: Single dictionary, list of feature dictionaries, or DataFrame.
        preprocessor: Fitted scikit-learn preprocessing pipeline.
        feature_schema: Feature schema with numeric and categorical features.

    Returns:
        Transformed features array ready for model inference.
    """
    if isinstance(features, pd.DataFrame):
        df = features.copy()
    elif isinstance(features, list):
        df = pd.DataFrame(features)
    else:
        df = pd.DataFrame([features])

    if feature_schema:
        numeric_cols = feature_schema.get("numeric_features", [])
        categorical_cols = feature_schema.get("categorical_features", [])
        # Strictly preserve schema list ordering across all calls
        expected_cols = list(numeric_cols) + [
            c for c in categorical_cols if c not in numeric_cols
        ]

        # Fill missing features with None/NaN
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None

        df = df[expected_cols]

    # Apply preprocessing transform
    transformed = preprocessor.transform(df)
    return transformed


def make_prediction(
    model: Any, features_processed: Any
) -> tuple[np.ndarray, np.ndarray | None]:
    """Make prediction using the trained model.

    Args:
        model: Trained model with predict and optional predict_proba methods.
        features_processed: Preprocessed features array.

    Returns:
        Tuple of (prediction, probabilities).
    """
    probabilities: np.ndarray | None = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_processed)

    prediction = model.predict(features_processed)
    prediction = np.asarray(prediction).flatten()

    return prediction, probabilities
