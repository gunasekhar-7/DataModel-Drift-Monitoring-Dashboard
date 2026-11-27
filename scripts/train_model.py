"""Baseline ML model training pipeline.

Trains a baseline classification model on the Telco Customer Churn dataset
and persists the model, preprocessing pipeline, feature schema, and metadata.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_telco_data(filepath: str = None) -> pd.DataFrame:
    """Load the Telco Customer Churn dataset.

    Returns:
        Loaded DataFrame with raw telco data.
    """
    if filepath is None:
        possible_paths = [
            "data/raw/Telco-Customer-Churn.csv",
            "data/raw/churn.csv",
            "data/raw/telco_churn.csv",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break

    if filepath is None:
        raise FileNotFoundError(
            "Telco Churn dataset not found. "
            "Please place it at data/raw/Telco-Customer-Churn.csv"
        )

    df = pd.read_csv(filepath, encoding="utf-8")
    return df


def inspect_data(df: pd.DataFrame) -> dict[str, Any]:
    """Perform basic exploratory data analysis.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with EDA results.
    """
    profile = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        "missing_values": {
            col: int(count) for col, count in df.isnull().sum().items()
        },
        "class_distribution": df.iloc[:, -1].value_counts().to_dict()
        if len(df.columns) > 0
        else {},
    }
    return profile


def engineer_features(df: pd.DataFrame) -> tuple:
    """Engineer features for the classification model.

    Handles:
    - Missing value imputation
    - Categorical encoding
    - Numeric feature scaling
    - Feature column separation

    Returns:
        Tuple of (X, y, feature_info) where:
        - X: Feature DataFrame
        - y: Target Series
        - feature_info: Dictionary with feature schema information
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Identify target column (last column or 'churn')
    target_col = "churn"
    if target_col not in df.columns:
        # Try to find binary classification target
        target_col = df.columns[-1]

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle missing values in numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Impute numeric columns with median
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    # Impute categorical columns with mode
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else "unknown")

    # Convert target to binary if needed
    y = y.apply(lambda x: 1 if str(x).lower() in ["yes", "true", "churn", "1"] else 0)

    # Ensure target is integer
    y = y.astype(int)

    # Determine which numeric features need scaling vs. which are IDs
    # Exclude obvious ID columns (customer ID)
    id_columns = [col for col in X.columns if "customer" in col.lower() or "id" in col.lower()]
    numeric_features = [col for col in numeric_cols if col not in id_columns]

    # Ensure categorical features are string type
    for col in categorical_cols:
        X[col] = X[col].astype(str)

    feature_info = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_cols,
        "target_feature": target_col,
        "id_columns": id_columns,
        "total_features": len(numeric_features) + len(categorical_cols),
    }

    return X, y, feature_info


def create_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Create the preprocessing pipeline using ColumnTransformer.

    Args:
        numeric_features: List of numeric feature column names.
        categorical_features: List of categorical feature column names.

    Returns:
        Configured ColumnTransformer pipeline.
    """
    # Numeric pipeline: impute with median, scale with StandardScaler
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline: impute with most frequent, one-hot encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # Drop any other unclassified columns
    )

    return preprocessor


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    feature_info: dict[str, Any],
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train the baseline model and evaluate performance.

    Args:
        X: Feature matrix.
        y: Target vector.
        feature_info: Feature schema information.
        test_size: Proportion of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with training results and metrics.
    """
    # Split data chronologically (sort by tenure or contract date if available)
    # For telco data, we'll use a random split with fixed seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(
        feature_info["numeric_features"],
        feature_info["categorical_features"],
    )

    # Select model: XGBoost for imbalanced classification, Logistic Regression as baseline
    # Using XGBoost with scale_pos_weight for class imbalance
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        random_state=random_state,
        n_jobs=-1,
    )

    # Create full pipeline with preprocessing
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    # Train the model
    full_pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = full_pipeline.predict(X_test)
    y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        roc_auc = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Feature names after preprocessing
    preprocessor_names = list(full_pipeline.named_steps["preprocessor"].get_feature_names_out())
    model_features = preprocessor_names

    # Training results
    results = {
        "preprocessor": full_pipeline.named_steps["preprocessor"],
        "model": full_pipeline.named_steps["classifier"],
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
        },
        "confusion_matrix": cm,
        "feature_names": model_features,
        "feature_info": feature_info,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "random_state": random_state,
    }

    return results


def persist_artifacts(
    results: dict[str, Any],
    model_path: str = None,
    preprocessor_path: str = None,
    features_path: str = None,
    metadata_path: str = None,
) -> dict[str, str]:
    """Persist model artifacts to disk.

    Args:
        results: Training results dictionary from train_and_evaluate.
        model_path: Path to save the model (defaults to settings path).
        preprocessor_path: Path to save the preprocessor (defaults to settings path).
        features_path: Path to save the feature schema (defaults to settings path).
        metadata_path: Path to save the model metadata (defaults to settings path).

    Returns:
        Dictionary mapping artifact names to their saved file paths.
    """
    from app.core.config import settings

    if model_path is None:
        model_path = settings.model_artifact_path
    if preprocessor_path is None:
        preprocessor_path = settings.preprocessor_artifact_path
    if features_path is None:
        features_path = settings.feature_schema_path
    if metadata_path is None:
        metadata_path = settings.model_metadata_path

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    # Save model
    joblib.dump(results["model"], model_path)
    logger.info(f"Model saved to {model_path}")

    # Save preprocessor
    joblib.dump(results["preprocessor"], preprocessor_path)
    logger.info(f"Preprocessor saved to {preprocessor_path}")

    # Save feature schema
    feature_schema = {
        "numeric_features": results["feature_info"]["numeric_features"],
        "categorical_features": results["feature_info"]["categorical_features"],
        "target_feature": results["feature_info"]["target_feature"],
        "total_features": results["feature_info"]["total_features"],
    }
    with open(features_path, "w") as f:
        json.dump(feature_schema, f, indent=2)
    logger.info(f"Feature schema saved to {features_path}")

    # Save model metadata
    metadata = {
        "name": "baseline_churn_model",
        "version": results.get("model_version", "v1.0.0"),
        "training_date": datetime.now(timezone.utc).isoformat() + "Z",
        "dataset_name": "Telco Customer Churn",
        "dataset_version": "v1.0.0",
        "metrics": results["metrics"],
        "feature_schema": feature_schema,
        "artifact_path": model_path,
        "preprocessor_path": preprocessor_path,
        "feature_schema_path": features_path,
        "status": "production",
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model metadata saved to {metadata_path}")

    return {
        "model": model_path,
        "preprocessor": preprocessor_path,
        "features": features_path,
        "metadata": metadata_path,
    }


def train_baseline_pipeline(
    data_path: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
    model_version: str = "v1.0.0",
) -> dict[str, Any]:
    """Run the complete baseline training pipeline.

    This is the main entry point for training the baseline model:
    1. Load data
    2. EDA
    3. Feature engineering
    4. Train/test split
    5. Create preprocessing pipeline
    6. Train model
    7. Evaluate and metrics
    8. Persist artifacts
    9. Log run to MLflow if tracking server configured

    Args:
        data_path: Path to the Telco Churn dataset.
        test_size: Proportion of data for testing.
        random_state: Random seed for reproducibility.
        model_version: Model version tag string.

    Returns:
        Complete training results dictionary.
    """
    # Step 1: Load data
    logger.info("Loading Telco Customer Churn dataset...")
    df = load_telco_data(data_path)

    # Step 2: EDA
    logger.info(f"Dataset shape: {df.shape}")
    eda_profile = inspect_data(df)
    logger.info(f"EDA profile: {eda_profile}")

    # Step 3: Feature engineering
    logger.info("Engineering features...")
    X, y, feature_info = engineer_features(df)

    logger.info(f"Features - Numeric: {len(feature_info['numeric_features'])}, "
                f"Categorical: {len(feature_info['categorical_features'])}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    # Step 4-6: Train and evaluate
    logger.info("Training model and evaluating...")
    results = train_and_evaluate(X, y, feature_info, test_size=test_size, random_state=random_state)
    results["model_version"] = model_version

    # Step 7: Persist artifacts
    logger.info("Persisting model artifacts...")
    paths = persist_artifacts(results)

    # Combine results with paths
    results["artifact_paths"] = paths

    logger.info(
        f"Training complete! Metrics: "
        f"accuracy={results['metrics']['accuracy']:.4f}, "
        f"roc_auc={results['metrics']['roc_auc']:.4f}"
    )

    # Step 8: Log run to MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///app.db")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT", "drift_monitoring")
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"train_{model_version}"):
            mlflow.set_tag("model_version", model_version)
            mlflow.log_params({"test_size": test_size, "random_state": random_state})
            for m_key, m_val in results["metrics"].items():
                if isinstance(m_val, (int, float)):
                    mlflow.log_metric(m_key, float(m_val))
            logger.info(f"Logged to MLflow experiment '{experiment_name}' run for version '{model_version}'")
    except Exception as e:
        logger.info(f"MLflow logging notice: {e}")

    return results


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Train baseline churn prediction model.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw churn dataset")
    parser.add_argument("--model-version", type=str, default="v1.0.0", help="Model version identifier")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of test split")
    parser.add_argument("--random-state", type=int, default=42, help="Random state seed")

    args = parser.parse_args()

    results = train_baseline_pipeline(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
        model_version=args.model_version,
    )

    print("\n=== Training Complete ===")
    print(f"Version: {args.model_version}")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Precision: {results['metrics']['precision']:.4f}")
    print(f"Recall: {results['metrics']['recall']:.4f}")
    print(f"F1-score: {results['metrics']['f1']:.4f}")
    print(f"ROC-AUC: {results['metrics']['roc_auc']:.4f}")
    print("\nArtifacts saved:")
    for key, path in results["artifact_paths"].items():
        print(f"  {key}: {path}")
