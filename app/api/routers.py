"""API routers for the Data & Model Drift Monitoring Platform."""

from datetime import datetime, timezone
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.security import (
    add_request_id_to_logger,
    generate_request_id,
    get_request_id_from_request,
)
from app.ml.model import (
    load_model_artifacts,
    make_prediction,
    preprocess_features,
)
from app.models import (
    ModelMetadata,
    PredictionHistory,
    PredictionRequests,
    PredictionResults,
)
from app.schemas.schemas import (
    BatchPredictResponse,
    ErrorDetail,
    ErrorResponse,
    HealthCheckResponse,
    PredictBatchRequest,
    PredictionResponse,
    PredictRequest,
)

# Routers
predict_router = APIRouter(prefix="/api/v1", tags=["prediction"])
health_router = APIRouter(prefix="/api/v1", tags=["health"])


@health_router.get("/health", response_model=HealthCheckResponse)
async def health_check(request: Request) -> HealthCheckResponse:
    """Liveness and readiness health check endpoint."""
    request_id = get_request_id_from_request(request) or generate_request_id()

    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=settings.app_version,
    )


@predict_router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    payload: PredictRequest,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Single prediction endpoint.

    Accepts feature dictionary and returns model prediction.
    """
    start_time = time.time()

    # Generate or extract request ID
    header_request_id = get_request_id_from_request(request)
    request_id = header_request_id or generate_request_id()

    logger = logging.getLogger(__name__)
    logger = add_request_id_to_logger(logger, request_id)

    try:
        # Load model and preprocessing pipeline (cached)
        model, preprocessor, feature_schema = load_model_artifacts()

        # Validate and format features
        features_dict = payload.features

        # Apply preprocessing
        features_processed = preprocess_features(
            features_dict, preprocessor, feature_schema
        )

        # Make prediction
        prediction, probabilities = make_prediction(model, features_processed)

        # Extract scalar probability values safely
        prob_val: float | None = None
        prob_list: list[float] = []
        if probabilities is not None and len(probabilities) > 0:
            prob_row = probabilities[0]
            prob_list = [float(p) for p in prob_row]
            if len(prob_row) > 1:
                prob_val = float(prob_row[1])

        processing_time_ms = int((time.time() - start_time) * 1000)
        model_ver = getattr(model, "version", settings.app_version)

        # Build response
        response = PredictionResponse(
            request_id=request_id,
            model_version=model_ver,
            predicted_class=int(prediction[0]),
            probability=prob_val,
            prediction_probabilities=prob_list,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Persist to database
        try:
            pred_request = PredictionRequests(
                request_id=request_id,
                model_version=model_ver,
                features=json.dumps(features_dict),
                status="completed",
            )
            db.add(pred_request)

            pred_result = PredictionResults(
                request_id=request_id,
                predicted_class=int(prediction[0]),
                probability=prob_val,
                prediction_probabilities=prob_list,
                processing_time_ms=processing_time_ms,
            )
            db.add(pred_result)

            pred_history = PredictionHistory(
                request_id=request_id,
                model_version=model_ver,
                predicted_class=int(prediction[0]),
                probability=prob_val,
            )
            db.add(pred_history)

            # Ensure baseline model metadata exists
            model_meta = (
                db.query(ModelMetadata)
                .filter(ModelMetadata.version == model_ver)
                .first()
            )
            if not model_meta:
                model_meta = ModelMetadata(
                    name="baseline_churn_model",
                    version=model_ver,
                    dataset_name="Telco Customer Churn",
                    dataset_version="v1.0.0",
                    metrics={
                        "accuracy": 0.80,
                        "precision": 0.75,
                        "recall": 0.70,
                        "f1": 0.72,
                        "roc_auc": 0.85,
                    },
                    feature_schema=feature_schema,
                    artifact_path=settings.model_artifact_path,
                    status="production",
                )
                db.add(model_meta)

            db.commit()
        except Exception as e:
            db.rollback()
            logger.warning(
                f"Failed to persist prediction to database: {e}",
                extra={"request_id": request_id},
            )

        return response

    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.error(
            f"Prediction failed: {e!s}",
            extra={
                "request_id": request_id,
                "processing_time_ms": processing_time_ms,
                "error": str(e),
            },
        )

        error_detail = ErrorDetail(
            location=["body", "features"],
            message=str(e),
            error_type=type(e).__name__,
        )
        error_response = ErrorResponse(
            detail=str(e),
            errors=[error_detail],
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_response.model_dump(),
        )


@predict_router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    request: Request,
    payload: PredictBatchRequest,
    db: Session = Depends(get_db),
) -> BatchPredictResponse:
    """Batch prediction endpoint.

    Accepts list of feature dictionaries and returns predictions for all rows in a vectorized pass.
    """
    start_time = time.time()

    # Generate or extract request ID
    header_request_id = get_request_id_from_request(request)
    request_id = header_request_id or generate_request_id()

    logger = logging.getLogger(__name__)
    logger = add_request_id_to_logger(logger, request_id)

    try:
        # Load model and preprocessing pipeline (cached)
        model, preprocessor, feature_schema = load_model_artifacts()

        # Vectorized preprocessing and inference
        features_processed = preprocess_features(
            payload.features, preprocessor, feature_schema
        )
        predictions, probabilities = make_prediction(model, features_processed)

        total_rows = len(payload.features)
        total_time_ms = int((time.time() - start_time) * 1000)
        per_row_ms = max(1, total_time_ms // total_rows) if total_rows > 0 else 0
        model_ver = getattr(model, "version", settings.app_version)
        now_iso = datetime.now(timezone.utc).isoformat()

        response_predictions: list[PredictionResponse] = []

        try:
            for i, feat_dict in enumerate(payload.features):
                row_req_id = f"{request_id}-{i}"
                pred_class = int(predictions[i])
                prob_val: float | None = None
                prob_list: list[float] = []

                if probabilities is not None and len(probabilities) > i:
                    prob_row = probabilities[i]
                    prob_list = [float(p) for p in prob_row]
                    if len(prob_row) > 1:
                        prob_val = float(prob_row[1])

                pred_response = PredictionResponse(
                    request_id=row_req_id,
                    model_version=model_ver,
                    predicted_class=pred_class,
                    probability=prob_val,
                    prediction_probabilities=prob_list,
                    processing_time_ms=per_row_ms,
                    timestamp=now_iso,
                )
                response_predictions.append(pred_response)

                db.add(
                    PredictionRequests(
                        request_id=row_req_id,
                        model_version=model_ver,
                        features=json.dumps(feat_dict),
                        status="completed",
                    )
                )
                db.add(
                    PredictionResults(
                        request_id=row_req_id,
                        predicted_class=pred_class,
                        probability=prob_val,
                        prediction_probabilities=prob_list,
                        processing_time_ms=per_row_ms,
                    )
                )
                db.add(
                    PredictionHistory(
                        request_id=row_req_id,
                        model_version=model_ver,
                        predicted_class=pred_class,
                        probability=prob_val,
                    )
                )

            db.commit()
        except Exception as e:
            db.rollback()
            logger.warning(
                f"Failed to persist batch predictions to database: {e}",
                extra={"request_id": request_id},
            )

        return BatchPredictResponse(
            predictions=response_predictions,
            total_count=len(response_predictions),
            request_id=request_id,
        )

    except Exception as e:
        logger.error(
            f"Batch prediction failed: {e!s}",
            extra={"request_id": request_id, "error": str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
