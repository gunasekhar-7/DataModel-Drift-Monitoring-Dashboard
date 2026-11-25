"""Pydantic schemas for prediction requests and responses."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class FeaturesSchema(BaseModel):
    """Schema for input features - dynamically populated."""


# Request schemas


class FeatureInput(BaseModel):
    """Individual feature input with validation hints."""

    name: str = Field(..., description="Feature name")
    value: Any = Field(..., description="Feature value")

    @field_validator("value")
    @classmethod
    def validate_numeric(cls, v: Any) -> Any:
        """Validate numeric features."""
        try:
            float(v)
        except (ValueError, TypeError):
            raise ValueError("Value must be numeric")
        return v


class PredictRequest(BaseModel):
    """Request schema for single prediction."""

    model_version: str | None = Field(
        default=None, description="Optional model version override"
    )
    features: dict[str, Any] = Field(
        ..., description="Dictionary of feature names to values"
    )
    required_features: list[str] | None = None

    @field_validator("features")
    @classmethod
    def features_must_not_be_empty(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v or len(v) == 0:
            raise ValueError("Features dictionary must not be empty")
        return v


class PredictBatchRequest(BaseModel):
    """Request schema for batch prediction."""

    model_version: str | None = Field(
        default=None, description="Optional model version override"
    )
    features: list[dict[str, Any]] = Field(
        ..., description="List of feature dictionaries, one per row"
    )

    @field_validator("features")
    @classmethod
    def features_list_must_not_be_empty(
        cls, v: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not v or len(v) == 0:
            raise ValueError("Features list must not be empty")
        if any(not row for row in v):
            raise ValueError("Feature dictionaries must not be empty")
        return v


# Response schemas


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""

    request_id: str = Field(..., description="Unique request identifier")
    model_version: str = Field(..., description="Model version used for prediction")
    predicted_class: int = Field(..., description="Predicted class (0 or 1)")
    probability: float | None = Field(
        None, description="Probability of the predicted class"
    )
    prediction_probabilities: list[float] = Field(
        ..., description="Full probability vector for all classes"
    )
    processing_time_ms: int = Field(
        ..., description="Processing time in milliseconds"
    )
    timestamp: str = Field(..., description="ISO-8601 timestamp of prediction")


class BatchPredictResponse(BaseModel):
    """Response schema for batch prediction."""

    predictions: list[PredictionResponse]
    total_count: int = Field(..., description="Total number of predictions")
    request_id: str = Field(..., description="Unique request identifier")


# Health check schemas


class HealthCheckResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Service health status")
    timestamp: str = Field(..., description="ISO-8601 timestamp")
    version: str = Field(..., description="Application version")


# API error schemas


class ErrorDetail(BaseModel):
    """Detail of a validation error."""

    location: list[str] = Field(..., description="Dot-notation path to error")
    message: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Python error type")


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    detail: str = Field(..., description="Human-readable error summary")
    errors: list[ErrorDetail] | None = Field(
        None, description="List of validation errors"
    )
    request_id: str | None = Field(None, description="Request ID if available")
    timestamp: str = Field(
        ..., description="ISO-8601 timestamp when error occurred"
    )
