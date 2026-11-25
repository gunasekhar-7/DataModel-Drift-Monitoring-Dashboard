"""Integration tests for core API endpoints."""

from fastapi.testclient import TestClient
import pytest

from app import get_app
from app.core.database import init_db


@pytest.fixture(scope="module")
def client():
    """Create test client with initialized database."""
    init_db()
    app = get_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client: TestClient):
    """Test /api/v1/health returns 200 and ISO-8601 timestamp."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data
    # Ensure timestamp is an ISO format string, not a request ID
    assert "T" in data["timestamp"] or "-" in data["timestamp"]


def test_single_prediction_endpoint(client: TestClient):
    """Test /api/v1/predict returns valid prediction."""
    payload = {
        "model_version": "v1.0.0",
        "features": {
            "tenure": 30,
            "monthly_charges": 70.5,
            "total_charges": 2100.0,
            "contract": "Month-to-month",
            "payment_method": "Electronic check",
            "gender": "Female",
            "senior_citizen": 0,
            "partner": "No",
            "dependents": "No",
        },
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert data["predicted_class"] in [0, 1]
    assert isinstance(data["prediction_probabilities"], list)
    assert len(data["prediction_probabilities"]) >= 1
    assert data["processing_time_ms"] >= 0


def test_single_prediction_validation_error(client: TestClient):
    """Test /api/v1/predict returns 422 for empty features."""
    response = client.post(
        "/api/v1/predict", json={"model_version": "v1.0.0", "features": {}}
    )
    assert response.status_code == 422


def test_batch_prediction_endpoint(client: TestClient):
    """Test /api/v1/predict/batch vectorized prediction."""
    payload = {
        "model_version": "v1.0.0",
        "features": [
            {
                "tenure": 12,
                "monthly_charges": 50.0,
                "total_charges": 600.0,
                "contract": "Month-to-month",
                "payment_method": "Mailed check",
                "gender": "Male",
                "senior_citizen": 0,
                "partner": "No",
                "dependents": "No",
            },
            {
                "tenure": 48,
                "monthly_charges": 85.0,
                "total_charges": 4000.0,
                "contract": "Two year",
                "payment_method": "Credit card (automatic)",
                "gender": "Female",
                "senior_citizen": 1,
                "partner": "Yes",
                "dependents": "Yes",
            },
        ],
    }
    response = client.post("/api/v1/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 2
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert pred["predicted_class"] in [0, 1]
        assert len(pred["prediction_probabilities"]) >= 1


def test_metrics_endpoint(client: TestClient):
    """Test /metrics exposes Prometheus formatted metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"app_requests_total" in response.content or response.text.startswith("#")

