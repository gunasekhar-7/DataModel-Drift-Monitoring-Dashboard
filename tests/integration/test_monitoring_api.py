"""Integration tests for monitoring API endpoints."""

from fastapi.testclient import TestClient
import pytest

from app import get_app
from app.core.database import get_db_session, init_db
from app.models import Alert, ModelMetadata


@pytest.fixture(scope="module")
def client():
    """Create test client with seeded database."""
    init_db()
    with get_db_session() as db:
        # Seed test model metadata if not present
        if not db.query(ModelMetadata).filter_by(version="v1.0.0").first():
            db.add(
                ModelMetadata(
                    name="baseline_churn_model",
                    version="v1.0.0",
                    dataset_name="Telco Customer Churn",
                    dataset_version="v1.0.0",
                    metrics={"accuracy": 0.80, "roc_auc": 0.84},
                    artifact_path="app/ml/model.pkl",
                    status="production",
                )
            )
            db.commit()

    app = get_app()
    with TestClient(app) as test_client:
        yield test_client


def test_monitoring_health(client: TestClient):
    """Test /api/v1/monitoring/health endpoint."""
    response = client.get("/api/v1/monitoring/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_monitoring_model_health(client: TestClient):
    """Test /api/v1/monitoring/model-health endpoint."""
    response = client.get("/api/v1/monitoring/model-health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_monitoring_data_quality_history(client: TestClient):
    """Test /api/v1/monitoring/data-quality endpoint."""
    response = client.get("/api/v1/monitoring/data-quality?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "count" in data


def test_monitoring_drift_history(client: TestClient):
    """Test /api/v1/monitoring/drift endpoint."""
    response = client.get("/api/v1/monitoring/drift?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data


def test_monitoring_drift_summary(client: TestClient):
    """Test /api/v1/monitoring/drift/summary endpoint."""
    response = client.get("/api/v1/monitoring/drift/summary")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data or "feature_summaries" in data or "total_features_tested" in data


def test_monitoring_alerts_history(client: TestClient):
    """Test /api/v1/monitoring/alerts endpoint."""
    response = client.get("/api/v1/monitoring/alerts?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data


def test_monitoring_retraining_jobs(client: TestClient):
    """Test /api/v1/monitoring/retraining/jobs endpoint."""
    response = client.get("/api/v1/monitoring/retraining/jobs?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data

