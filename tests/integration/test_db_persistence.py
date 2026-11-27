"""Tests for database persistence."""

from sqlalchemy import text

from app.core.database import get_db_session, init_db
from app.core.security import generate_request_id
from app.models import (
    PredictionRequests,
    PredictionResults,
)


def test_database_initialization():
    """Test that all database tables can be initialized."""
    init_db()

    with get_db_session() as db:
        tables = db.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
        table_names = [t[0] for t in tables]

        expected_tables = [
            "model_metadata",
            "prediction_requests",
            "prediction_results",
            "batch_metadata",
            "prediction_history",
            "data_quality_metrics",
            "drift_metrics",
            "alerts",
            "retraining_jobs",
        ]
        for table in expected_tables:
            assert table in table_names, f"Table {table} not found in database"


def test_prediction_persistence():
    """Test that prediction requests and results can be persisted."""
    init_db()

    with get_db_session() as db:
        req_id = generate_request_id()

        pred_request = PredictionRequests(
            request_id=req_id,
            model_version="v1.0.0",
            features={"tenure": 30, "monthly_charges": 70.5},
            status="completed",
        )
        db.add(pred_request)

        pred_result = PredictionResults(
            request_id=req_id,
            predicted_class=1,
            probability=0.78,
            prediction_probabilities=[0.22, 0.78],
            processing_time_ms=25,
        )
        db.add(pred_result)
        db.commit()

        queried_request = (
            db.query(PredictionRequests).filter_by(request_id=req_id).first()
        )
        queried_result = (
            db.query(PredictionResults).filter_by(request_id=req_id).first()
        )

        assert queried_request is not None
        assert queried_result is not None
        assert queried_result.predicted_class == 1
        assert queried_result.probability == 0.78
