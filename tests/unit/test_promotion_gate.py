"""Unit tests for the Promotion Gate."""

import uuid
import pytest

from app.core.database import get_db_session, init_db
from app.models import ModelMetadata
from app.monitoring.promotion_gate import PromotionGate


@pytest.fixture
def db_session():
    """Yield a database session for testing."""
    init_db()
    with get_db_session() as session:
        yield session


def test_promotion_gate_candidate_exists_check(db_session):
    """Test candidate existence validation check."""
    gate = PromotionGate(db=db_session)
    run_id = uuid.uuid4().hex[:8]

    # Check for nonexistent model
    check_fail = gate._check_candidate_exists(f"nonexistent_{run_id}")
    assert check_fail.passed is False
    assert "not found" in check_fail.message

    # Add mock model with unique version
    cand_version = f"v_cand_{run_id}"
    mock_model = ModelMetadata(
        name="test_churn_candidate",
        version=cand_version,
        dataset_name="Telco",
        artifact_path="app/ml/model.pkl",
        metrics={"accuracy": 0.85, "roc_auc": 0.88},
        status="staging",
    )
    db_session.add(mock_model)
    db_session.commit()

    check_pass = gate._check_candidate_exists(cand_version)
    assert check_pass.passed is True


def test_promotion_gate_minimum_performance_check(db_session):
    """Test minimum performance validation check against thresholds."""
    gate = PromotionGate(
        db=db_session,
        min_accuracy=0.70,
        min_roc_auc=0.60,
    )
    run_id = uuid.uuid4().hex[:8]
    low_version = f"v_low_perf_{run_id}"

    # Add sub-threshold model with unique version
    low_perf = ModelMetadata(
        name="low_perf_candidate",
        version=low_version,
        dataset_name="Telco",
        artifact_path="app/ml/model.pkl",
        metrics={"accuracy": 0.50, "roc_auc": 0.45},
        status="staging",
    )
    db_session.add(low_perf)
    db_session.commit()

    check = gate._check_minimum_performance(low_version)
    assert check.passed is False
    assert "accuracy" in check.message or "roc_auc" in check.message
