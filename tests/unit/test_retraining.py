"""Unit tests for the Retraining Orchestrator."""

from datetime import datetime, timezone
import uuid
import pytest

from app.core.database import get_db_session, init_db
from app.models import Alert
from app.monitoring.retraining import (
    RetrainingConfig,
    RetrainingOrchestrator,
    RetrainingTrigger,
)


@pytest.fixture
def db_session():
    """Yield a database session for testing."""
    init_db()
    with get_db_session() as session:
        yield session


def test_retraining_orchestrator_initialization(db_session):
    """Test RetrainingOrchestrator initializes properly with config."""
    config = RetrainingConfig(
        critical_drift_count_threshold=2,
        drift_window_hours=12,
        retraining_cooldown_hours=24,
    )
    orchestrator = RetrainingOrchestrator(db=db_session, config=config)
    assert orchestrator.config.critical_drift_count_threshold == 2
    assert orchestrator.config.drift_window_hours == 12


def test_retraining_eligibility_no_triggers(db_session):
    """Test eligibility returns false when no drift or data triggers are met."""
    config = RetrainingConfig(
        critical_drift_count_threshold=99999,
        retraining_cooldown_hours=0,
    )
    orchestrator = RetrainingOrchestrator(db=db_session, config=config)
    eligibility = orchestrator.check_eligibility()
    assert eligibility.eligible is False
    assert eligibility.trigger is None


def test_retraining_eligibility_drift_trigger(db_session):
    """Test eligibility triggers when critical drift alerts accumulate in window."""
    config = RetrainingConfig(
        critical_drift_count_threshold=2,
        drift_window_hours=24,
        retraining_cooldown_hours=0,
    )
    orchestrator = RetrainingOrchestrator(db=db_session, config=config)

    # Insert 2 critical drift alerts with unique alert_ids
    run_id = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc)
    for i in range(2):
        alert = Alert(
            alert_id=f"test-drift-{run_id}-{i}",
            batch_id=f"batch-test-{run_id}-{i}",
            model_version="v1.0.0",
            timestamp=now,
            alert_type="drift",
            severity="critical",
            feature_name=f"feature_{i}",
            detection_method="ks_test",
            measured_value=0.001,
            threshold_value=0.05,
            message="Critical drift detected",
            status="active",
        )
        db_session.add(alert)
    db_session.commit()

    eligibility = orchestrator._check_sustained_critical_drift()
    assert eligibility.eligible is True
    assert eligibility.trigger == RetrainingTrigger.SUSTAINED_CRITICAL_DRIFT
