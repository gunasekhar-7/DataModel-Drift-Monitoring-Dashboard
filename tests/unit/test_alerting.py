"""Unit tests for the Alerting Engine."""

import uuid
from datetime import datetime, timezone

import pytest

from app.core.database import get_db_session, init_db
from app.models import Alert
from app.monitoring.alerting import (
    AlertContext,
    AlertEngine,
    AlertSeverity,
    AlertType,
)


@pytest.fixture
def db_session():
    """Yield a database session for testing."""
    init_db()
    with get_db_session() as session:
        yield session


def test_alert_creation_and_cooldown(db_session):
    """Test that AlertEngine creates alerts and suppresses duplicates without crashing."""
    engine = AlertEngine(db=db_session)
    run_id = uuid.uuid4().hex[:8]

    context = AlertContext(
        batch_id=f"test_batch_{run_id}",
        model_version="v1.0.0",
        timestamp=datetime.now(timezone.utc),
    )

    alert1 = engine._create_alert(
        context=context,
        rule_name="drift_ks_tenure",
        alert_type=AlertType.DRIFT,
        severity=AlertSeverity.CRITICAL,
        feature_name=f"tenure_{run_id}",
        detection_method="kolmogorov_smirnov",
        measured_value=0.0001,
        threshold_value=0.05,
        message="Distribution drift in tenure",
        explanation="Significant distribution shift",
        recommended_action="Investigate feature drift",
    )

    assert alert1 is not None
    engine._persist_alert(alert1)

    # Verify persisted in database
    saved = db_session.query(Alert).filter_by(alert_id=alert1.alert_id).first()
    assert saved is not None
    assert saved.feature_name == f"tenure_{run_id}"

    # Second alert within cooldown should be suppressed (return None)
    alert2 = engine._create_alert(
        context=context,
        rule_name="drift_ks_tenure",
        alert_type=AlertType.DRIFT,
        severity=AlertSeverity.CRITICAL,
        feature_name=f"tenure_{run_id}",
        detection_method="kolmogorov_smirnov",
        measured_value=0.0001,
        threshold_value=0.05,
        message="Distribution drift in tenure",
        explanation="Significant distribution shift",
        recommended_action="Investigate feature drift",
    )

    assert alert2 is None
    # Persisting None should not crash
    engine._persist_alert(alert2)


def test_alert_engine_evaluate_empty_metrics(db_session):
    """Test evaluation on clean context returns empty list without error."""
    engine = AlertEngine(db=db_session)
    context = AlertContext(
        batch_id=f"test_batch_clean_{uuid.uuid4().hex[:8]}",
        model_version="v1.0.0",
        timestamp=datetime.now(timezone.utc),
    )
    alerts = engine.evaluate(context)
    assert isinstance(alerts, list)
    assert len(alerts) == 0
