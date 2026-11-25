# Architecture Documentation

## Overview Architecture

The Data & Model Drift Monitoring Platform follows a **modular, layered architecture** with clean separation of concerns:

```
+---------------------+        +---------------------+        +---------------------+
|   Streamlit UI      | <----> |   FastAPI Backend   | <----> |   PostgreSQL DB      |
|   (Port 8501)       |        |   (Port 8000)       |        |   (Port 5432)        |
+---------------------+        +---------------------+        +---------------------+
        |                           |                           |
        |           Request/Resp    |           Queries         |
        |                           |                           |
        v                           v                           v
+---------------------+        +---------------------+        +---------------------+
|  Monitoring Router  |        |  API Routers        |        |  Tables & Records    |
|  (11 endpoints)     |        |  (predict, health)  |        |  (metrics, alerts)   |
+---------------------+        +---------------------+        +---------------------+
        |                           |                           |
        |           Business Logic        |           Orchestration   |
        v                           v                           v
+---------------------+        +---------------------+        +---------------------+
|  Data Quality Engine|        |  Drift Detectors    |        |  MLflow Tracking     |
|  (10 validators)    |        |  (KS, PSI, Wasserstein)|   |  (experiments, models)|
+---------------------+        +---------------------+        +---------------------+
        |                           |                           |
        |           Configuration     |           Artifacts       |
        v                           v                           v
+---------------------+        +---------------------+        +---------------------+
|  Scenario Registry  |        |  Retraining Orches- |        |  Model Artifacts     |
|  (10 drift scenarios)|      |   trator            |        |  (model.pkl, etc.)   |
+---------------------+        +---------------------+        +---------------------+
```

## Component Diagram

### 1. FastAPI Backend (`app/`)
The core server providing RESTful endpoints for monitoring, prediction, and governance.

**Sub-modules:**
- `app.api.monitoring_routes` - 11 monitoring-specific endpoints
- `app.api.routers` - Prediction and health endpoints
- `app.core` - Application factory, config, database initialization
- `app.monitoring.data_quality` - 10 data quality validators
- `app.monitoring.drift.numerical` - KS, PSI, Wasserstein drift detection
- `app.monitoring.alerting` - Alert engine with 20+ rules
- `app.monitoring.notifications` - Slack/Email notification channels
- `app.monitoring.mlflow_integration` - MLflow experiment tracking
- `app.monitoring.retraining` - Retirement orchestration with eligibility
- `app.monitoring.promotion_gate` - 8-check model promotion validation
- `app.monitoring.pipeline` - End-to-end monitoring pipeline

### 2. Streamlit Dashboard (`dashboard/`)
Interactive web interface for monitoring exploration and alert investigation.

**Pages (7):**
1. Model Health - Overall model status and latest metrics
2. Data Quality - Historical data quality trends and issue analysis
3. Feature Drift - Per-feature drift detection results
4. Prediction Monitoring - Prediction distribution over time
5. Model Performance - Model metrics across versions
6. Alert History - Active and historical alerts
7. Retraining - Retraining job status and governance

### 3. Data Layer
- **PostgreSQL** as the persistent storage layer
- **SQLAlchemy 2.0 ORM** for database interactions
- **Alembic** for migration management
- Key tables:
  - `prediction_history` - Request/result persistence
  - `model_metadata` - Model versioning and metrics
  - `data_quality_metrics` - Batch validation results
  - `drift_metrics` - Drift detection results per feature/test
  - `alerts` - Alert lifecycle history
  - `retraining_jobs` - Retraining orchestration records

### 4. ML Pipeline
- **Training**: `scripts/train_model.py` - Baseline model training with XGBoost
- **Batch Generation**: `scripts/generate_batches.py` - 10 scenario-based data generation
- **Inference**: FastAPI `/predict` and `/predict/batch` endpoints

### 5. Drift Detection Engine

Three complementary statistical methods:

| Component | Implementation | Strengths |
|-----------|---------------|-----------|
| **KS Test** | `app.monitoring.drift.numerical.KS_drift` | Mean shift detection, small sample support |
| **PSI** | `app.monitoring.drift.numerical.PSI_drift` | Industry standard, magnitude quantification |
| **Wasserstein** | `app.monitoring.drift.numerical.wasserstein_distance` | Distribution shift, gradient sensitivity |

All detectors operate on reference vs. current batch comparisons with automatic threshold scaling.

### 6. Alerting & Notification

**Alert Engine** (`app.monitoring.alerting`):
- 20+ pre-configured rules covering data quality and drift conditions
- Severity classification: info/warning/critical
- Duplicate suppression based on feature+severity+time window
- Lifecycle: active → acknowledged → resolved

**Notification Channels**:
- **Slack**: Webhook-based messages with alert summary
- **Email**: SMTP-enabled error/report messages

**Alert Rule Types**:
- Data quality: missing values, duplicates, schema changes, out-of-range, infinite values, all-null columns, unexpected categoricals
- Drift: KS test, PSI thresholds, Wasserstein distance
- Aggregate: multiple features drifting simultaneously

### 7. Retraining & Governance

**Retraining Orchestrator** (`app.monitoring.retraining`):
- Eligibility checking with cooldown periods (configurable)
- 5 trigger types for automated initiation
- Status tracking: pending → running → completed/failed
- Integration with promotion gate for model validation

**Promotion Gate** (`app.monitoring.promotion_gate`):
- 8 validation checks:
  1. Candidate model exists
  2. Artifacts loadable
  3. Schema compatibility
  4. Minimum performance threshold
  5. Candidate comparison vs. champion
  6. Inference compatibility
  7. Artifact integrity (file sizes)
  8. Full governance validation

**Eligibility Criteria**:
- Minimum ROC AUC > 0.55
- Maximum regression loss < 0.02
- Performance metrics within acceptable range
- No schema incompatibility with production
- Artifact integrity verified
- Champion currently in production must outperform candidate

### 8. MLflow Integration

**Experiment Tracking**:
- Automatic run naming with model/version/timestamp
- Tag injection: model_name, version, framework, training_date
- Model registry with stage transitions (none → staging → production)

**Champion-vs-Candidate**:
- Current production model (champion) vs. new candidate
- Comparison based on: performance, inference, schema compatibility
- Automated promotion gate decision

### 9. Deployment Topology

**Development/Testing** (Docker Compose):
```
+--------+      +--------+      +--------+      +--------+      +--------+
|        |      |        |      |        |      |        |      |        |
| Studio | ---> | FastAPI| ---> | Postgres| ---> | MLflow | ---> | Dashboard|
| (Local) |     | (8000) |     | (5432) |     | (5000) |     | (8501) |
|        |      |        |      |        |      |        |      |        |
+--------+      +--------+      +--------+      +--------+      +--------+
```

**Production**:
- API behind reverse proxy (nginx/caddy) with SSL
- PostgreSQL with replication and backups
- MLflow with artifact storage (S3/GCS)
- Dashboard accessed through API gateway
- Prometheus + Grafana for observability
- Separate CI/CD pipeline for deployments

## Database Schema

### Core Tables

```sql
-- Prediction history (request/result persistence)
CREATE TABLE prediction_history (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(50) NOT NULL,
    model_version VARCHAR(50),
    predicted_class INTEGER,
    probability FLOAT,
    prediction_probabilities JSONB,
    processing_time_ms INTEGER,
    timestamp TIMESTAMP NOT NULL,
    model_version VARCHAR(50),
    batch_id VARCHAR(50)
);

-- Model metadata
CREATE TABLE model_metadata (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    version VARCHAR(50) NOT NULL,
    dataset_name VARCHAR(100),
    dataset_version VARCHAR(50),
    metrics JSONB,
    feature_schema JSONB,
    artifact_path VARCHAR(200),
    status VARCHAR(20) DEFAULT 'staging',
    training_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Data quality metrics
CREATE TABLE data_quality_metrics (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(50) NOT NULL,
    model_version VARCHAR(50),
    total_rows INTEGER,
    total_columns INTEGER,
    passed BOOLEAN,
    missing_values_count INTEGER,
    duplicate_rows_count INTEGER,
    infinite_values_count INTEGER,
    out_of_range_count INTEGER,
    unexpected_categorical_count INTEGER,
    all_null_columns_count INTEGER,
    invalid_records_count INTEGER,
    details JSONB,
    reference_batch_id VARCHAR(50),
    timestamp TIMESTAMP NOT NULL,
    timestamp TIMESTAMP NOT NULL
);

-- Drift metrics
CREATE TABLE drift_metrics (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(50) NOT NULL,
    reference_batch_id VARCHAR(50),
    model_version VARCHAR(50),
    feature_name VARCHAR(100),
    test_name VARCHAR(50),
    statistic FLOAT,
    p_value FLOAT,
    threshold FLOAT,
    severity VARCHAR(20),
    overall_drift BOOLEAN,
    details JSONB,
    timestamp TIMESTAMP NOT NULL
);

-- Alerts
CREATE TABLE alerts (
    alert_id VARCHAR(50) PRIMARY KEY,
    batch_id VARCHAR(50),
    model_version VARCHAR(50),
    timestamp TIMESTAMP NOT NULL,
    alert_type VARCHAR(50),
    severity VARCHAR(20),
    feature_name VARCHAR(100),
    detection_method VARCHAR(50),
    measured_value FLOAT,
    threshold_value FLOAT,
    message TEXT,
    explanation TEXT,
    recommended_action TEXT,
    status VARCHAR(20) DEFAULT 'active',
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(100),
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(100)
);

-- Retraining jobs
CREATE TABLE retraining_jobs (
    job_id VARCHAR(50) PRIMARY KEY,
    trigger_batch_id VARCHAR(50),
    trigger_reason TEXT,
    status VARCHAR(20),
    mlflow_run_id VARCHAR(50),
    candidate_model_version VARCHAR(50),
    candidate_metrics JSONB,
    champion_model_version VARCHAR(50),
    champion_metrics JSONB,
    promotion_decision VARCHAR(10),
    promotion_reason TEXT,
    completed_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Indexes for Performance

```sql
-- Frequently queried indexes
CREATE INDEX idx_prediction_history_timestamp ON prediction_history(timestamp DESC);
CREATE INDEX idx_prediction_history_model_version ON prediction_history(model_version);
CREATE INDEX idx_drift_metrics_timestamp ON drift_metrics(timestamp DESC);
CREATE INDEX idx_drift_metrics_feature_test ON drift_metrics(feature_name, test_name);
CREATE INDEX idx_alerts_timestamp ON alerts(timestamp DESC);
CREATE INDEX idx_alerts_status ON alerts(status);
CREATE INDEX idx_retraining_jobs_status ON retraining_jobs(status);
CREATE INDEX idx_model_metadata_version ON model_metadata(version);
```

## API Documentation

### Authentication
The API currently operates without authentication for internal/debug use. For production deployment, add:
- FastAPI Security dependency with OAuth2 password flow
- API key validation in `app.core.security`
- Rate limiting per endpoint

### Error Handling
All API endpoints follow consistent error patterns:

```json
{
  "detail": "Error description",
  "error_type": "ExceptionClass",
  "location": ["body", "query", "path"],
  "request_id": "req-abc123def456"
}
```

HTTP status codes:
- `200` - Successful response
- `201` - Created (for resource creation)
- `400` - Bad request (validation error)
- `401` - Unauthorized (not implemented yet)
- `404` - Not found (alert/job ID not found)
- `422` - Validation error (Pydantic model validation)
- `500` - Internal server error

### Request IDs
All requests include a `X-Request-ID` header (or it's auto-generated). This ID is tracked through:
- API routes
- Database persistence
- Log entries
- Alert lifecycle
- Retraining job tracking

This enables end-to-end traceability for debugging and auditing.

## Installation & Setup

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/your-username/data-drift-monitoring.git
cd data-drift-monitoring

# 2. Install dependencies
poetry install

# 3. Configure environment
cp .env.example .env
# Edit .env with your configuration (database, MLflow, notifications)

# 4. Initialize the database
poetry run init-db
# or manually:
cd app
python -c "from app.core.database import init_db, Base, engine; init_db()"

# 5. Train the baseline model
poetry run train

# 6. Start the server
uvicorn app.core:create_app --host 0.0.0.0 --port 8000
```

### Docker Development

```bash
# Build and start all services
docker-compose up --build

# Or start in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Deployment

1. Set environment variables for production
2. Configure SSL termination at reverse proxy
3. Set up PostgreSQL with proper backups
4. Configure MLflow with persistent artifact storage
5. Set up Prometheus + Grafana for observability
6. Run database migration: `alembic upgrade head`
7. Verify all health checks pass
8. Monitor initial dashboard load for data consistency

## Development Roadmap

### Phase 5 (Planned)
- Concept drift detection enhancement
- Remote MLflow backend support (AWS SageMaker, GCP Vertex, Azure ML)
- Kubernetes deployment support
- Advanced alert routing and escalation
- predictive alerting (drift before performance degradation)
- Label-delay compensation for performance estimation

### Phase 6 (Planned)
- Multi-model support (parallel monitoring of multiple models)
- A/B testing integration
- Automated data drift remediation suggestions
- Explainable AI (XAI) integration for drift explanations
- Real-time streaming monitoring (Kafka integration)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow existing code style (ruff configured)
- Add unit tests for new functionality (maintain 100% test pass rate)
- Update documentation for API changes
- Run `poetry run lint` before committing
- Run `poetry run typecheck` before committing
- Ensure all new endpoints are documented in ARCHITECTURE.md

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Project repository: https://github.com/your-username/data-drift-monitoring

Issues and feedback: Use the GitHub Issues page.

License: MIT License - free for commercial and personal use.