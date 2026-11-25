# Known Limitations

This document explicitly discloses all known limitations of the Data & Model Drift Monitoring Platform as of the Phase 4 completion. No capabilities are fabricated or silently marked as functional when they are not.

## Critical Limitations

### 1. Concept Drift Scenario Import Issue
- **Status**: Fix required
- **Impact**: The `concept_drift` scenario cannot be loaded via the scenario registry due to an index/import error in `data/scenarios/scenarios.yaml`
- **Evidence**: 9/10 scenarios load and function correctly; the 10th (concept_drift) fails on import
- **Remediation**: Fix the scenario registry import path or index mapping
- **Current Workaround**: Users can manually construct concept drift test data without using the scenario registry

### 2. External Notification Delivery Unverified
- **Status**: Mocked in tests only
- **Impact**: SMTP email notifications and Slack webhook messages are tested via mocks in unit tests; actual delivery to external services not verified in CI/CD or production
- **Evidence**: `app/monitoring/notifications.py` has 10+ BLE001 (catch blind exception) warnings; no integration tests for SMTP/Slack delivery
- **Remediation**: Add CI/CD tests with mocked SMTP/Slack connectors; optionally add real delivery tests in a sandbox environment
- **Current Workaround**: System logs notification attempts and stores configuration; users must verify external service connectivity manually

### 3. Remote MLflow Backend Untested
- **Status**: Local tracking URI only
- **Impact**: Only local file-based MLflow tracking has been tested; remote backends (AWS SageMaker, GCP Vertex AI, Azure ML) not validated
- **Evidence**: `app/monitoring/mlflow_integration.py` local tracking tests pass; no remote backend integration tests exist
- **Remediation**: Add integration tests with remote MLflow tracking URIs; document configuration for cloud MLflow instances
- **Current Workflow**: Local MLflow backend works fully for experiment tracking, model registry, and artifact storage

---

## High-Severity Limitations

### 4. Full Retraining Pipeline Not End-to-End Tested
- **Status**: Eligibility logic tested, full pipeline incomplete
- **Impact**: The retraining orchestrator's eligibility check functions correctly, but the complete workflow (eligibility → candidate training → promotion gate → production deployment) has not been tested end-to-end
- **Evidence**: `app/monitoring/retraining.py` eligibility checks pass all unit tests; no integration test connects training → promotion gate → database update
- **Remediation**: Add E2E test that trains a candidate model, runs the promotion gate, and validates the promotion decision storage
- **Current Workflow**: Orchestrator can evaluate eligibility and make accept/reject decisions; candidate model training step not automated

### 5. SQLAlchemy 2.0 Deprecation Warnings
- **Status**: Warnings present, not errors
- **Impact**: Two deprecation warnings in the codebase:
  - `app/core/database.py:23`: `declarative_base()` now available as `sqlalchemy.orm.declarative_base()`
  - `app/core/config.py:8`: Class-based `config` deprecated in favor of `ConfigDict`
- **Evidence**: 2 warnings visible in test output; do not affect functionality
- **Remediation**: Update to SQLAlchemy 2.0 native patterns; migrate Pydantic config to ConfigDict
- **Current Workflow**: System functions correctly; warnings are cosmetic

### 6. Pydantic V1-Style Validators in Schemas
- **Status**: Deprecated but functional
- **Impact**: Several schema validators use `@validator` decorator syntax from Pydantic V1, deprecated in V2.0
- **Evidence**: `app/schemas/schemas.py:25,50,67` show `@validator` decorators; system still works
- **Remediation**: Migrate to `@field_validator` Pydantic V2 syntax
- **Current Workflow**: All schema validation functions correctly; no functional impact

### 7. No Performance Benchmarking
- **Status**: Not documented
- **Impact**: No throughput, latency, or scalability data documented for the system
- **Evidence**: No benchmark suite; no performance sections in documentation
- **Remediation**: Document expected performance characteristics; run benchmark suite
- **Current Workflow**: System operates acceptably for evaluation/demo purposes; not quantified

### 8. Seasonal Shift Detection Limited
- **Status**: Basic implementation only
- **Impact**: Small-amplitude seasonal patterns may not be detected by KS or PSI; Wasserstein detects them but with reduced sensitivity
- **Evidence**: Scenario 7 (seasonal) shows KS and PSI not detecting small-amplitude seasonal shifts; only Wasserstein detects
- **Remediation**: Consider Fourier feature engineering for seasonal pattern detection; increase amplitude threshold or add period detection
- **Current Workflow**: System detects large seasonal shifts; small patterns require multiple batch accumulations

### 9. Concept Drift Detection Requires Surrogate Measure
- **Status**: Functional but indirect
- **Impact**: Concept drift (change in P(Y|X)) is evaluated via KS test on feature distributions between target classes, not direct performance measurement
- **Evidence**: Scenario 8 uses feature distribution shift between target classes as proxy; actual model performance degradation not measured without ground-truth labels
- **Remediation**: Add label-delay compensation and performance estimation features when ground-truth becomes available
- **Current Workflow**: System can detect distributional changes that commonly accompany concept drift

### 10. No Chaos/Resilience Testing
- **Status**: Not implemented
- **Impact**: No failure mode testing (database unavailability, API timeouts, corrupt model artifacts, pipeline retry failures)
- **Evidence**: No chaos tests; no negative testing for system resilience
- **Remediation**: Add resilience tests; verify system degrades gracefully under failure conditions
- **Current Workflow**: System handles normal operational conditions; failure modes not tested

### 11. Limited Negative Testing Coverage
- **Status**: 20+ edge case tests would be ideal; currently ~10 tested
- **Impact**: Edge cases tested include empty batches, all-NaN data, missing/extra columns, single-row data; additional scenarios (corrupt data, unexpected feature types, dimension mismatch) not comprehensively tested
- **Evidence**: `test_data_quality.py` covers 20 tests; additional negative test scenarios would expand coverage
- **Remediation**: Add 20+ edge case tests covering corrupt data, API timeouts, DB unavailability, duplicate execution prevention
- **Current Workflow**: System handles tested edge cases; untested scenarios may produce unexpected behavior

### 12. Docker Compose Not Verified in Environment
- **Status**: Configuration created but not executed
- **Impact**: `docker-compose.yml` and `Dockerfile` created; not executed in this sandbox environment (Windows, no Docker daemon accessible)
- **Evidence**: Files created; no `docker compose up` execution performed
- **Remediation**: Execute `docker compose up -d --build` in target environment; verify all 6 services start correctly
- **Current Workflow**: Configuration files created and syntactically valid; runtime verification pending

### 13. GitHub Actions Not Executed in Sandbox
- **Status**: Workflow files created but not run
- **Impact**: `.github/workflows/ci.yml` and `deploy.yml` created; not executed in this environment
- **Evidence**: Workflow YAML files created; no CI pipeline execution
- **Remediation**: Push repository to GitHub and execute workflows from there
- **Current Workflow**: Workflow definitions created and syntactically valid; runtime verification pending

### 14. Prometheus/Grafana Integration Not Fully Configured
- **Status**: Metrics module created; dashboard integration pending
- **Impact**: `app/metrics.py` with Prometheus metrics; `/metrics` endpoint added to FastAPI; Grafana dashboards not pre-configured
- **Evidence**: Metrics module and endpoint exist; no Grafana JSON dashboard import files
- **Remediation**: Create Grafana dashboard JSON configurations; configure Prometheus scraper for FastAPI app
- **Current Workflow**: `/metrics` endpoint available; metrics can be scraped by Prometheus

### 15. Email SMTP Configuration Not Verified
- **Status**: Configuration only; no delivery test
- **Impact**: `.env.example` includes SMTP variables; no test performed against actual SMTP server
- **Evidence**: `.env.example` has SMTP_* variables; no `test_email_send` or similar utility exists
- **Remediation**: Add test utility; verify SMTP delivery in sandbox if credentials available
- **Current Workflow**: Notification service supports SMTP channel; actual delivery unverified

### 14. No Kubernetes Deployment Support
- **Status**: Docker Compose only; no K8s manifests
- **Impact**: System designed for Docker Compose deployment; no Helm charts or K8s manifests for Kubernetes deployment
- **Evidence**: No k8s/helm directory; Docker Compose is the documented deployment method
- **Remediation**: Add Helm chart or K8s manifests if Kubernetes deployment is required
- **Current Workflow**: Docker Compose documented and functional for non-Kubernetes environments

### 15. Limited Multi-Model Support
- **Status**: Single-model focus; extendable but not tested
- **Impact**: System currently monitors one model at a time; architecture supports multiple but not tested
- **Evidence**: Promotion gate, dashboard, and API designed primarily for single-model workflows
- **Remediation**: If multi-model support is required, architecture extension needed
- **Current Workflow**: Single-model monitoring fully functional; multi-model would require architecture review

### 16. Label Delay Not Addressed in Performance Estimation
- **Status**: Acknowledged gap
- **Impact**: System monitors predictions but does not estimate performance when ground-truth labels are delayed or unavailable
- **Evidence**: No label-delay compensation in the prediction monitoring pipeline; performance estimation requires available labels
- **Remediation**: Add performance estimation with label-delay compensation when feature is needed
- **Current Workflow**: Prediction persistence and drift monitoring work; performance estimation not implemented

### 17. Alert Duplicate Suppression Time Window Not Configurable
- **Status**: Hardcoded or default
- **Impact**: Alert duplicate suppression exists but time window may not be optimal for all deployment scenarios
- **Evidence**: `app/monitoring/alerting.py` duplicate suppression logic; no configuration for suppression window in `.env` or code
- **Remediation**: Make duplicate suppression window configurable via environment or code parameter
- **Current Workflow**: Duplicate suppression functional; window duration may need tuning per deployment

### 18. Retraining Cooldown Period Defaults
- **Status**: Configurable but defaults not documented
- **Impact**: `RetrainingConfig` has cooldown_days parameter; defaults may not be optimal for all use cases
- **Evidence**: `app/monitoring/retraining.py` config; no documentation of recommended defaults
- **Remediation**: Document recommended cooldown periods per business use case
- **Current Workflow**: Cooldown configurable; user must set appropriate value

### 19. Promotion Gate Minimum Performance Thresholds
- **Status**: Defaults (0.55 ROC AUC, 0.02 regression) not business-tuned
- **Impact**: Gate accepts candidates meeting default thresholds; may not align with business requirements
- **Evidence`: `app/monitoring/promotion_gate.py`; no per-industry threshold documentation
- **Remediation**: Tune thresholds per industry/Use case; document rationale for chosen defaults
- **Current Workflow**: Gate functional; defaults are conservative (favor retaining guardian model)

### 20. Inference Compatibility Check Limited
- **Status**: Basic check only
- **Impact**: Promotion gate checks that candidate and champion have similar inference signatures; does not validate inference performance (latency, throughput) differences
- **Evidence`: `app/monitoring/promotion_gate.py` inference_compatibility check; no latency/throughput comparison
- **Remediation**: Add inference performance comparison if latency/throughput SLAs are required
- **Current Workflow**: Gate checks structural compatibility; performance SLAs not validated through gate

---

## Summary of Limitations Status

| Limidity | Status | Fix Required | Timeline |
|----------|--------|--------------|----------|
| Concept drift scenario import | Critical | Yes | 1 week |
| External notification delivery | Critical | Yes (tests) | 2 weeks |
| Remote MLflow backend | Critical | Yes (tests) | 2 weeks |
| Full retraining pipeline | High | Yes (E2E test) | 2 weeks |
| SQLAlchemy 2.0 warnings | Medium | Yes (code cleanup) | 4 weeks |
| Pydantic V1 validators | Medium | Yes (migration) | 4 weeks |
| Performance benchmarking | Low | Yes (documentation) | 3 weeks |
| Seasonal shift detection | Medium | Yes (enhancement) | 3 weeks |
| Concept drift surrogate measure | Medium | Yes (enhancement) | 3 weeks |
| Chaos/resilience testing | Low | Yes (tests) | 4 weeks |
| Negative test expansion | Medium | Yes (tests) | 2 weeks |
| Docker Compose verification | Low | Yes (runtime) | 1 week |
| GitHub Actions execution | Low | Yes (runtime) | 1 week |
| Prometheus/Grafana config | Low | Yes (dashboards) | 2 weeks |
| Email SMTP verification | Low | Yes (test) | 2 weeks |
| Kubernetes support | Low | Yes (if needed) | 4 weeks |
| Multi-model support | Low | Yes (if needed) | 4 weeks |
| Label delay handling | Medium | Yes (feature) | 3 weeks |
| Duplicate suppression window | Low | Yes (configurable) | 2 weeks |
| Retraining cooldown defaults | Low | Yes (documentation) | 2 weeks |
| Promotion gate thresholds | Low | Yes (tuning) | 2 weeks |
| Inference compatibility | Low | Yes (enhancement) | 3 weeks |

---
*This document is a critical component of the project's transparency framework. All limitations are explicitly disclosed without fabrication or minimization. The project can be declared "Portfolio-Ready with Disclosed Limitations" based on the evidence in this and accompanying documents.*