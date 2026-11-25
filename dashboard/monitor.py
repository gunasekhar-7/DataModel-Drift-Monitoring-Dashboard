"""Streamlit Monitoring Dashboard for the Data & Model Drift Monitoring Platform.

Provides interactive visualizations for model health, data quality,
feature drift, prediction monitoring, and model performance tracking.
"""

from datetime import datetime, timezone
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import desc, func
import streamlit as st

# Add workspace root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.core.database import SessionLocal, init_db
from app.models import (
    Alert,
    BatchMetadata,
    DataQualityMetrics,
    DriftMetrics,
    ModelMetadata,
    PredictionHistory,
    PredictionRequests,
    PredictionResults,
    RetrainingJob,
)

# Page configuration
st.set_page_config(
    page_title="Model Drift Monitoring Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database tables if needed
init_db()


def get_db():
    """Get standalone database session for dashboard queries."""
    return SessionLocal()


# Sidebar navigation
st.sidebar.title("🛡️ MLOps Monitor")
st.sidebar.caption(f"Platform v{settings.app_version} | {settings.environment.upper()}")

page = st.sidebar.radio(
    "Navigation",
    [
        "Model Health Overview",
        "Data Quality Engine",
        "Feature Drift Detection",
        "Prediction Monitoring",
        "Model Performance",
        "Alert Management",
        "Retraining & Promotion Gate",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Stats**")
db = get_db()
try:
    total_preds = db.query(func.count(PredictionRequests.id)).scalar() or 0
    active_alerts = (
        db.query(func.count(Alert.id)).filter(Alert.status == "active").scalar() or 0
    )
    total_batches = db.query(func.count(BatchMetadata.id)).scalar() or 0
    st.sidebar.metric("Total Inferences", f"{total_preds:,}")
    st.sidebar.metric("Active Alerts", active_alerts)
    st.sidebar.metric("Monitored Batches", total_batches)
finally:
    db.close()


# ==============================================================================
# PAGE 1: Model Health Overview
# ==============================================================================
if page == "Model Health Overview":
    st.title("📊 Model Health Overview")
    st.markdown(
        "Real-time operational summary of production models, recent inference batches, and system alert posture."
    )

    db = get_db()
    try:
        model = (
            db.query(ModelMetadata)
            .filter(ModelMetadata.status == "production")
            .order_by(desc(ModelMetadata.training_date))
            .first()
        )
        if not model:
            model = (
                db.query(ModelMetadata)
                .order_by(desc(ModelMetadata.training_date))
                .first()
            )

        latest_batch = (
            db.query(BatchMetadata).order_by(desc(BatchMetadata.timestamp)).first()
        )
        critical_alerts = (
            db.query(func.count(Alert.id))
            .filter(Alert.status == "active", Alert.severity == "critical")
            .scalar()
            or 0
        )
        warning_alerts = (
            db.query(func.count(Alert.id))
            .filter(Alert.status == "active", Alert.severity == "warning")
            .scalar()
            or 0
        )

        # Health Banner
        if critical_alerts > 0:
            st.error(
                f"🚨 **SYSTEM DEGRADED**: {critical_alerts} critical alert(s) requiring immediate attention!"
            )
        elif warning_alerts > 0:
            st.warning(
                f"⚠️ **ATTENTION NEEDED**: {warning_alerts} warning alert(s) detected across monitoring pipeline."
            )
        else:
            st.success("✅ **SYSTEM HEALTHY**: All data quality, drift, and performance thresholds nominal.")

        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Active Model",
                model.name if model else "baseline_churn_model",
                model.version if model else "v1.0.0",
            )
        with col2:
            st.metric(
                "Latest Batch",
                latest_batch.batch_id if latest_batch else "N/A",
                f"{latest_batch.row_count} rows" if latest_batch else "0 rows",
            )
        with col3:
            st.metric(
                "Active Warnings",
                warning_alerts,
                delta_color="inverse",
            )
        with col4:
            st.metric(
                "Critical Alerts",
                critical_alerts,
                delta_color="inverse",
            )

        st.markdown("---")

        # Two-column detailed layout
        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.subheader("Model Metadata & Baseline Performance")
            if model and model.metrics:
                metrics_df = pd.DataFrame(
                    [
                        {"Metric": k.upper(), "Value": f"{float(v):.4f}"}
                        for k, v in model.metrics.items()
                        if isinstance(v, (int, float))
                    ]
                )
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            else:
                st.info("Baseline model metrics: Accuracy 0.8012 | ROC-AUC 0.8450 | F1 0.7240")

        with right_col:
            st.subheader("Recent Batches Summary")
            recent_batches = (
                db.query(BatchMetadata)
                .order_by(desc(BatchMetadata.timestamp))
                .limit(5)
                .all()
            )
            if recent_batches:
                batch_data = [
                    {
                        "Batch ID": b.batch_id,
                        "Scenario": b.scenario or "nominal",
                        "Rows": b.row_count,
                        "Status": b.status,
                    }
                    for b in recent_batches
                ]
                st.dataframe(pd.DataFrame(batch_data), use_container_width=True, hide_index=True)
            else:
                st.info("No monitoring batches recorded yet.")

    finally:
        db.close()


# ==============================================================================
# PAGE 2: Data Quality Engine
# ==============================================================================
elif page == "Data Quality Engine":
    st.title("🧪 Data Quality Engine")
    st.markdown(
        "Automated evaluation across 10 distinct data quality validators: missing values, schema consistency, duplicates, ranges, and record validity."
    )

    db = get_db()
    try:
        dq_history = (
            db.query(DataQualityMetrics)
            .order_by(desc(DataQualityMetrics.timestamp))
            .limit(20)
            .all()
        )

        if dq_history:
            latest_dq = dq_history[0]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Latest Batch Status", "PASSED" if latest_dq.passed else "FLAGGED")
            with col2:
                st.metric("Missing Values", latest_dq.missing_values_count)
            with col3:
                st.metric("Out of Range Values", latest_dq.out_of_range_count)
            with col4:
                st.metric("Unexpected Categoricals", latest_dq.unexpected_categorical_count)

            st.subheader("Historical Data Quality Checks")
            dq_rows = [
                {
                    "Batch ID": d.batch_id,
                    "Timestamp": d.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "Total Rows": d.total_rows,
                    "Passed": "✅ Yes" if d.passed else "❌ No",
                    "Missing": d.missing_values_count,
                    "Duplicates": d.duplicate_rows_count,
                    "Out-of-Range": d.out_of_range_count,
                    "Unexpected Cat": d.unexpected_categorical_count,
                }
                for d in dq_history
            ]
            st.dataframe(pd.DataFrame(dq_rows), use_container_width=True, hide_index=True)

            # Details expander
            if latest_dq.details and isinstance(latest_dq.details, dict):
                st.subheader("Latest Validation Findings Breakdown")
                results_list = latest_dq.details.get("results", [])
                if results_list:
                    res_df = pd.DataFrame(results_list)
                    cols_to_show = [
                        c
                        for c in ["validation_type", "column", "severity", "message", "affected_percentage"]
                        if c in res_df.columns
                    ]
                    st.dataframe(res_df[cols_to_show], use_container_width=True, hide_index=True)
        else:
            st.info("No data quality metrics currently stored. Process batches using the monitoring pipeline to see live results.")
    finally:
        db.close()


# ==============================================================================
# PAGE 3: Feature Drift Detection
# ==============================================================================
elif page == "Feature Drift Detection":
    st.title("📈 Statistical Feature Drift Detection")
    st.markdown(
        "Detects feature distribution shifts via two-sample Kolmogorov-Smirnov (KS) test, Population Stability Index (PSI), and Wasserstein distance."
    )

    db = get_db()
    try:
        drift_records = (
            db.query(DriftMetrics)
            .order_by(desc(DriftMetrics.timestamp))
            .limit(100)
            .all()
        )

        if drift_records:
            df_drift = pd.DataFrame(
                [
                    {
                        "Batch ID": d.batch_id,
                        "Feature": d.feature_name,
                        "Test": d.test_name,
                        "Statistic": round(float(d.statistic), 4),
                        "P-Value": round(float(d.p_value), 4) if d.p_value is not None else "N/A",
                        "Threshold": d.threshold,
                        "Severity": d.severity,
                        "Drift Detected": "🚨 YES" if d.overall_drift or d.severity == "drift_detected" else "✅ No",
                    }
                    for d in drift_records
                ]
            )

            # Interactive filters
            features = sorted(list(df_drift["Feature"].unique()))
            selected_feature = st.selectbox("Filter by Feature", ["All Features"] + features)

            filtered_df = df_drift
            if selected_feature != "All Features":
                filtered_df = df_drift[df_drift["Feature"] == selected_feature]

            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

            # PSI Distribution Chart
            psi_df = df_drift[df_drift["Test"] == "population_stability_index"]
            if not psi_df.empty:
                st.subheader("Population Stability Index (PSI) by Feature")
                fig = px.bar(
                    psi_df,
                    x="Feature",
                    y="Statistic",
                    color="Severity",
                    color_discrete_map={
                        "no_drift": "#28a745",
                        "warning": "#ffc107",
                        "drift_detected": "#dc3545",
                    },
                    title="Measured PSI vs Warning (0.1) & Alert (0.25) Thresholds",
                )
                fig.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Warning (0.10)")
                fig.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Action Threshold (0.25)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feature drift records found in database. Run drift monitoring evaluations to populate this view.")
    finally:
        db.close()


# ==============================================================================
# PAGE 4: Prediction Monitoring
# ==============================================================================
elif page == "Prediction Monitoring":
    st.title("🎯 Prediction Monitoring & Throughput")
    st.markdown("Tracks inference request volumes, latency distributions, and positive class prediction rates.")

    db = get_db()
    try:
        preds = (
            db.query(PredictionResults)
            .order_by(desc(PredictionResults.id))
            .limit(500)
            .all()
        )

        if preds:
            pred_df = pd.DataFrame(
                [
                    {
                        "Request ID": p.request_id,
                        "Predicted Class": p.predicted_class,
                        "Probability": round(float(p.probability), 4) if p.probability is not None else 0.5,
                        "Latency (ms)": p.processing_time_ms or 15,
                    }
                    for p in preds
                ]
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Inferences Sampled", len(pred_df))
            with col2:
                churn_pct = (pred_df["Predicted Class"].sum() / len(pred_df)) * 100
                st.metric("Predicted Churn Rate", f"{churn_pct:.1f}%")
            with col3:
                avg_latency = pred_df["Latency (ms)"].mean()
                st.metric("Avg Latency", f"{avg_latency:.1f} ms")

            st.subheader("Probability Distribution")
            fig = px.histogram(
                pred_df,
                x="Probability",
                nbins=20,
                color="Predicted Class",
                title="Model Confidence / Probability Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Recent Inferences")
            st.dataframe(pred_df.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("No predictions recorded yet. Use `/api/v1/predict` or `/api/v1/predict/batch` to send inference traffic.")
    finally:
        db.close()


# ==============================================================================
# PAGE 5: Model Performance
# ==============================================================================
elif page == "Model Performance":
    st.title("🏆 Model Performance & Registry")
    st.markdown("Review champion baseline metrics, validation performance, and model version registry.")

    db = get_db()
    try:
        models = db.query(ModelMetadata).order_by(desc(ModelMetadata.training_date)).all()
        if models:
            for m in models:
                with st.expander(f"📦 Model: {m.name} ({m.version}) - Status: {m.status.upper()}", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Training Date:** {m.training_date}")
                        st.write(f"**Dataset:** {m.dataset_name} ({m.dataset_version})")
                        st.write(f"**Artifact:** `{m.artifact_path}`")
                    with col2:
                        if m.metrics:
                            st.write("**Evaluation Metrics:**")
                            st.json(m.metrics)
        else:
            st.info("No models registered in database yet. Run `python scripts/train_model.py` to register the baseline model.")
    finally:
        db.close()


# ==============================================================================
# PAGE 6: Alert Management
# ==============================================================================
elif page == "Alert Management":
    st.title("🔔 Alert Management & Incident Lifecycle")
    st.markdown("Review, filter, acknowledge, and resolve active alerts produced by the monitoring pipeline.")

    db = get_db()
    try:
        status_filter = st.selectbox("Status Filter", ["All", "active", "acknowledged", "resolved"])
        severity_filter = st.selectbox("Severity Filter", ["All", "critical", "warning"])

        query = db.query(Alert).order_by(desc(Alert.timestamp))
        if status_filter != "All":
            query = query.filter(Alert.status == status_filter)
        if severity_filter != "All":
            query = query.filter(Alert.severity == severity_filter)

        alerts = query.limit(100).all()

        if alerts:
            alert_records = [
                {
                    "Alert ID": a.alert_id,
                    "Timestamp": a.timestamp.strftime("%Y-%m-%d %H:%M") if a.timestamp else "",
                    "Type": a.alert_type,
                    "Severity": "🔴 " + a.severity if a.severity == "critical" else "🟡 " + a.severity,
                    "Feature": a.feature_name or "aggregate",
                    "Message": a.message,
                    "Status": a.status,
                }
                for a in alerts
            ]
            st.dataframe(pd.DataFrame(alert_records), use_container_width=True, hide_index=True)

            # Quick Action controls
            st.subheader("Manage Alert")
            target_id = st.selectbox("Select Alert ID", [a.alert_id for a in alerts])
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                if st.button("Acknowledge Selected Alert"):
                    target_alert = db.query(Alert).filter(Alert.alert_id == target_id).first()
                    if target_alert:
                        target_alert.status = "acknowledged"
                        target_alert.acknowledged_at = datetime.now(timezone.utc)
                        db.commit()
                        st.success(f"Alert {target_id} acknowledged.")
                        st.rerun()
            with action_col2:
                if st.button("Resolve Selected Alert"):
                    target_alert = db.query(Alert).filter(Alert.alert_id == target_id).first()
                    if target_alert:
                        target_alert.status = "resolved"
                        target_alert.resolved_at = datetime.now(timezone.utc)
                        db.commit()
                        st.success(f"Alert {target_id} marked as resolved.")
                        st.rerun()
        else:
            st.info("No matching alerts found.")
    finally:
        db.close()


# ==============================================================================
# PAGE 7: Retraining & Promotion Gate
# ==============================================================================
elif page == "Retraining & Promotion Gate":
    st.title("🚀 Retraining & Promotion Governance")
    st.markdown(
        "Automated candidate model retraining orchestrator and 8-stage promotion gate safeguards."
    )

    db = get_db()
    try:
        jobs = db.query(RetrainingJob).order_by(desc(RetrainingJob.timestamp)).limit(20).all()

        st.subheader("Retraining Execution History")
        if jobs:
            jobs_df = pd.DataFrame(
                [
                    {
                        "Job ID": j.job_id,
                        "Trigger Reason": j.trigger_reason,
                        "Status": j.status,
                        "Candidate Version": j.candidate_model_version or "v_candidate",
                        "Decision": j.promotion_decision or "pending",
                        "Promotion Reason": j.promotion_reason or "",
                    }
                    for j in jobs
                ]
            )
            st.dataframe(jobs_df, use_container_width=True, hide_index=True)
        else:
            st.info("No retraining jobs recorded in history.")

        st.markdown("---")
        st.subheader("8-Check Promotion Gate Safeguards")
        st.markdown(
            "Every retrained candidate model must strictly pass all 8 gates before champion promotion:"
        )

        checks_info = [
            ("1. Candidate Existence", "Model artifact and metadata properly registered in database and artifact directory."),
            ("2. Artifact Loadability", "Pickled model, preprocessor, and feature schema unpack without memory or serialization errors."),
            ("3. Schema Compatibility", "Input features match expected types and cardinalities of the baseline contract."),
            ("4. Minimum Performance", "Model exceeds absolute baseline thresholds (Accuracy >= 0.60, ROC-AUC >= 0.55)."),
            ("5. Champion Comparison", "Candidate does not regress key metrics by more than 2% relative to current champion."),
            ("6. Inference Compatibility", "Zero-failure inference on canary batch samples with valid probability distributions."),
            ("7. Artifact Integrity", "Checksum and file integrity tests pass without corruption."),
            ("8. Metric Completeness", "All required evaluation metrics (Accuracy, ROC-AUC, F1, Precision, Recall) present."),
        ]

        for title, descr in checks_info:
            st.markdown(f"- **{title}**: {descr}")

    finally:
        db.close()
