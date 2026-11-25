"""Notification service for alert delivery."""

from __future__ import annotations

import logging
import os
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests

from app.models import Alert

logger = logging.getLogger(__name__)


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""
    slack_webhook_url: str | None = None
    slack_channel: str | None = None
    email_smtp_host: str | None = None
    email_smtp_port: int = 587
    email_username: str | None = None
    email_password: str | None = None
    email_from: str | None = None
    email_to: list[str] = None
    email_use_tls: bool = True
    
    def __post_init__(self):
        if self.email_to is None:
            self.email_to = []
    
    @classmethod
    def from_env(cls) -> NotificationConfig:
        """Create config from environment variables."""
        email_to_str = os.getenv("ALERT_EMAIL_TO", "")
        email_to = [e.strip() for e in email_to_str.split(",") if e.strip()]
        
        return cls(
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
            slack_channel=os.getenv("SLACK_CHANNEL"),
            email_smtp_host=os.getenv("EMAIL_SMTP_HOST"),
            email_smtp_port=int(os.getenv("EMAIL_SMTP_PORT", "587")),
            email_username=os.getenv("EMAIL_USERNAME"),
            email_password=os.getenv("EMAIL_PASSWORD"),
            email_from=os.getenv("EMAIL_FROM"),
            email_to=email_to,
            email_use_tls=os.getenv("EMAIL_USE_TLS", "true").lower() == "true",
        )


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""
    
    @abstractmethod
    def send(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send notification for an alert.
        
        Returns:
            True if successful, False otherwise
        """
    
    @abstractmethod
    def is_configured(self, config: NotificationConfig) -> bool:
        """Check if channel is properly configured."""


class SlackNotificationChannel(NotificationChannel):
    """Slack webhook notification channel."""
    
    def is_configured(self, config: NotificationConfig) -> bool:
        return bool(config.slack_webhook_url)
    
    def send(self, alert: Alert, config: NotificationConfig) -> bool:
        if not self.is_configured(config):
            return False
        
        # Determine color based on severity
        color_map = {
            "warning": "#FFA500",  # Orange
            "critical": "#FF0000",  # Red
        }
        color = color_map.get(alert.severity, "#808080")
        
        # Build Slack message
        payload = {
            "channel": config.slack_channel,
            "username": "Drift Monitor",
            "icon_emoji": ":warning:" if alert.severity == "critical" else ":large_yellow_circle:",
            "attachments": [
                {
                    "color": color,
                    "title": f"[{alert.severity.upper()}] {alert.message}",
                    "fields": [
                        {
                            "title": "Alert ID",
                            "value": alert.alert_id,
                            "short": True,
                        },
                        {
                            "title": "Model Version",
                            "value": alert.model_version or "N/A",
                            "short": True,
                        },
                        {
                            "title": "Batch ID",
                            "value": alert.batch_id or "N/A",
                            "short": True,
                        },
                        {
                            "title": "Feature",
                            "value": alert.feature_name or "N/A (aggregate)",
                            "short": True,
                        },
                        {
                            "title": "Detection Method",
                            "value": alert.detection_method or "N/A",
                            "short": True,
                        },
                        {
                            "title": "Measured Value",
                            "value": f"{alert.measured_value:.4f}" if alert.measured_value is not None else "N/A",
                            "short": True,
                        },
                        {
                            "title": "Threshold",
                            "value": f"{alert.threshold_value:.4f}" if alert.threshold_value is not None else "N/A",
                            "short": True,
                        },
                    ],
                    "text": alert.explanation or "No explanation provided",
                    "footer": "Data & Model Drift Monitoring Platform",
                    "ts": int(alert.timestamp.timestamp()) if alert.timestamp else int(datetime.now(timezone.utc).timestamp()),
                }
            ],
        }
        
        # Add recommended action if present
        if alert.recommended_action:
            payload["attachments"][0]["fields"].append({
                "title": "Recommended Action",
                "value": alert.recommended_action,
                "short": False,
            })
        
        try:
            response = requests.post(
                config.slack_webhook_url,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            logger.info(f"Slack notification sent for alert {alert.alert_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Slack notification for alert {alert.alert_id}: {e}")
            return False


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""
    
    def is_configured(self, config: NotificationConfig) -> bool:
        return all([
            config.email_smtp_host,
            config.email_username,
            config.email_password,
            config.email_from,
            config.email_to,
        ])
    
    def send(self, alert: Alert, config: NotificationConfig) -> bool:
        if not self.is_configured(config):
            return False
        
        severity_label = alert.severity.upper()
        subject = f"[{severity_label}] Drift Monitor Alert: {alert.message}"
        
        # Build HTML email
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: {'#FF0000' if alert.severity == 'critical' else '#FFA500'};">
                    [{severity_label}] Drift Monitoring Alert
                </h2>
                
                <p><strong>Message:</strong> {alert.message}</p>
                
                <table style="border-collapse: collapse; width: 100%; margin: 20px 0;">
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold; width: 30%;">Alert ID</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert.alert_id}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Model Version</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert.model_version or 'N/A'}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Batch ID</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert.batch_id or 'N/A'}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Feature</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert.feature_name or 'N/A (aggregate)'}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Alert Type</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert.alert_type}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Detection Method</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert.detection_method or 'N/A'}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Measured Value</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{f'{alert.measured_value:.4f}' if alert.measured_value is not None else 'N/A'}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Threshold</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{f'{alert.threshold_value:.4f}' if alert.threshold_value is not None else 'N/A'}</td>
                    </tr>
                    <tr>
                        <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Timestamp</td>
                        <td style="border: 1px solid #ddd; padding: 8px;">{alert.timestamp.isoformat() if alert.timestamp else 'N/A'}</td>
                    </tr>
                </table>
                
                {f'<p><strong>Explanation:</strong> {alert.explanation}</p>' if alert.explanation else ''}
                
                {f'<p><strong>Recommended Action:</strong> <span style="background-color: #fff3cd; padding: 10px; display: inline-block;">{alert.recommended_action}</span></p>' if alert.recommended_action else ''}
                
                <hr style="margin: 20px 0;">
                <p style="font-size: 12px; color: #666;">
                    This alert was generated by the Data & Model Drift Monitoring Platform.
                    <br>Alert ID: {alert.alert_id}
                </p>
            </div>
        </body>
        </html>
        """
        
        # Also create plain text version
        text_body = f"""
[{severity_label}] Drift Monitoring Alert

Message: {alert.message}
Alert ID: {alert.alert_id}
Model Version: {alert.model_version or 'N/A'}
Batch ID: {alert.batch_id or 'N/A'}
Feature: {alert.feature_name or 'N/A (aggregate)'}
Alert Type: {alert.alert_type}
Detection Method: {alert.detection_method or 'N/A'}
Measured Value: {f'{alert.measured_value:.4f}' if alert.measured_value is not None else 'N/A'}
Threshold: {f'{alert.threshold_value:.4f}' if alert.threshold_value is not None else 'N/A'}
Timestamp: {alert.timestamp.isoformat() if alert.timestamp else 'N/A'}

{alert.explanation if alert.explanation else ''}

{alert.recommended_action if alert.recommended_action else ''}

---
This alert was generated by the Data & Model Drift Monitoring Platform.
Alert ID: {alert.alert_id}
        """
        
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = config.email_from
        msg["To"] = ", ".join(config.email_to)
        
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
        
        try:
            with smtplib.SMTP(config.email_smtp_host, config.email_smtp_port) as server:
                if config.email_use_tls:
                    server.starttls()
                server.login(config.email_username, config.email_password)
                server.send_message(msg)
            
            logger.info(f"Email notification sent for alert {alert.alert_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email notification for alert {alert.alert_id}: {e}")
            return False


class NotificationService:
    """Service for managing and sending notifications across multiple channels."""
    
    def __init__(
        self,
        config: NotificationConfig | None = None,
        channels: list[NotificationChannel] | None = None,
    ):
        """Initialize the notification service.
        
        Args:
            config: Notification configuration (loads from env if None)
            channels: Custom notification channels (uses defaults if None)
        """
        self.config = config or NotificationConfig.from_env()
        self.channels = channels or [
            SlackNotificationChannel(),
            EmailNotificationChannel(),
        ]
        
        # Log configuration status
        for channel in self.channels:
            if channel.is_configured(self.config):
                logger.info(f"Notification channel configured: {channel.__class__.__name__}")
            else:
                logger.debug(f"Notification channel not configured: {channel.__class__.__name__}")
    
    def send_alert(self, alert: Alert) -> dict[str, bool]:
        """Send alert notification through all configured channels.
        
        Args:
            alert: Alert to notify about
            
        Returns:
            Dictionary mapping channel names to success status
        """
        results = {}
        for channel in self.channels:
            if channel.is_configured(self.config):
                try:
                    success = channel.send(alert, self.config)
                    results[channel.__class__.__name__] = success
                except Exception as e:
                    logger.error(f"Error in {channel.__class__.__name__}: {e}")
                    results[channel.__class__.__name__] = False
            else:
                results[channel.__class__.__name__] = False
        
        return results
    
    def send_alerts(self, alerts: list[Alert]) -> dict[str, dict[str, bool]]:
        """Send multiple alerts.
        
        Args:
            alerts: List of alerts to notify about
            
        Returns:
            Dictionary mapping alert IDs to channel results
        """
        results = {}
        for alert in alerts:
            results[alert.alert_id] = self.send_alert(alert)
        return results
    
    def test_channels(self) -> dict[str, bool]:
        """Test all configured channels with a test message.
        
        Returns:
            Dictionary mapping channel names to test success status
        """
        from app.models import Alert
        
        test_alert = Alert(
            alert_id="test-alert-001",
            batch_id="test-batch",
            model_version="v1.0.0",
            timestamp=datetime.now(timezone.utc),
            alert_type="test",
            severity="warning",
            feature_name="test_feature",
            detection_method="test",
            measured_value=0.5,
            threshold_value=0.1,
            message="Test notification from Drift Monitoring Platform",
            explanation="This is a test notification to verify channel configuration",
            recommended_action="No action required",
            status="active",
        )
        
        return self.send_alert(test_alert)


def create_notification_service(
    config: NotificationConfig | None = None,
) -> NotificationService:
    """Factory function to create a NotificationService instance."""
    return NotificationService(config=config)