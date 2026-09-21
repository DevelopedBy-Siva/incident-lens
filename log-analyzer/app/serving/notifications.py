import os
import logging
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class NotificationService:
    """Send notifications to Discord and Email based on disposition"""

    def __init__(self, project=None):
        self.discord_escalate_url = project.discord_webhook_escalate if project else None
        self.discord_dev_url = project.discord_webhook_dev if project else None

        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.oncall_email = project.user_email if project else None
        self.dashboard_url = os.getenv("DASHBOARD_URL", "http://localhost:3000")

    def send_discord(self, webhook_url: str, incident, analysis) -> bool:
        """Send formatted message to Discord"""
        if not webhook_url:
            logger.info("Discord webhook not configured")
            return False

        color_map = {
            "critical": 0xFF0000,
            "high": 0xFF6B00,
            "medium": 0xFFDD00,
            "low": 0x00FF00,
        }
        color = color_map.get(analysis.severity, 0x808080)

        embed = {
            "title": f"{analysis.summary[:100]}",
            "description": analysis.summary,
            "color": color,
            "fields": [
                {
                    "name": "📍 Source",
                    "value": f"`{incident.source}` ({incident.environment})",
                    "inline": True,
                },
                {"name": "🔢 Count", "value": f"`{incident.count}x`", "inline": True},
                {
                    "name": "⚠️ Severity",
                    "value": f"`{analysis.severity.upper()}`",
                    "inline": True,
                },
                {
                    "name": "🎯 Disposition",
                    "value": f"`{analysis.disposition}`",
                    "inline": True,
                },
                {
                    "name": "🕐 First Seen",
                    "value": incident.first_seen.strftime("%Y-%m-%d %H:%M:%S"),
                    "inline": True,
                },
                {
                    "name": "🕐 Last Seen",
                    "value": incident.last_seen.strftime("%Y-%m-%d %H:%M:%S"),
                    "inline": True,
                },
                {
                    "name": "📝 Sample Error",
                    "value": f"```\n{incident.sample_lines[0][:500] if incident.sample_lines else 'N/A'}\n```",
                    "inline": False,
                },
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": f"Incident ID: {incident.id[:8]}"},
        }

        if analysis.next_steps:
            steps_text = "\n".join(
                [f"{i+1}. {step}" for i, step in enumerate(analysis.next_steps[:5])]
            )
            embed["fields"].append(
                {
                    "name": "🔧 Next Steps",
                    "value": steps_text[:1024],
                    "inline": False,
                }
            )

        if analysis.analysis_source == "llm" and analysis.ticket_title:
            embed["fields"].append(
                {
                    "name": "🎫 Ticket Draft",
                    "value": f"**{analysis.ticket_title}**\n\n{analysis.ticket_body[:500]}...",
                    "inline": False,
                }
            )

        payload = {
            "username": "Log Analyzer",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/2621/2621040.png",
            "embeds": [embed],
        }

        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 204:
                logger.info("Discord notification sent")
                return True
            else:
                logger.warning("Discord failed: %s", response.status_code)
                return False
        except Exception as e:
            logger.warning("Discord error: %s", e)
            return False

    def send_email(self, incident, analysis) -> bool:
        """Send email notification for NEEDS_ONCALL"""
        if not all(
            [self.smtp_host, self.smtp_user, self.smtp_password, self.oncall_email]
        ):
            logger.info("Email not configured")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{analysis.severity.upper()}] {analysis.summary[:80]}"
            msg["From"] = self.smtp_user
            msg["To"] = self.oncall_email

            text = f"""
INCIDENT ALERT: {analysis.disposition}

Summary: {analysis.summary}

Details:
- Source: {incident.source} ({incident.environment})
- Severity: {analysis.severity.upper()}
- Count: {incident.count}x
- First Seen: {incident.first_seen}
- Last Seen: {incident.last_seen}

Sample Error:
{incident.sample_lines[0] if incident.sample_lines else 'N/A'}

Next Steps:
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(analysis.next_steps or [])])}

---
Incident ID: {incident.id}
Dashboard: {self.dashboard_url}
            """

            html = f"""
<html>
<body style="font-family: Arial, sans-serif; color: #333;">
    <h2 style="color: #c53030;">🚨 INCIDENT ALERT: {analysis.disposition}</h2>
    
    <h3>{analysis.summary}</h3>
    
    <table style="border-collapse: collapse; margin: 20px 0;">
        <tr>
            <td style="padding: 8px; font-weight: bold;">Source:</td>
            <td style="padding: 8px;">{incident.source} ({incident.environment})</td>
        </tr>
        <tr style="background: #f5f5f5;">
            <td style="padding: 8px; font-weight: bold;">Severity:</td>
            <td style="padding: 8px; color: #c53030; font-weight: bold;">{analysis.severity.upper()}</td>
        </tr>
        <tr>
            <td style="padding: 8px; font-weight: bold;">Count:</td>
            <td style="padding: 8px;">{incident.count}x</td>
        </tr>
        <tr style="background: #f5f5f5;">
            <td style="padding: 8px; font-weight: bold;">First Seen:</td>
            <td style="padding: 8px;">{incident.first_seen}</td>
        </tr>
        <tr>
            <td style="padding: 8px; font-weight: bold;">Last Seen:</td>
            <td style="padding: 8px;">{incident.last_seen}</td>
        </tr>
    </table>
    
    <h4>Sample Error:</h4>
    <pre style="background: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 5px; overflow-x: auto;">
{incident.sample_lines[0] if incident.sample_lines else 'N/A'}
    </pre>
    
    <h4>Next Steps:</h4>
    <ol>
        {"".join([f"<li>{step}</li>" for step in (analysis.next_steps or [])])}
    </ol>
    
    <hr style="margin: 30px 0; border: none; border-top: 1px solid #ddd;">
    <p style="color: #666; font-size: 12px;">
        Incident ID: {incident.id}<br>
        <a href="{self.dashboard_url}">View Dashboard</a>
    </p>
</body>
</html>
            """

            msg.attach(MIMEText(text, "plain"))
            msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info("Email sent to %s", self.oncall_email)
            return True

        except Exception as e:
            logger.warning("Email error: %s", e)
            return False

    def route_notification(self, incident, analysis) -> bool:
        """Route notification based on disposition"""
        disposition = analysis.disposition

        if disposition == "ESCALATE":
            return self.send_discord(self.discord_escalate_url, incident, analysis)

        elif disposition == "NEEDS_ONCALL":
            return self.send_email(incident, analysis)

        elif disposition == "NEEDS_DEV":
            return self.send_discord(self.discord_dev_url, incident, analysis)

        else:
            logger.info("No notification for %s", disposition)
            return False


def get_notification_service(project=None) -> NotificationService:
    """Get notification service for a specific project"""
    return NotificationService(project=project)
