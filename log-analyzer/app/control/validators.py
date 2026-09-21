import re
import requests
from typing import Tuple


def validate_log_source_url(url: str) -> Tuple[bool, str]:
    """
    Validate log source URL by:
    1. Checking format
    2. Actually calling the URL to verify it's a valid log server
    3. Checking for expected response

    Returns (is_valid, error_message)
    """
    if not url:
        return False, "Log source URL is required"

    url_pattern = re.compile(
        r"^https?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)?$",
        re.IGNORECASE,
    )

    if not url_pattern.match(url):
        return False, "Invalid URL format. Must start with http:// or https://"

    url = url.rstrip("/")

    try:
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            try:
                data = response.json()

                if isinstance(data, dict) and "service" in data:
                    service_name = data.get("service", "").lower()

                    if "log" in service_name or data.get("status") == "running":
                        return (
                            True,
                            f"Connected to log server: {data.get('service', 'Unknown')}",
                        )
                    else:
                        return (
                            False,
                            f"URL responds but doesn't appear to be a log server (service: {service_name})",
                        )
                else:
                    return (
                        False,
                        "URL responds but doesn't return expected log server format",
                    )

            except ValueError:
                return False, "URL responds but doesn't return valid JSON"
        else:
            return False, f"Log server returned status code {response.status_code}"

    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to the log source URL. Is the server running?"

    except requests.exceptions.Timeout:
        return False, "Connection to log source URL timed out"

    except requests.exceptions.RequestException as e:
        return False, f"Error connecting to log source: {str(e)}"


def validate_url(url: str) -> Tuple[bool, str]:
    """
    Simple URL format validation (used for non-log URLs if needed).
    For log source URLs, use validate_log_source_url instead.
    """
    if not url:
        return False, "URL is required"

    url_pattern = re.compile(
        r"^https?://"
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"
        r"localhost|"
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        r"(?::\d+)?"
        r"(?:/?|[/?]\S+)?$",
        re.IGNORECASE,
    )

    if not url_pattern.match(url):
        return False, "Invalid URL format. Must start with http:// or https://"

    return True, ""


def validate_discord_webhook(webhook_url: str) -> Tuple[bool, str]:
    """
    Validate Discord webhook URL format and test it.
    Returns (is_valid, error_message)
    """
    if not webhook_url:
        return False, "Discord webhook URL is required"

    discord_pattern = re.compile(
        r"^https://(?:www\.)?(?:discord(?:app)?\.com)/api/webhooks/\d+/[\w-]+$"
    )
    if not discord_pattern.match(webhook_url):
        return (
            False,
            "Invalid Discord webhook URL format. Expected: https://discord.com/api/webhooks/...",
        )

    try:
        response = requests.post(
            webhook_url,
            json={
                "content": "**Log Analyzer Setup**: Discord webhook verified successfully!",
                "username": "Log Analyzer",
            },
            timeout=5,
        )

        if response.status_code == 204:
            return True, "Discord webhook verified and test message sent"
        else:
            return False, f"Webhook test failed with status {response.status_code}"

    except requests.exceptions.RequestException as e:
        return False, f"Could not reach Discord webhook: {str(e)}"


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate email format.
    Returns (is_valid, error_message)
    """
    if not email:
        return False, "Email is required"

    email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

    if not email_pattern.match(email):
        return False, "Invalid email format"

    return True, ""
