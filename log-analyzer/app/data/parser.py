import re
from datetime import datetime
from typing import Optional


class ParsedLog:
    """Structured representation of a log line"""

    def __init__(self, raw: str):
        self.raw = raw
        self.timestamp = self._extract_timestamp()
        self.level = self._extract_level()
        self.message = self._extract_message()
        self.exception_type = self._extract_exception()

    def _extract_timestamp(self) -> Optional[datetime]:
        """Extract ISO timestamp from log line"""
        match = re.search(r"\[([\d\-T:.]+)\]", self.raw)
        if match:
            try:
                return datetime.fromisoformat(match.group(1))
            except:
                pass
        return datetime.utcnow()

    def _extract_level(self) -> str:
        """Extract log level (ERROR, WARN, INFO, etc.)"""
        for level in ["CRITICAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG"]:
            if level in self.raw.upper():
                return level
        return "INFO"

    def _extract_message(self) -> str:
        """Extract the actual message, removing timestamp and level"""
        msg = re.sub(r"\[[\d\-T:.]+\]", "", self.raw)
        msg = re.sub(
            r"(CRITICAL|ERROR|WARN|WARNING|INFO|DEBUG):", "", msg, flags=re.IGNORECASE
        )
        return msg.strip()

    def _extract_exception(self) -> Optional[str]:
        """Extract exception type if present"""
        match = re.search(r"(\w+Exception)", self.message)
        if match:
            return match.group(1)

        if "Traceback" in self.message or "Error:" in self.message:
            return "PythonException"

        if "SQLSyntaxError" in self.message or "SQLError" in self.message:
            return "SQLException"

        return None
