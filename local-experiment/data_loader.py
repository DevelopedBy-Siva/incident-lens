"""
Data loader for training dataset.

Loads and validates training data in JSONL format.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


class TrainingExample:
    """Single training example matching production schema."""

    def __init__(self, data: Dict[str, Any]):
        self.input_data = data["input"]
        self.expected_output = data["expected_output"]

        # Input fields
        self.service = self.input_data["service"]
        self.environment = self.input_data["environment"]
        self.count = self.input_data["count"]
        self.logs = self.input_data["logs"]
        self.related_incidents = self.input_data.get("related_incidents", [])
        self.metadata = self.input_data.get("metadata", {})

        # Expected output fields (production schema)
        self.severity = self.expected_output["severity"]
        self.disposition = self.expected_output["disposition"]
        self.confidence = self.expected_output["confidence"]
        self.summary = self.expected_output["summary"]
        self.suspected_root_cause = self.expected_output.get("suspected_root_cause")
        self.next_steps = self.expected_output["next_steps"]
        self.ticket_title = self.expected_output["ticket_title"]
        self.ticket_body = self.expected_output["ticket_body"]

    def validate(self) -> List[str]:
        """Validate example against production schema."""
        errors = []

        # Validate severity
        valid_severities = ["low", "medium", "high", "critical"]
        if self.severity not in valid_severities:
            errors.append(f"Invalid severity: {self.severity}")

        # Validate disposition
        valid_dispositions = [
            "NO_ACTION",
            "OBSERVE",
            "NEEDS_DEV",
            "NEEDS_ONCALL",
            "ESCALATE",
        ]
        if self.disposition not in valid_dispositions:
            errors.append(f"Invalid disposition: {self.disposition}")

        # Validate confidence
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(f"Invalid confidence: {self.confidence}")

        # Validate required string fields
        if not self.summary or not self.summary.strip():
            errors.append("Missing or empty summary")

        if not self.next_steps or not isinstance(self.next_steps, list):
            errors.append("Missing or invalid next_steps")

        # Validate NO_ACTION examples don't require tickets
        if self.disposition == "NO_ACTION":
            # Acceptable for ticket fields to be empty
            pass
        else:
            # Other dispositions should have ticket info
            if not self.ticket_title or not self.ticket_title.strip():
                errors.append(f"Missing ticket_title for {self.disposition}")

        return errors

    def to_training_format(self) -> Dict[str, str]:
        """
        Convert to training format with input prompt and expected JSON output.
        This format will be used for fine-tuning.
        """
        # Build evidence context like production
        evidence_lines = ["=== Sample log lines ==="]
        for log in self.logs:
            evidence_lines.append(f"  {log}")

        if self.related_incidents:
            evidence_lines.append("\n=== Related open incidents ===")
            for ri in self.related_incidents:
                evidence_lines.append(
                    f"  • [{ri.get('source', 'unknown')}] {ri.get('signature', '')[:80]}"
                )
        else:
            evidence_lines.append("\n=== Related open incidents ===")
            evidence_lines.append("  (none)")

        evidence_context = "\n".join(evidence_lines)

        # Production-like prompt
        user_prompt = f"""Analyze this incident:

Source: {self.service} | Environment: {self.environment}
Count: {self.count}

{evidence_context}

Provide your analysis as a JSON object."""

        # Expected output as JSON
        output = {
            "severity": self.severity,
            "disposition": self.disposition,
            "confidence": self.confidence,
            "summary": self.summary,
            "suspected_root_cause": self.suspected_root_cause,
            "next_steps": self.next_steps,
            "ticket_title": self.ticket_title,
            "ticket_body": self.ticket_body,
        }

        return {
            "prompt": user_prompt,
            "completion": json.dumps(output, indent=2),
        }


def load_training_data(path: Path) -> List[TrainingExample]:
    """Load training data from JSONL file."""
    examples = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                example = TrainingExample(data)
                examples.append(example)
            except Exception as e:
                raise ValueError(f"Line {line_num}: Failed to parse - {e}")

    return examples


def validate_training_data(examples: List[TrainingExample]) -> Dict[str, Any]:
    """
    Validate all training examples and return statistics.
    """
    validation_report = {
        "total_examples": len(examples),
        "valid_examples": 0,
        "invalid_examples": 0,
        "errors": [],
        "severity_distribution": {"low": 0, "medium": 0, "high": 0, "critical": 0},
        "disposition_distribution": {
            "NO_ACTION": 0,
            "OBSERVE": 0,
            "NEEDS_DEV": 0,
            "NEEDS_ONCALL": 0,
            "ESCALATE": 0,
        },
        "benign_count": 0,
        "actionable_count": 0,
    }

    for idx, example in enumerate(examples):
        errors = example.validate()
        if errors:
            validation_report["invalid_examples"] += 1
            validation_report["errors"].append({"example_index": idx, "errors": errors})
        else:
            validation_report["valid_examples"] += 1

        # Statistics
        if example.severity in validation_report["severity_distribution"]:
            validation_report["severity_distribution"][example.severity] += 1

        if example.disposition in validation_report["disposition_distribution"]:
            validation_report["disposition_distribution"][example.disposition] += 1

        if example.disposition in ["NO_ACTION", "OBSERVE"]:
            validation_report["benign_count"] += 1
        else:
            validation_report["actionable_count"] += 1

    return validation_report
