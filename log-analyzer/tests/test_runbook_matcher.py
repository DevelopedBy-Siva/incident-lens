import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core import runbook_matcher
from app.core.runbook_loader import Runbook
from app.core.runbook_matcher import (
    HIGH_CONFIDENCE_THRESHOLD,
    get_runbook_candidates,
    match_runbook,
    select_runbook_for_incident,
)


def incident(signature="", sample_lines=None, **overrides):
    defaults = {
        "id": "incident-1",
        "signature": signature,
        "sample_lines": sample_lines or [],
        "count": 1,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class DeterministicRunbookMatcherTests(unittest.TestCase):
    def test_db_pool_exhaustion_matches_db_runbook(self):
        runbook, score = match_runbook(
            incident(
                signature="database connection pool exhausted",
                sample_lines=[
                    "too many connections",
                    "checkout database timeout",
                ],
            )
        )

        self.assertIsNotNone(runbook)
        self.assertEqual(runbook.id, "db_connection_pool_exhausted")
        self.assertGreaterEqual(score, HIGH_CONFIDENCE_THRESHOLD)

    def test_healthcheck_noise_matches_low_risk_runbook(self):
        runbook, score = match_runbook(
            incident(
                signature="synthetic health check timeout",
                sample_lines=[
                    "health probe failed",
                    "readiness probe timeout",
                    "no customer impact",
                ],
            )
        )

        self.assertIsNotNone(runbook)
        self.assertEqual(runbook.id, "healthcheck_timeout_noise")
        self.assertGreater(score, 0)
        self.assertIn(runbook.default_severity, {"low", "medium"})
        self.assertIn(runbook.disposition, {"NO_ACTION", "OBSERVE"})

    def test_generic_timeout_alone_does_not_overmatch_severe_runbook(self):
        candidates = get_runbook_candidates(
            incident(
                signature="request timeout",
                sample_lines=["operation failed"],
            ),
            limit=3,
        )

        db_scores = [
            score
            for runbook, score in candidates
            if runbook.id == "db_connection_pool_exhausted"
        ]
        self.assertTrue(all(score < HIGH_CONFIDENCE_THRESHOLD for score in db_scores))

    def test_candidate_helper_returns_top_candidates_sorted(self):
        candidates = get_runbook_candidates(
            incident(
                signature="payment gateway timeout",
                sample_lines=[
                    "stripe did not respond",
                    "transaction aborted",
                    "vendor api timeout",
                ],
            ),
            limit=3,
        )

        self.assertLessEqual(len(candidates), 3)
        self.assertEqual(candidates, sorted(candidates, key=lambda item: item[1], reverse=True))
        self.assertTrue(all(score > 0 for _, score in candidates))


class RunbookTieBreakerTests(unittest.TestCase):
    def setUp(self):
        self.db = Runbook(
            {
                "id": "db_connection_pool_exhausted",
                "name": "DB Connection Pool Exhausted",
                "description": "Database connections are exhausted",
                "patterns": ["database connection pool exhausted"],
                "steps": ["Check pool utilization", "Review queries", "Escalate to owner"],
            }
        )
        self.payment = Runbook(
            {
                "id": "payment_gateway",
                "name": "Payment Gateway Timeout",
                "description": "Payment gateway is timing out",
                "patterns": ["payment gateway timeout"],
                "steps": ["Check provider", "Review retries", "Escalate to owner"],
            }
        )

    def test_deterministic_high_score_wins_without_llm(self):
        candidates = [(self.db, 0.90), (self.payment, 0.55)]

        with patch.object(runbook_matcher, "_call_llm_tiebreaker") as call:
            result = select_runbook_for_incident(incident(), candidates=candidates)

        self.assertEqual(result.runbook.id, "db_connection_pool_exhausted")
        self.assertEqual(result.source, "deterministic")
        call.assert_not_called()

    def test_borderline_match_uses_llm_tiebreaker(self):
        candidates = [(self.db, 0.62), (self.payment, 0.58)]
        choice = {
            "selected_runbook_id": "payment_gateway",
            "confidence": 0.72,
            "reason": "Payment evidence is more specific",
        }

        with patch.object(runbook_matcher, "_call_llm_tiebreaker", return_value=choice):
            result = select_runbook_for_incident(incident(), candidates=candidates)

        self.assertEqual(result.runbook.id, "payment_gateway")
        self.assertEqual(result.source, "llm_tiebreaker")
        self.assertIn(result.runbook.id, result.candidate_runbooks)

    def test_invalid_llm_runbook_id_is_rejected(self):
        candidates = [(self.db, 0.62), (self.payment, 0.58)]
        choice = {
            "selected_runbook_id": "made_up_runbook",
            "confidence": 0.90,
            "reason": "Invalid invented runbook",
        }

        with patch.object(runbook_matcher, "_call_llm_tiebreaker", return_value=choice):
            result = select_runbook_for_incident(incident(), candidates=candidates)

        self.assertIsNone(result.runbook)
        self.assertEqual(result.source, "none")

    def test_malformed_llm_json_falls_back_safely(self):
        candidates = [(self.db, 0.62), (self.payment, 0.58)]

        with patch.object(runbook_matcher, "_call_llm_tiebreaker", return_value=None):
            result = select_runbook_for_incident(incident(), candidates=candidates)

        self.assertIsNone(result.runbook)
        self.assertEqual(result.source, "none")

    def test_llm_failure_does_not_break_deterministic_metadata(self):
        candidates = [(self.db, 0.62), (self.payment, 0.58)]

        with patch.object(runbook_matcher, "_call_llm_tiebreaker", return_value=None):
            result = select_runbook_for_incident(incident(), candidates=candidates)

        self.assertIsNone(result.runbook)
        self.assertEqual(
            result.candidate_runbooks,
            ["db_connection_pool_exhausted", "payment_gateway"],
        )


if __name__ == "__main__":
    unittest.main()
