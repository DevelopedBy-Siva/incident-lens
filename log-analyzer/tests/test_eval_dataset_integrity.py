"""Guards labels, provenance, and family metadata for the diagnostic eval suite."""

import json
import unittest
from pathlib import Path


class EvaluationDatasetIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "evals" / "incident_triage_cases.json"
        cls.cases = json.loads(path.read_text(encoding="utf-8"))

    def test_cases_have_valid_labels_and_explicit_synthetic_provenance(self):
        severities = {"low", "medium", "high", "critical"}
        dispositions = {"NO_ACTION", "OBSERVE", "NEEDS_DEV", "NEEDS_ONCALL", "ESCALATE"}
        self.assertEqual(len(self.cases), 210)
        self.assertEqual(len({case["id"] for case in self.cases}), len(self.cases))

        for case in self.cases:
            with self.subTest(case=case["id"]):
                expected = case["expected"]
                self.assertIn(expected["severity"], severities)
                self.assertIn(expected["disposition"], dispositions)
                self.assertTrue(case["logs"])
                self.assertIs(case["synthetic"], True)
                self.assertTrue(case["scenario_family"])


if __name__ == "__main__":
    unittest.main()
