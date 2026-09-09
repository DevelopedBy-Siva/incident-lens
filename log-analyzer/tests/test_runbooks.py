import re
import unittest
from pathlib import Path

import yaml


RUNBOOK_DIR = Path(__file__).resolve().parents[1] / "runbooks"
VALID_SEVERITIES = {"low", "medium", "high", "critical"}
VALID_DISPOSITIONS = {"NO_ACTION", "OBSERVE", "NEEDS_DEV", "NEEDS_ONCALL", "ESCALATE"}
UNSAFE_STEP_PHRASES = [
    "automatically restart",
    "auto restart",
    "automatically rollback",
    "automatically deploy",
    "automatically modify",
    "automatically delete",
    "increase database pool automatically",
    "modify database automatically",
]


def load_yaml_runbooks():
    runbooks = []
    for path in sorted(RUNBOOK_DIR.glob("*.yaml")):
        with path.open() as handle:
            runbooks.append((path, yaml.safe_load(handle)))
    return runbooks


class RunbookYamlTests(unittest.TestCase):
    def test_all_yaml_runbooks_load_successfully(self):
        runbooks = load_yaml_runbooks()
        self.assertGreater(len(runbooks), 0)

        for path, data in runbooks:
            with self.subTest(path=path.name):
                self.assertIsInstance(data.get("id"), str)
                self.assertIsInstance(data.get("name"), str)
                self.assertIsInstance(data.get("patterns"), list)
                self.assertGreaterEqual(len(data["patterns"]), 3)
                self.assertIsInstance(data.get("steps"), list)
                self.assertGreaterEqual(len(data["steps"]), 3)
                self.assertIn(data.get("default_severity"), VALID_SEVERITIES)
                self.assertIn(data.get("disposition"), VALID_DISPOSITIONS)

    def test_runbook_ids_are_stable_snake_case(self):
        ids = [data["id"] for _, data in load_yaml_runbooks()]

        self.assertEqual(len(ids), len(set(ids)))
        for runbook_id in ids:
            with self.subTest(runbook_id=runbook_id):
                self.assertRegex(runbook_id, r"^[a-z0-9_]+$")

    def test_runbook_steps_do_not_auto_remediate(self):
        for path, data in load_yaml_runbooks():
            for step in data.get("steps", []):
                step_lower = step.lower()
                with self.subTest(path=path.name, step=step):
                    for phrase in UNSAFE_STEP_PHRASES:
                        self.assertNotIn(phrase, step_lower)


if __name__ == "__main__":
    unittest.main()
