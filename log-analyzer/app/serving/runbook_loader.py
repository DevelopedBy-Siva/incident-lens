import yaml
from pathlib import Path
from typing import List, Dict, Optional

RUNBOOKS_DIR = Path(__file__).resolve().parents[2] / "runbooks"


class Runbook:
    """Parsed runbook object"""

    def __init__(self, data: dict):
        self.id = data["id"]
        self.name = data["name"]
        self.description = data.get("description", "")
        self.default_severity = data.get("default_severity", "medium")
        self.disposition = data.get("disposition", "OBSERVE")
        self.patterns = data.get("patterns", [])
        self.steps = data.get("steps", [])
        self.routing = data.get("routing", {})
        self.observe_threshold = data.get("observe_threshold")
        self.cooldown_minutes = data.get("cooldown_minutes", 15)

    def __repr__(self):
        return f"<Runbook {self.id}: {self.name}>"


def load_runbooks() -> List[Runbook]:
    """Load all YAML runbooks from the runbooks directory"""
    runbooks = []

    if not RUNBOOKS_DIR.exists():
        print(f"Runbooks directory not found: {RUNBOOKS_DIR}")
        return runbooks

    for yaml_file in RUNBOOKS_DIR.glob("*.yaml"):
        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)
                runbook = Runbook(data)
                runbooks.append(runbook)
        except Exception as e:
            print(f"Failed to load runbook {yaml_file}: {e}")

    print(f"Loaded {len(runbooks)} runbooks")
    return runbooks


_runbooks_cache: Optional[List[Runbook]] = None


def get_runbooks() -> List[Runbook]:
    """Get cached runbooks (singleton pattern)"""
    global _runbooks_cache
    if _runbooks_cache is None:
        _runbooks_cache = load_runbooks()
    return _runbooks_cache


def reload_runbooks():
    """Force reload of runbooks from disk"""
    global _runbooks_cache
    _runbooks_cache = load_runbooks()
