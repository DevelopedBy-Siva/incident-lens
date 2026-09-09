#!/usr/bin/env python3
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_log_server_module():
    module_path = REPO_ROOT / "log-server" / "server.py"
    spec = importlib.util.spec_from_file_location("incidentlens_log_server", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _make_incident(source: str, message: str):
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return SimpleNamespace(
        source=source,
        sample_lines=[f"[{ts}] ERROR: {message}"],
        count=1,
    )


def main():
    from app.core.runbook_matcher import match_runbook

    log_server = _load_log_server_module()
    uncovered = []

    generator_samples = []
    for generator in log_server.ERROR_GENERATORS:
        name = generator.__name__
        try:
            sample = generator()
        except Exception as exc:
            uncovered.append((f"generator:{name}", f"failed to generate sample: {exc}"))
            continue
        generator_samples.append((f"generator:{name}", sample))

    scenario_samples = []
    for scenario_name, scenario in log_server.SCENARIOS.items():
        for idx, step in enumerate(scenario["steps"], start=1):
            if step.get("level", "ERROR").upper() in {"ERROR", "WARN", "WARNING", "CRITICAL"}:
                if "log" in step:
                    sample = step["log"]
                else:
                    sample = log_server.format_scenario_log(step, scenario_name, 1, idx)
                scenario_samples.append(
                    (f"scenario:{scenario_name}:step:{idx}", sample)
                )

    samples = generator_samples + scenario_samples
    covered = 0
    for name, sample in samples:
        runbook, score = match_runbook(_make_incident(name, sample))
        if runbook is None:
            uncovered.append((name, sample))
        else:
            covered += 1

    print("Runbook Coverage")
    print(f"samples checked: {len(samples)}")
    print(f"covered: {covered}")
    print(f"uncovered: {len(uncovered)}")
    if uncovered:
        print("uncovered samples:")
        for name, sample in uncovered[:20]:
            print(f"- {name}: {sample}")


if __name__ == "__main__":
    main()
