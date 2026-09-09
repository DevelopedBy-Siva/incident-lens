import json
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple
from app.core.runbook_loader import Runbook, get_runbooks

HIGH_CONFIDENCE_THRESHOLD = 0.75
BORDERLINE_THRESHOLD = 0.40
LLM_TIEBREAKER_CONFIDENCE_THRESHOLD = 0.50

GENERIC_PATTERNS = {
    "error",
    "failed",
    "failure",
    "timeout",
    "unavailable",
    "exception",
}


@dataclass
class RunbookSelection:
    runbook: Optional[Runbook]
    score: float
    source: str
    candidate_runbooks: list[str] = field(default_factory=list)
    reason: str = ""


def _incident_text(incident) -> str:
    signature = getattr(incident, "signature", "") or ""
    sample_lines = getattr(incident, "sample_lines", None) or []
    return " ".join([signature, *sample_lines]).strip()


def _pattern_weight(pattern: str) -> float:
    if pattern.startswith("regex:"):
        return 0.45

    normalized = pattern.lower().strip()
    if normalized in GENERIC_PATTERNS:
        return 0.0

    token_count = len(re.findall(r"[a-z0-9]+", normalized))
    if token_count >= 3:
        return 0.45
    if token_count == 2:
        return 0.35
    return 0.20


def _match_count(runbook: Runbook, incident_text: str) -> int:
    incident_lower = incident_text.lower()
    matches = 0

    for pattern in runbook.patterns:
        if pattern.startswith("regex:"):
            regex_pattern = pattern[6:]
            if re.search(regex_pattern, incident_lower, re.IGNORECASE):
                matches += 1
        else:
            if pattern.lower() in incident_lower:
                matches += 1

    return matches


def _match_weight(runbook: Runbook, incident_text: str) -> float:
    incident_lower = incident_text.lower()
    score = 0.0

    for pattern in runbook.patterns:
        weight = _pattern_weight(pattern)
        if weight == 0.0:
            continue

        if pattern.startswith("regex:"):
            regex_pattern = pattern[6:]
            if re.search(regex_pattern, incident_lower, re.IGNORECASE):
                score += weight
        else:
            if pattern.lower() in incident_lower:
                score += weight

    return score


def score_runbook(runbook: Runbook, incident_text: str) -> float:
    """
    Score how well a runbook matches an incident.
    Returns a score between 0.0 and 1.0
    """
    if not runbook.patterns:
        return 0.0

    weighted_score = _match_weight(runbook, incident_text)
    return min(1.0, weighted_score)


def get_runbook_candidates(incident, limit: int = 3) -> list[tuple[Runbook, float]]:
    """
    Return top matching runbooks sorted by descending deterministic score.
    Zero-score candidates are omitted.
    """
    runbooks = get_runbooks()
    if not runbooks or limit <= 0:
        return []

    incident_text = _incident_text(incident)
    scored_runbooks = [
        (runbook, score_runbook(runbook, incident_text)) for runbook in runbooks
    ]
    scored_runbooks.sort(key=lambda x: x[1], reverse=True)

    return [(runbook, score) for runbook, score in scored_runbooks if score > 0][:limit]


def match_runbook(incident) -> Tuple[Optional[Runbook], float]:
    """
    Find the best matching runbook for an incident.

    Returns:
        (matched_runbook, confidence_score) or (None, 0.0)
    """
    candidates = get_runbook_candidates(incident, limit=1)
    if not candidates:
        return None, 0.0

    best_runbook, best_score = candidates[0]

    if best_score <= 0:
        return None, 0.0
    if best_score < 0.3:
        return None, 0.0

    return best_runbook, best_score


def select_runbook_for_incident(
    incident, evidence=None, candidates=None, project=None
) -> RunbookSelection:
    """
    Select a runbook while keeping deterministic matching as the primary path.
    The LLM, when available, may only choose from deterministic candidate IDs.
    """
    candidate_pairs = candidates or get_runbook_candidates(incident, limit=3)
    candidate_pairs = [(rb, score) for rb, score in candidate_pairs if score > 0]
    candidate_ids = [runbook.id for runbook, _ in candidate_pairs]

    if not candidate_pairs:
        return RunbookSelection(None, 0.0, "none", [], "No deterministic candidates")

    best_runbook, best_score = candidate_pairs[0]
    if best_score >= HIGH_CONFIDENCE_THRESHOLD:
        return RunbookSelection(
            best_runbook,
            best_score,
            "deterministic",
            candidate_ids,
            "Deterministic score met high-confidence threshold",
        )

    if best_score < BORDERLINE_THRESHOLD and len(candidate_pairs) < 2:
        return RunbookSelection(
            None,
            best_score,
            "none",
            candidate_ids,
            "Deterministic score below tie-breaker threshold",
        )

    llm_choice = _call_llm_tiebreaker(incident, evidence, candidate_pairs, project)
    if not llm_choice:
        return RunbookSelection(
            None,
            best_score,
            "none",
            candidate_ids,
            "LLM tie-breaker unavailable or invalid",
        )

    selected_id = llm_choice.get("selected_runbook_id")
    confidence = llm_choice.get("confidence", 0.0)
    if selected_id not in candidate_ids or confidence < LLM_TIEBREAKER_CONFIDENCE_THRESHOLD:
        return RunbookSelection(
            None,
            best_score,
            "none",
            candidate_ids,
            "LLM selected no valid high-confidence candidate",
        )

    by_id = {runbook.id: (runbook, score) for runbook, score in candidate_pairs}
    selected_runbook, selected_score = by_id[selected_id]
    return RunbookSelection(
        selected_runbook,
        selected_score,
        "llm_tiebreaker",
        candidate_ids,
        str(llm_choice.get("reason", ""))[:300],
    )


def _call_llm_tiebreaker(incident, evidence, candidates, project=None) -> Optional[dict]:
    try:
        from app.core.decision_engine import _make_llm

        llm = _make_llm(project=project)
        if llm is None:
            return None

        candidate_payload = [
            {
                "id": runbook.id,
                "name": runbook.name,
                "description": runbook.description,
                "score": round(score, 2),
            }
            for runbook, score in candidates
        ]
        allowed_ids = [item["id"] for item in candidate_payload]
        evidence_text = (
            evidence.as_prompt_context()
            if evidence and hasattr(evidence, "as_prompt_context")
            else _incident_text(incident)
        )
        prompt = f"""Select the best runbook for this incident.

You may choose only one of these IDs, or null: {allowed_ids}
Candidate runbooks:
{json.dumps(candidate_payload, indent=2)}

Incident evidence:
{evidence_text[:3000]}

Return strict JSON only:
{{
  "selected_runbook_id": "one candidate id or null",
  "confidence": 0.0,
  "reason": "short reason"
}}"""
        response = llm.invoke(prompt)
        content = getattr(response, "content", response)
        parsed = json.loads(str(content))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def should_escalate(incident, runbook: Runbook) -> bool:
    """
    Check if incident should be escalated based on runbook threshold.
    """
    if not runbook.observe_threshold:
        return False

    threshold = runbook.observe_threshold
    required_count = threshold.get("count", 10)

    return incident.count >= required_count
