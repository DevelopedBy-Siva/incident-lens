"""Serving-plane incident analysis orchestration and audit trail."""

from datetime import datetime
import logging

from app.data.incidents import (
    apply_root_cause_chain,
    fetch_recent_root_cause_candidates,
    stamp_actioned,
)

logger = logging.getLogger(__name__)


def _notification_allowed(disposition: str, policy) -> bool:
    allowed = set(getattr(policy, "allowed_actions", []) or [])
    if disposition == "ESCALATE":
        return "send_discord_notification" in allowed or "notify_oncall" in allowed
    if disposition == "NEEDS_ONCALL":
        return "send_email_notification" in allowed or "notify_oncall" in allowed
    if disposition == "NEEDS_DEV":
        return "send_discord_notification" in allowed
    return False


# ---------------------------------------------------------------------------
# InvestigationRun writer — audit trail for /investigation endpoint
# ---------------------------------------------------------------------------


def _write_investigation_run(
    incident,
    project,
    evidence,
    tool_calls: list,
    iterations: int,
    fallback_used: bool,
    analysis_source: str,
    analysis,
    policy,
    actions: list,
    started_at,
):
    """Persist the full agent run for inspection via /api/incidents/{id}/investigation."""
    try:
        from app.serving.models import InvestigationRun
        from app.shared.database import SessionLocal

        db = SessionLocal()
        try:
            run = InvestigationRun(
                incident_id=incident.id,
                project_id=project.id,
                started_at=started_at,
                finished_at=datetime.utcnow(),
                evidence_samples=len(evidence.sample_lines) if evidence else 0,
                evidence_related_count=(
                    len(evidence.related_incidents) if evidence else 0
                ),
                evidence_runbook=evidence.runbook_name if evidence else None,
                evidence_snapshot=evidence.as_prompt_context() if evidence else None,
                tool_calls=tool_calls,
                iterations=iterations,
                fallback_used=fallback_used,
                analysis_source=analysis_source,
                policy_allowed=policy.allow if policy else None,
                policy_reason=policy.reason if policy else None,
                policy_tags=policy.tags if policy else [],
                effective_disposition=policy.effective_disposition if policy else None,
                actions_taken=actions,
                verifier_outcome="pending",
                final_severity=analysis.severity if analysis else None,
                final_disposition=analysis.disposition if analysis else None,
                final_confidence=analysis.confidence if analysis else None,
                final_summary=analysis.summary if analysis else None,
            )
            db.add(run)
            db.commit()
            logger.info("[TASKS] InvestigationRun written for %s", incident.id)
        except Exception as e:
            db.rollback()
            logger.warning(
                "[TASKS] InvestigationRun write failed for %s: %s", incident.id, e
            )
        finally:
            db.close()
    except ImportError:
        logger.debug("[TASKS] InvestigationRun model not available yet")


# ---------------------------------------------------------------------------
# Core: analyze_incident
# ---------------------------------------------------------------------------


def analyze_incident(incident, project, force=False):
    from app.serving.models import Analysis
    from app.shared.database import SessionLocal
    from app.serving.runbook_matcher import match_runbook, should_escalate
    from app.serving.evidence import build_serving_evidence
    from app.serving.investigator import get_investigation_loop
    from app.serving.policy import evaluate as policy_eval
    from app.serving.action_executor import execute_actions
    from app.serving.notifications import get_notification_service

    db = SessionLocal()
    try:
        investigation_started_at = datetime.utcnow()
        existing = (
            db.query(Analysis).filter(Analysis.incident_id == incident.id).first()
        )
        if existing and not force:
            return None

        notification_service = get_notification_service(project=project)

        # Phase 1 — evidence
        evidence = build_serving_evidence(incident, project)

        # Tracking vars for InvestigationRun
        tool_calls_log = []
        iterations_log = 0
        fallback_used = False
        analysis_source_log = "unknown"

        # Runbook fast-path
        runbook, score = match_runbook(incident)
        use_runbook = runbook and score >= 0.5

        if use_runbook:
            disposition = runbook.disposition
            if disposition == "OBSERVE" and should_escalate(incident, runbook):
                disposition = runbook.observe_threshold.get("escalate_to", "ESCALATE")

            new_severity = runbook.default_severity
            new_disposition = disposition
            new_confidence = score
            new_summary = f"{runbook.name}: {runbook.description}"
            new_next_steps = runbook.steps
            new_ticket_title = runbook.name
            new_ticket_body = "\n".join(runbook.steps)
            new_source = "runbook"
            matched_runbook_id = runbook.id
            runbook_match_score = score
            analysis_source_log = "runbook"

        else:
            # Phase 2 — investigation loop
            loop = get_investigation_loop()

            llm_analysis = loop.investigate(
                incident, project=project, evidence=evidence
            )

            # Best-effort extraction of loop metadata
            try:
                tool_calls_log = getattr(loop, "_last_tool_calls", [])
                iterations_log = getattr(loop, "_last_iterations", 0)
                fallback_used = getattr(loop, "_last_fallback", False)
            except Exception:
                pass

            if not llm_analysis:
                logger.warning(
                    "[WORKER] Investigation returned None for %s", incident.id
                )
                return None

            new_severity = llm_analysis.severity
            new_disposition = llm_analysis.disposition
            new_confidence = llm_analysis.confidence
            new_summary = llm_analysis.summary
            new_next_steps = llm_analysis.next_steps
            new_ticket_title = llm_analysis.ticket_title
            new_ticket_body = llm_analysis.ticket_body
            new_source = "llm"
            matched_runbook_id = None
            runbook_match_score = None
            analysis_source_log = "llm"

        # Persist analysis
        if existing and force:
            existing.severity = new_severity
            existing.disposition = new_disposition
            existing.confidence = new_confidence
            existing.summary = new_summary
            existing.next_steps = new_next_steps
            existing.ticket_title = new_ticket_title
            existing.ticket_body = new_ticket_body
            existing.analysis_source = new_source
            existing.created_at = datetime.utcnow()
            db.commit()
            db.refresh(existing)
            analysis = existing
        else:
            analysis = Analysis(
                incident_id=incident.id,
                severity=new_severity,
                disposition=new_disposition,
                confidence=new_confidence,
                summary=new_summary,
                next_steps=new_next_steps,
                ticket_title=new_ticket_title,
                ticket_body=new_ticket_body,
                analysis_source=new_source,
                matched_runbook_id=matched_runbook_id,
                runbook_match_score=runbook_match_score,
            )
            db.add(analysis)
            db.commit()
            db.refresh(analysis)

        # Phase 3 — policy gate
        policy = policy_eval(incident, analysis)

        if not policy.allow:
            logger.info("[POLICY] Blocked %s — %s", incident.id, policy.reason)
            actions = execute_actions(incident, analysis, policy, project)
            _write_investigation_run(
                incident,
                project,
                evidence,
                tool_calls_log,
                iterations_log,
                fallback_used,
                analysis_source_log,
                analysis,
                policy,
                actions,
                investigation_started_at,
            )
            return analysis

        # Apply effective disposition
        if (
            policy.effective_disposition
            and policy.effective_disposition != analysis.disposition
        ):

            class _Effective:
                def __init__(self, base, disp):
                    self._base = base
                    self.disposition = disp

                def __getattr__(self, n):
                    return getattr(self._base, n)

            effective_analysis = _Effective(analysis, policy.effective_disposition)
        else:
            effective_analysis = analysis

        # Notify only when the action-level policy allows the notification.
        if _notification_allowed(effective_analysis.disposition, policy):
            notification_service.route_notification(incident, effective_analysis)
            stamp_actioned(incident.id)

        # Phase 4 — actions
        actions = execute_actions(incident, analysis, policy, project)

        # Write audit trail
        _write_investigation_run(
            incident,
            project,
            evidence,
            tool_calls_log,
            iterations_log,
            fallback_used,
            analysis_source_log,
            analysis,
            policy,
            actions,
            investigation_started_at,
        )

        print(
            f"[WORKER] {incident.id} — {analysis.severity}/{analysis.disposition} "
            f"effective={effective_analysis.disposition} actions={actions}"
        )
        return analysis

    except Exception as e:
        logger.error("[WORKER] analyze_incident failed for %s: %s", incident.id, e)
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Root cause chaining (unchanged)
# ---------------------------------------------------------------------------


def run_root_cause_chaining(new_incident, project_id, project):
    from app.serving.decision_engine import get_decision_engine

    try:
        candidates = fetch_recent_root_cause_candidates(project_id, new_incident.id)
        if not candidates:
            return
        print(f"[CHAIN] {new_incident.id} vs {len(candidates)} candidate(s)")
        engine = get_decision_engine()
        result = engine.chain_root_cause(new_incident, candidates, project=project)
        if result and result.has_cause and result.cause_incident_id:
            apply_root_cause_chain(
                new_incident.id,
                result.cause_incident_id,
                result.cause_explanation or "",
            )
        else:
            print(f"[CHAIN] No link for {new_incident.id}")
    except Exception as e:
        print(f"[CHAIN] Failed: {e}")
