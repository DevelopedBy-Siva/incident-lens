"""
worker/tasks.py  —  IncidentLens full pipeline with InvestigationRun audit trail

Pipeline per incident:
  1. cluster_log_db           existing
  2. build_evidence           Phase 1
  3. InvestigationLoop        Phase 2  (tool-calling, falls back gracefully)
  4. policy.evaluate          Phase 3
  5. route_notification       existing
  6. execute_actions          Phase 4
  7. _write_investigation_run  audit trail for /investigation endpoint
"""

from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

CLUSTER_WINDOW_MINUTES = 2
MAX_SAMPLES = 10
ROOT_CAUSE_LOOKBACK_MINUTES = 10


# ---------------------------------------------------------------------------
# Clustering (unchanged)
# ---------------------------------------------------------------------------


def cluster_log_db(project_id, source, environment, parsed_log, signature):
    from app.services.storage import Incident, SessionLocal

    db = SessionLocal()
    try:
        window_start = datetime.utcnow() - timedelta(minutes=CLUSTER_WINDOW_MINUTES)
        incident = (
            db.query(Incident)
            .filter(
                Incident.project_id == project_id,
                Incident.signature == signature,
                Incident.status == "open",
                Incident.last_seen >= window_start,
            )
            .order_by(Incident.last_seen.desc())
            .first()
        )
        if incident:
            incident.count += 1
            incident.last_seen = datetime.utcnow()
            if len(incident.sample_lines or []) < MAX_SAMPLES:
                lines = list(incident.sample_lines or [])
                lines.append(parsed_log.raw)
                incident.sample_lines = lines
            db.commit()
            db.refresh(incident)
            return incident, False

        new_incident = Incident(
            project_id=project_id,
            source=source,
            environment=environment,
            signature=signature,
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow(),
            count=1,
            sample_lines=[parsed_log.raw],
            status="open",
        )
        db.add(new_incident)
        db.commit()
        db.refresh(new_incident)
        return new_incident, True
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Root cause helpers (unchanged)
# ---------------------------------------------------------------------------


def _fetch_recent_root_cause_candidates(project_id, new_incident_id):
    from app.services.storage import Incident, SessionLocal

    cutoff = datetime.utcnow() - timedelta(minutes=ROOT_CAUSE_LOOKBACK_MINUTES)
    db = SessionLocal()
    try:
        candidates = (
            db.query(Incident)
            .filter(
                Incident.project_id == project_id,
                Incident.id != new_incident_id,
                Incident.status == "open",
                Incident.first_seen >= cutoff,
                Incident.root_cause_incident_id == None,
            )
            .order_by(Incident.first_seen.asc())
            .all()
        )
        for c in candidates:
            db.expunge(c)
        return candidates
    finally:
        db.close()


def _apply_root_cause_chain(new_incident_id, cause_incident_id, explanation):
    from app.services.storage import Incident, SessionLocal

    db = SessionLocal()
    try:
        incident = db.query(Incident).filter(Incident.id == new_incident_id).first()
        if incident:
            incident.root_cause_incident_id = cause_incident_id
            incident.cause_explanation = explanation
            db.commit()
            print(f"[CHAIN] {new_incident_id} → {cause_incident_id}")
    except Exception as e:
        db.rollback()
        print(f"[CHAIN] Failed: {e}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Cooldown stamp
# ---------------------------------------------------------------------------


def _stamp_actioned(incident_id: str):
    from app.services.storage import Incident, SessionLocal

    db = SessionLocal()
    try:
        row = db.query(Incident).filter(Incident.id == incident_id).first()
        if row and hasattr(row, "last_actioned_at"):
            row.last_actioned_at = datetime.utcnow()
            db.commit()
    except Exception as e:
        db.rollback()
        logger.warning("[TASKS] stamp_actioned failed for %s: %s", incident_id, e)
    finally:
        db.close()


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
        from app.services.storage import InvestigationRun, SessionLocal

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
    from app.services.storage import Analysis, SessionLocal
    from app.core.runbook_matcher import match_runbook, should_escalate
    from app.core.evidence import build_evidence
    from app.core.investigator import get_investigation_loop
    from app.core.policy import evaluate as policy_eval
    from app.core.action_executor import execute_actions
    from app.services.notifications import get_notification_service

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
        evidence = build_evidence(incident, project)

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
            _stamp_actioned(incident.id)

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
    from app.core.decision_engine import get_decision_engine

    try:
        candidates = _fetch_recent_root_cause_candidates(project_id, new_incident.id)
        if not candidates:
            return
        print(f"[CHAIN] {new_incident.id} vs {len(candidates)} candidate(s)")
        engine = get_decision_engine()
        result = engine.chain_root_cause(new_incident, candidates, project=project)
        if result and result.has_cause and result.cause_incident_id:
            _apply_root_cause_chain(
                new_incident.id,
                result.cause_incident_id,
                result.cause_explanation or "",
            )
        else:
            print(f"[CHAIN] No link for {new_incident.id}")
    except Exception as e:
        print(f"[CHAIN] Failed: {e}")


# ---------------------------------------------------------------------------
# Batch entry point — loki_watcher.py unchanged
# ---------------------------------------------------------------------------


def process_log_batch(payload: dict):
    from app.core.parser import ParsedLog
    from app.core.signatures import generate_signature
    from app.services.storage import SessionLocal, Project

    project_id = payload.get("project_id")
    source = payload["source"]
    environment = payload["environment"]
    logs = payload["logs"]
    project = payload.get("_project")

    if project is None:
        db = SessionLocal()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                db.expunge(project)
            else:
                print(f"[WORKER] Project {project_id} not found")
                return
        finally:
            db.close()

    print(f"[WORKER] {len(logs)} logs for '{project.name}'")
    created = updated = failed = 0

    for log_line in logs:
        try:
            parsed = ParsedLog(log_line)
            if parsed.level not in ["ERROR", "WARN", "WARNING", "CRITICAL"]:
                continue

            sig = generate_signature(source, parsed)
            incident, is_new = cluster_log_db(
                project_id=project_id,
                source=source,
                environment=environment,
                parsed_log=parsed,
                signature=sig,
            )

            if is_new:
                analyze_incident(incident, project=project, force=False)
                run_root_cause_chaining(incident, project_id, project=project)
                created += 1
            else:
                if incident.count in {5, 10, 20}:
                    analyze_incident(incident, project=project, force=True)
                updated += 1

        except Exception as e:
            print(f"[WORKER] Failed: {e} | {log_line[:80]}")
            failed += 1

    print(f"[WORKER] created={created} updated={updated} failed={failed}")
    if failed > 0 and created == 0 and updated == 0:
        raise RuntimeError(f"Batch entirely failed — {failed} errors")

    return {
        "incidents_created": created,
        "incidents_updated": updated,
        "failed": failed,
    }
