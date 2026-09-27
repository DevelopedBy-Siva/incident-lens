#!/usr/bin/env python
"""
Diagnostic script to understand why "Agent analyzed: 0" and identify ALL issues
preventing the incident analysis system from working properly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy import text
from app.shared.database import SessionLocal, init_db
from app.data.models import Incident
from app.serving.models import Analysis
from app.control.models import Project

def main():
    print("=" * 80)
    print("INCIDENTLENS ANALYSIS SYSTEM DIAGNOSTIC")
    print("=" * 80)
    print()
    
    # Initialize database
    try:
        init_db()
        db = SessionLocal()
        print("✓ Database connection successful")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return 1
    
    try:
        # Check projects
        print("\n" + "="*80)
        print("PROJECTS")
        print("="*80)
        projects = db.query(Project).all()
        print(f"Total projects: {len(projects)}")
        
        for project in projects:
            print(f"\nProject: {project.name} (ID: {project.id})")
            print(f"  Active: {project.is_active}")
            print(f"  Test project: {project.is_test}")
            print(f"  Active artifact: {project.active_artifact_id or 'None (using base model only)'}")
            print(f"  Datadog configured: {bool(project.datadog_api_key and project.datadog_site and project.datadog_service)}")
            if project.datadog_api_key:
                print(f"    - API key: {'*' * 10}{project.datadog_api_key[-4:] if len(project.datadog_api_key) > 4 else '****'}")
                print(f"    - Site: {project.datadog_site}")
                print(f"    - Service: {project.datadog_service}")
                print(f"    - Query: {project.datadog_query or '(not set)'}")
        
        # Check incidents
        print("\n" + "="*80)
        print("INCIDENTS")
        print("="*80)
        incidents = db.query(Incident).all()
        print(f"Total incidents: {len(incidents)}")
        
        status_counts = {}
        for incident in incidents:
            status_counts[incident.status] = status_counts.get(incident.status, 0) + 1
        
        print(f"By status:")
        for status, count in sorted(status_counts.items()):
            print(f"  {status}: {count}")
        
        # Check analyses
        print("\n" + "="*80)
        print("ANALYSES")
        print("="*80)
        analyses = db.query(Analysis).all()
        print(f"Total analyses: {len(analyses)}")
        
        if analyses:
            severity_counts = {}
            disposition_counts = {}
            for analysis in analyses:
                severity_counts[analysis.severity] = severity_counts.get(analysis.severity, 0) + 1
                disposition_counts[analysis.disposition] = disposition_counts.get(analysis.disposition, 0) + 1
            
            print(f"\nBy severity:")
            for severity, count in sorted(severity_counts.items()):
                print(f"  {severity}: {count}")
            
            print(f"\nBy disposition:")
            for disposition, count in sorted(disposition_counts.items()):
                print(f"  {disposition}: {count}")
        
        # Check which incidents have analyses
        print("\n" + "="*80)
        print("INCIDENT → ANALYSIS MAPPING")
        print("="*80)
        incidents_with_analysis = set(a.incident_id for a in analyses)
        all_incident_ids = set(i.id for i in incidents)
        unanalyzed_incidents = all_incident_ids - incidents_with_analysis
        
        print(f"Incidents with analysis: {len(incidents_with_analysis)}")
        print(f"Incidents without analysis: {len(unanalyzed_incidents)}")
        
        if unanalyzed_incidents:
            print(f"\nSample unanalyzed incidents (first 5):")
            for incident_id in list(unanalyzed_incidents)[:5]:
                incident = db.query(Incident).filter(Incident.id == incident_id).first()
                if incident:
                    print(f"  {incident.id[:20]}... - {incident.source} - {incident.signature[:50]}")
                    print(f"    Created: {incident.first_seen}, Count: {incident.count}, Status: {incident.status}")
        
        # Check model configuration
        print("\n" + "="*80)
        print("MODEL CONFIGURATION")
        print("="*80)
        try:
            from app.shared.model_config import configured_base_model, configured_runtime_settings
            base_model = configured_base_model()
            settings = configured_runtime_settings()
            print(f"Base model: {base_model}")
            print(f"Device: {settings.device}")
            print(f"Provider: {os.getenv('MODEL_PROVIDER', 'not set')}")
        except Exception as e:
            print(f"✗ Failed to load model configuration: {e}")
        
        # Check validate_analysis function
        print("\n" + "="*80)
        print("VALIDATE_ANALYSIS FUNCTION")
        print("="*80)
        try:
            from app.serving.decision_engine import validate_analysis
            print("✓ validate_analysis function exists and can be imported")
        except ImportError as e:
            print(f"✗ validate_analysis import failed: {e}")
            print("  This will cause investigator.py to crash!")
        
        # Check investigation loop
        print("\n" + "="*80)
        print("INVESTIGATION LOOP")
        print("="*80)
        try:
            from app.serving.investigator import get_investigation_loop
            loop = get_investigation_loop()
            print(f"✓ Investigation loop can be instantiated")
            print(f"  Max iterations: {getattr(loop, 'MAX_ITERATIONS', 'unknown')}")
        except Exception as e:
            print(f"✗ Investigation loop instantiation failed: {e}")
        
        # Final diagnosis
        print("\n" + "="*80)
        print("DIAGNOSIS")
        print("="*80)
        
        issues = []
        
        if not projects:
            issues.append("NO PROJECTS: No projects configured in the database")
        
        if not incidents:
            issues.append("NO INCIDENTS: No incidents in database - logs may not be ingesting")
        
        if incidents and not analyses:
            issues.append("NO ANALYSES: Incidents exist but none have been analyzed - investigation loop is failing")
        
        if incidents and analyses and len(analyses) < len(incidents):
            percent_analyzed = (len(analyses) / len(incidents)) * 100
            issues.append(f"PARTIAL ANALYSIS: Only {percent_analyzed:.1f}% of incidents have analyses")
        
        active_projects = [p for p in projects if p.is_active]
        if not active_projects:
            issues.append("NO ACTIVE PROJECTS: All projects are inactive")
        
        configured_projects = [p for p in projects if p.datadog_api_key and p.datadog_site and p.datadog_service]
        if not configured_projects:
            issues.append("NO CONFIGURED DATADOG: No projects have Datadog credentials configured")
        
        if issues:
            print("ISSUES FOUND:\n")
            for i, issue in enumerate(issues, 1):
                print(f"{i}. {issue}")
        else:
            print("✓ System appears to be functioning normally")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        if not projects:
            print("1. Create a project via the frontend or API")
        elif not configured_projects:
            print("1. Configure Datadog credentials for your project in Settings")
        elif not incidents:
            print("1. Ensure the log_source_watcher worker is running")
            print("2. Check that Datadog is returning logs for the configured query")
            print("3. Check POLL_INTERVAL and DATADOG_LOOKBACK_SECONDS in .env")
        elif not analyses:
            print("1. Check logs for '[WORKER] analyze_incident failed' errors")
            print("2. Verify model runtime can initialize (BASE_MODEL in .env)")
            print("3. Check that validate_analysis() exists in decision_engine.py")
            print("4. Test manually: python -c 'from app.serving.decision_engine import validate_analysis'")
        
        return 0
    
    finally:
        db.close()

if __name__ == "__main__":
    sys.exit(main())
