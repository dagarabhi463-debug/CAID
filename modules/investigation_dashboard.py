"""
Threat Investigation & Incident Reports dashboard for CAID Milestone 2.2.

Turns Milestone 2.1 cross-session correlation findings into grouped,
persisted incidents that analysts can triage, annotate, merge, split,
and export as PDF/CSV reports. Retrospective only — incidents never
modify per-event scores or detection channels.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from modules.correlation_dashboard import analyze_correlation, _evidence_json
from modules.scenario_data import (
    load_scenario_frame,
    render_scenario_selector,
    scenario_label,
)
from core.incident_manager import (
    IncidentConfig,
    IncidentManager,
    IncidentStore,
)
from core.incident_report import (
    generate_incident_csv,
    generate_incident_pdf,
)

def _manager() -> IncidentManager:
    """Return the session-scoped IncidentManager (JSON-backed store)."""
    if 'incident_manager' not in st.session_state:
        cfg = IncidentConfig()
        st.session_state.incident_manager = IncidentManager(
            config=cfg, store=IncidentStore(cfg.store_path))
    return st.session_state.incident_manager


def _ensure_analysis() -> Optional[Dict]:
    """Return the cached correlation result, or None if not loaded."""
    if 'investigation' not in st.session_state:
        st.session_state.investigation = None
    return st.session_state.investigation


def _finding_label(f: Dict) -> str:
    ents = ', '.join(str(e) for e in (f.get('entities') or [])[:3])
    return (f"{f.get('correlation_type', '?')} — {ents} "
            f"({float(f.get('correlation_score') or 0):.2f})")


def _linked_entities(inc: Dict,
                     scored: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Per-user and per-IP detail rows for an incident's entities."""
    users = list(inc.get('entity_users') or [])
    ips = list(inc.get('entity_ips') or [])
    user_rows, ip_rows = [], []
    ev = scored if scored is not None and len(scored) else pd.DataFrame()
    for u in users:
        sub = ev[ev['user_id'].astype(str) == str(u)] if len(ev) and \
            'user_id' in ev.columns else pd.DataFrame()
        user_rows.append({
            'user_id': u,
            'events': int(len(sub)),
            'sessions': int(sub['session_id'].nunique())
            if len(sub) and 'session_id' in sub.columns else 0,
            'max_score': round(float(sub['combined_score'].max()), 3)
            if len(sub) and 'combined_score' in sub.columns
            and sub['combined_score'].notna().any() else None,
            'risk_levels': ', '.join(sorted({
                str(x) for x in sub['risk_level'].dropna().unique()
                })) if len(sub) and 'risk_level' in sub.columns else '',
        })
    for ip in ips:
        sub = ev[ev['source_ip'].astype(str) == str(ip)] \
            if len(ev) and 'source_ip' in ev.columns else pd.DataFrame()
        ip_rows.append({
            'ip': ip,
            'users': int(sub['user_id'].nunique())
            if len(sub) and 'user_id' in sub.columns else 0,
            'sessions': int(sub['session_id'].nunique())
            if len(sub) and 'session_id' in sub.columns else 0,
            'events': int(len(sub)),
        })
    return {'users': pd.DataFrame(user_rows),
            'ips': pd.DataFrame(ip_rows)}


def _mitre_mappings(inc: Dict) -> List[Dict]:
    """ATT&CK mappings for an incident; [] when MITRE data unavailable."""
    try:
        from core.mitre_attack import MitreAttackStore
        store = MitreAttackStore()
        if not store.ensure_data(allow_download=False):
            return []
        return store.map_incident(inc)
    except Exception:
        return []


def _incident_row(inc: Dict) -> Dict:
    shared_ip_only = bool(inc.get('findings')) and all(
        bool(f.get('shared_ip_only')) for f in inc['findings'])
    return {
        'incident_id': inc.get('incident_id'),
        'title': inc.get('title'),
        'severity': inc.get('severity'),
        'status': inc.get('status'),
        'findings': len(inc.get('findings') or []),
        'users': len(inc.get('entity_users') or []),
        'sessions': len(inc.get('entity_sessions') or []),
        'shared_ip_only': shared_ip_only,
        'updated_at': str(inc.get('updated_at', ''))[:19],
    }


def render_investigation_page() -> None:
    st.subheader("Threat Investigation & Incident Reports (Milestone 2.2)")
    st.caption(
        "Groups cross-session correlation findings into persisted "
        "incidents for triage, annotation, and reporting. Retrospective "
        "analysis only — incidents never modify detection scores. "
        "Shared-IP-only incidents remain low-confidence leads, not proof "
        "of coordination.")

    mgr = _manager()
    if mgr.store.load_error:
        st.warning(f"Incident store could not be loaded "
                   f"({mgr.store.load_error}); starting empty.")

    # ---- Scenario selector + load ---------------------------------------
    scenario_key = render_scenario_selector()
    investigation = _ensure_analysis()
    if investigation is None or \
            investigation.get('scenario_key') != scenario_key:
        if investigation is not None:
            st.info("Scenario changed — reload to analyze the new data.")
        if st.button("Load and Analyze"):
            try:
                raw = load_scenario_frame(scenario_key)
                corr_result = analyze_correlation(raw)
                corr_result['scenario_key'] = scenario_key
                corr_result['scenario_label'] = scenario_label(scenario_key)
                mgr.build_incidents(corr_result['correlations'])
                st.session_state.investigation = corr_result
                st.success("Analysis and incident grouping complete.")
                st.rerun()
            except Exception as e:
                st.error(f"Investigation analysis failed: {e}")
                return
        return

    render_investigation_results(mgr, st.session_state.investigation)


def render_investigation_results(mgr: IncidentManager, result: Dict) -> None:
    """Render the investigation UI from an analysis result dict.

    Shared by the standalone page and the multipage dashboard.
    """
    corr = result['correlations']
    scored = result.get('scored_events')
    st.caption(f"Scenario: **{result.get('scenario_label', result.get('scenario_key', ''))}**")

    incidents = mgr.list_incidents()

    # ---- Summary metrics -----------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Incidents", len(incidents))
    c2.metric("Open",
              sum(1 for i in incidents if i.get('status') == 'Open'))
    c3.metric("Investigating",
              sum(1 for i in incidents if i.get('status') == 'Investigating'))
    c4.metric("Critical",
              sum(1 for i in incidents if i.get('severity') == 'Critical'))
    c5.metric("Shared-IP Only",
              sum(1 for i in incidents
                  if i.get('findings') and all(
                      f.get('shared_ip_only') for f in i['findings'])))

    if not incidents:
        st.info("No incidents. Correlation findings were grouped "
                "automatically; adjust thresholds or analyze new data.")
        _manual_grouping_section(mgr, corr)
        return

    # ---- Filters ---------------------------------------------------------
    f1, f2, f3 = st.columns(3)
    statuses = f1.multiselect(
        "Status", options=list(mgr.config.statuses),
        default=list(mgr.config.statuses), key='inv_status_filter')
    severities = f2.multiselect(
        "Severity", options=sorted(mgr.config.severity_thresholds,
                                   key=lambda k: mgr.config.severity_thresholds[k]),
        default=sorted(mgr.config.severity_thresholds,
                       key=lambda k: mgr.config.severity_thresholds[k]),
        key='inv_sev_filter')
    show_shared_only = f3.checkbox(
        "Shared-IP only incidents", value=False, key='inv_shared_only')

    rows = []
    for inc in incidents:
        if inc.get('status') not in statuses or \
                inc.get('severity') not in severities:
            continue
        row = _incident_row(inc)
        if show_shared_only and not row['shared_ip_only']:
            continue
        rows.append(row)
    inc_df = pd.DataFrame(rows)
    if len(inc_df):
        st.dataframe(inc_df, hide_index=True, width="stretch",
                     alt="Incident dashboard table")
    else:
        st.info("No incidents match the current filters.")

    # ---- Manual grouping --------------------------------------------------
    _manual_grouping_section(mgr, corr)
    _merge_section(mgr, incidents)

    # ---- Incident detail -------------------------------------------------
    st.subheader("Incident Detail")
    labels = {i['incident_id']: f"{i['incident_id']} — {i['title']} "
              f"[{i['severity']}/{i['status']}]" for i in incidents}
    choice = st.selectbox(
        "Select incident", options=list(labels.keys()),
        format_func=lambda k: labels[k], key='inv_select')
    inc = mgr.store.get(choice)
    if inc is None:
        return

    if all(f.get('shared_ip_only') for f in inc.get('findings') or []):
        st.warning("Shared-IP only — insufficient evidence of "
                   "coordination; treat as a low-confidence lead.")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"**Severity:** {inc.get('severity')}  |  "
                    f"**Status:** {inc.get('status')}  |  "
                    f"**Source:** {inc.get('source')}")
        st.markdown(f"**Users:** {', '.join(inc.get('entity_users') or []) or '—'}")
        st.markdown(f"**IPs:** {', '.join(inc.get('entity_ips') or []) or '—'}")
        st.markdown(f"**Sessions:** {len(inc.get('entity_sessions') or [])}")

    # Linked entities: per-user and per-IP detail pulled from scored events.
    with st.expander("Linked users & IPs"):
        linked = _linked_entities(inc, scored)
        lu, li = st.columns(2)
        with lu:
            st.markdown("**Users**")
            if len(linked['users']):
                st.dataframe(linked['users'], hide_index=True,
                             width="stretch", alt="Linked users detail")
            else:
                st.caption("No user detail available.")
        with li:
            st.markdown("**Shared IPs**")
            if len(linked['ips']):
                st.dataframe(linked['ips'], hide_index=True,
                             width="stretch", alt="Linked IPs detail")
            else:
                st.caption("No shared-IP detail available.")

    with d2:
        new_status = st.selectbox(
            "Set status", options=list(mgr.config.statuses),
            index=list(mgr.config.statuses).index(inc.get('status', 'Open'))
            if inc.get('status') in mgr.config.statuses else 0,
            key=f'inv_status_{inc["incident_id"]}')
        if st.button("Update status", key=f'inv_set_status_{inc["incident_id"]}'):
            mgr.set_status(inc['incident_id'], new_status)
            st.rerun()

    # Findings
    st.markdown("**Correlation findings in this incident:**")
    fdf = pd.DataFrame([{
        'correlation_id': f.get('correlation_id'),
        'correlation_type': f.get('correlation_type'),
        'entities': ', '.join(map(str, f.get('entities') or [])),
        'sessions': len(f.get('sessions') or []),
        'correlation_score': f.get('correlation_score'),
        'shared_ip_only': f.get('shared_ip_only'),
        'operating_note': f.get('operating_note'),
    } for f in inc.get('findings') or []])
    if len(fdf):
        st.dataframe(fdf, hide_index=True, width="stretch",
                     alt="Findings in incident")
        for f in inc.get('findings') or []:
            with st.expander(_finding_label(f)):
                st.code(_evidence_json(f.get('evidence', [])), language='json')

    # Split control
    if len(inc.get('findings') or []) > 1:
        with st.expander("Split findings into a new incident"):
            move = st.multiselect(
                "Findings to move",
                options=[f.get('correlation_id') for f in inc['findings']],
                format_func=lambda fid: next(
                    (_finding_label(f) for f in inc['findings']
                     if f.get('correlation_id') == fid), fid),
                key=f'inv_split_{inc["incident_id"]}')
            if st.button("Split selected", key=f'inv_split_btn_{inc["incident_id"]}'):
                new_inc = mgr.split(inc['incident_id'], move)
                if new_inc:
                    st.success(f"Created {new_inc['incident_id']}")
                    st.rerun()
                else:
                    st.warning("Select at least one finding to move "
                               "(the incident must keep at least one).")

    # Notes
    st.markdown("**Analyst notes:**")
    for note in inc.get('notes') or []:
        st.markdown(f"- `{str(note.get('timestamp', ''))[:19]}` "
                    f"**{note.get('author', '')}:** {note.get('text', '')}")
    note_text = st.text_area("Add note", key=f'inv_note_{inc["incident_id"]}',
                             height=68)
    if st.button("Add note", key=f'inv_note_btn_{inc["incident_id"]}'):
        mgr.add_note(inc['incident_id'], note_text)
        st.rerun()

    # Evidence timeline
    st.markdown("**Evidence timeline:**")
    timeline = mgr.evidence_timeline(inc['incident_id'], scored)
    if len(timeline):
        tl_cols = [c for c in ['timestamp', 'user_id', 'session_id',
                               'event_type', 'combined_score', 'risk_level',
                               'finding_ids'] if c in timeline.columns]
        st.dataframe(timeline[tl_cols].head(200), hide_index=True,
                     width="stretch", alt="Incident evidence timeline")
        if 'combined_score' in timeline.columns:
            chart_df = timeline.copy()
            chart_df['_idx'] = range(len(chart_df))
            st.line_chart(chart_df.set_index('_idx')[['combined_score']],
                          alt="Combined score over incident timeline")
    else:
        st.info("No scored events available for this incident's "
                "users/sessions.")

    # MITRE ATT&CK mappings
    mitre_maps = _mitre_mappings(inc)
    if mitre_maps:
        st.markdown("**MITRE ATT&CK mappings:**")
        mdf = pd.DataFrame([{
            'technique_id': m.get('technique_id'),
            'technique_name': m.get('technique_name'),
            'tactic': m.get('tactic'),
            'confidence': m.get('confidence'),
            'caid_source': m.get('caid_source'),
        } for m in mitre_maps])
        st.dataframe(mdf, hide_index=True, width="stretch",
                     alt="MITRE ATT&CK mappings for incident")
        st.caption("Potential mappings require analyst review; they are "
                   "not confirmed attacker behavior.")

    # Exports
    st.markdown("**Export:**")
    e1, e2 = st.columns(2)
    csv_bytes = generate_incident_csv(inc, timeline=timeline,
                                      mitre=mitre_maps)
    pdf_bytes = generate_incident_pdf(inc, timeline=timeline,
                                      mitre=mitre_maps)
    e1.download_button(
        "Download incident report (PDF)", data=pdf_bytes,
        file_name=f"incident_{inc['incident_id']}.pdf",
        mime="application/pdf", width="stretch",
        key=f'inv_pdf_{inc["incident_id"]}')
    e2.download_button(
        "Download incident data (CSV)", data=csv_bytes,
        file_name=f"incident_{inc['incident_id']}.csv",
        mime="text/csv", width="stretch",
        key=f'inv_csv_{inc["incident_id"]}')


def _manual_grouping_section(mgr: IncidentManager,
                             corr: Optional[pd.DataFrame]) -> None:
    """Create an incident manually from selected correlation findings."""
    with st.expander("Create incident from selected findings"):
        if corr is None or len(corr) == 0:
            st.info("No correlation findings available.")
            return
        options = corr['correlation_id'].tolist()
        labels = {r['correlation_id']: _finding_label(r)
                  for _, r in corr.iterrows()}
        selected = st.multiselect(
            "Correlation findings", options=options,
            format_func=lambda fid: labels.get(fid, fid),
            key='inv_manual_findings')
        note = st.text_input("Initial note (optional)", key='inv_manual_note')
        if st.button("Create incident", key='inv_manual_create'):
            inc = mgr.create_from_findings(selected, notes=note)
            if inc:
                st.success(f"Created {inc['incident_id']}")
                st.rerun()
            else:
                st.warning("Select at least one finding.")


def _merge_section(mgr: IncidentManager, incidents: List[Dict]) -> None:
    """Merge two or more incidents into one."""
    if len(incidents) < 2:
        return
    with st.expander("Merge incidents"):
        ids = [i['incident_id'] for i in incidents]
        labels = {i['incident_id']: f"{i['incident_id']} — {i['title']} "
                  f"[{i['severity']}]" for i in incidents}
        selected = st.multiselect(
            "Incidents to merge", options=ids,
            format_func=lambda k: labels.get(k, k),
            key='inv_merge_ids')
        if st.button("Merge selected", key='inv_merge_btn'):
            merged = mgr.merge(selected)
            if merged:
                st.success(f"Merged into {merged['incident_id']}")
                st.rerun()
            else:
                st.warning("Select at least two incidents to merge.")
