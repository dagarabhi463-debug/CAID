"""
Incident report generation for CAID Milestone 2.2.

Produces PDF (reportlab) and CSV exports for investigation incidents.
Reports carry the full evidence chain: incident metadata, correlation
findings with their evidence dicts, the merged evidence timeline, and
analyst notes. Shared-IP-only findings keep their caveat text.

Synthetic/defensive research data only — no real Canvas connectivity.
"""

from __future__ import annotations

import io
import json
from typing import Dict, List, Optional

import pandas as pd

from modules.export import _sanitize_csv_frame


_DISCLAIMER = (
    'CAID is a defensive research prototype. All data is synthetic; '
    'scores are correlation strengths, not probabilities. A shared IP '
    'alone is never proof of coordination.'
)


def _finding_label(f: Dict) -> str:
    ents = ', '.join(str(e) for e in (f.get('entities') or [])[:4])
    return (f"{f.get('correlation_type', '?')} — {ents} "
            f"(score {float(f.get('correlation_score') or 0):.2f})")


def _incident_findings(incident: Dict) -> List[Dict]:
    return list(incident.get('findings') or [])


def _shared_ip_only(incident: Dict) -> bool:
    findings = _incident_findings(incident)
    return bool(findings) and all(
        bool(f.get('shared_ip_only')) for f in findings)


# ----------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------

def generate_incident_csv(incident: Dict,
                          scored_events: Optional[pd.DataFrame] = None,
                          timeline: Optional[pd.DataFrame] = None,
                          mitre: Optional[List[Dict]] = None,
                          max_timeline_rows: int = 500) -> bytes:
    """Serialize an incident to a sanitized CSV (long/tidy format)."""
    rows: List[Dict] = []

    def add(section: str, item: str, detail=""):
        rows.append({'section': section, 'item': str(item),
                     'detail': '' if detail is None else str(detail)})

    add('incident', 'incident_id', incident.get('incident_id'))
    add('incident', 'title', incident.get('title'))
    add('incident', 'status', incident.get('status'))
    add('incident', 'severity', incident.get('severity'))
    add('incident', 'source', incident.get('source'))
    add('incident', 'created_at', incident.get('created_at'))
    add('incident', 'updated_at', incident.get('updated_at'))
    for u in incident.get('entity_users') or []:
        add('entities', 'user', u)
    for s in incident.get('entity_sessions') or []:
        add('entities', 'session', s)
    for ip in incident.get('entity_ips') or []:
        add('entities', 'ip', ip)

    for f in _incident_findings(incident):
        add('finding', 'correlation_id', f.get('correlation_id'))
        add('finding', 'correlation_type', f.get('correlation_type'))
        add('finding', 'entities', ', '.join(map(str, f.get('entities') or [])))
        add('finding', 'sessions',
            ', '.join(map(str, f.get('sessions') or [])))
        add('finding', 'correlation_score', f.get('correlation_score'))
        add('finding', 'shared_ip_only', f.get('shared_ip_only'))
        add('finding', 'operating_note', f.get('operating_note'))
        for ev in f.get('evidence') or []:
            if isinstance(ev, dict):
                add('evidence', ev.get('signal', '?'),
                    json.dumps(ev, default=str))

    if timeline is None and scored_events is not None:
        # Caller may pass precomputed timeline; otherwise leave empty.
        timeline = pd.DataFrame()
    if timeline is not None and len(timeline):
        tl = timeline.head(max_timeline_rows)
        for _, r in tl.iterrows():
            add('timeline',
                f"{r.get('timestamp', '')} | {r.get('user_id', '')} | "
                f"{r.get('session_id', '')} | {r.get('event_type', '')}",
                f"score={r.get('combined_score', '')} "
                f"risk={r.get('risk_level', '')} "
                f"findings={r.get('finding_ids', '')}")

    for m in mitre or []:
        add('mitre', m.get('technique_id', ''),
            f"{m.get('technique_name', '')} | tactic={m.get('tactic', '')} "
            f"| confidence={m.get('confidence', '')} "
            f"| source={m.get('caid_source', '')} "
            f"| rationale={m.get('rationale', '')}")

    for note in incident.get('notes') or []:
        add('note', f"{note.get('timestamp', '')} [{note.get('author', '')}]",
            note.get('text', ''))

    add('disclaimer', 'notice', _DISCLAIMER)

    return _sanitize_csv_frame(pd.DataFrame(rows)).to_csv(
        index=False).encode('utf-8')


# ----------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------

def generate_incident_pdf(incident: Dict,
                          timeline: Optional[pd.DataFrame] = None,
                          mitre: Optional[List[Dict]] = None,
                          max_timeline_rows: int = 100) -> bytes:
    """Render an incident report to PDF bytes via reportlab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.75 * inch, bottomMargin=0.75 * inch,
                            title=f"CAID Incident {incident.get('incident_id', '')}",
                            author='CAID')
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle('h1c', parent=styles['Heading1'], alignment=1)
    small = ParagraphStyle('small', parent=styles['Normal'],
                           fontSize=8, textColor=colors.grey)

    story = []
    story.append(Paragraph('CAID — Incident Report', h1))
    story.append(Paragraph(_DISCLAIMER, small))
    story.append(Spacer(1, 0.15 * inch))

    meta = [
        ['Incident ID', incident.get('incident_id', '')],
        ['Title', incident.get('title', '')],
        ['Severity', incident.get('severity', '')],
        ['Status', incident.get('status', '')],
        ['Source', incident.get('source', '')],
        ['Created', incident.get('created_at', '')],
        ['Updated', incident.get('updated_at', '')],
        ['Users', ', '.join(incident.get('entity_users') or []) or '—'],
        ['Sessions', str(len(incident.get('entity_sessions') or []))],
        ['IPs', ', '.join(incident.get('entity_ips') or []) or '—'],
    ]
    t = Table(meta, colWidths=[1.4 * inch, 5.0 * inch])
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.15 * inch))

    findings = _incident_findings(incident)
    story.append(Paragraph(
        f'Correlation Findings ({len(findings)})', styles['Heading2']))
    fdata = [['Type', 'Entities', 'Score', 'Shared-IP only', 'Operating note']]
    for f in findings:
        fdata.append([
            str(f.get('correlation_type', '')),
            ', '.join(str(e) for e in (f.get('entities') or [])[:4]),
            f"{float(f.get('correlation_score') or 0):.2f}",
            'yes' if f.get('shared_ip_only') else 'no',
            str(f.get('operating_note') or ''),
        ])
    ft = Table(fdata, colWidths=[1.6 * inch, 1.8 * inch, 0.5 * inch,
                                 0.7 * inch, 1.8 * inch], repeatRows=1)
    ft.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(ft)

    # Evidence detail per finding
    from xml.sax.saxutils import escape as _xml_escape
    for f in findings:
        story.append(Spacer(1, 0.08 * inch))
        story.append(Paragraph(
            _xml_escape(_finding_label(f)), styles['Heading3']))
        for ev in f.get('evidence') or []:
            if not isinstance(ev, dict):
                continue
            obs = _xml_escape(str(ev.get('observed_value', '')))
            exp = _xml_escape(str(ev.get('expected_baseline_or_threshold', '')))
            sig = _xml_escape(str(ev.get('signal', '?')))
            expl = _xml_escape(str(ev.get('explanation', '')))
            txt = (f"<b>{sig}</b>: {obs} "
                   f"(expected {exp}) — {expl}")
            story.append(Paragraph(txt, styles['Normal']))

    if _shared_ip_only(incident):
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(
            '<b>Shared-IP caveat:</b> every finding in this incident rests '
            'on shared-IP evidence only and remains capped/low-confidence '
            'unless independently corroborated.', styles['Normal']))

    if mitre:
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph('MITRE ATT&CK Mapping', styles['Heading2']))
        mdata = [['Technique', 'Name', 'Tactic', 'Confidence', 'CAID source']]
        for m in mitre:
            mdata.append([
                str(m.get('technique_id', '')),
                str(m.get('technique_name', '')),
                str(m.get('tactic', '')),
                str(m.get('confidence', '')),
                str(m.get('caid_source', '')),
            ])
        mt = Table(mdata, colWidths=[0.7 * inch, 1.7 * inch, 1.1 * inch,
                                     0.8 * inch, 2.0 * inch], repeatRows=1)
        mt.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(mt)
        story.append(Paragraph(
            'Potential mappings indicate resemblance only and require '
            'analyst review; they are not confirmed attacker behavior.',
            small))

    if timeline is not None and len(timeline):
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph(
            f'Evidence Timeline ({min(len(timeline), max_timeline_rows)}'
            f'{" of " + str(len(timeline)) if len(timeline) > max_timeline_rows else ""} events)',
            styles['Heading2']))
        tl = timeline.head(max_timeline_rows)
        tdata = [['Timestamp', 'User', 'Session', 'Event', 'Score', 'Risk']]
        for _, r in tl.iterrows():
            tdata.append([
                str(r.get('timestamp', ''))[:19],
                str(r.get('user_id', '')),
                str(r.get('session_id', ''))[:18],
                str(r.get('event_type', '')),
                f"{float(r.get('combined_score') or 0):.2f}"
                if pd.notna(r.get('combined_score')) else '',
                str(r.get('risk_level', '')),
            ])
        tt = Table(tdata, colWidths=[1.3 * inch, 0.9 * inch, 1.1 * inch,
                                     1.4 * inch, 0.5 * inch, 0.6 * inch],
                   repeatRows=1)
        tt.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
        ]))
        story.append(tt)

    notes = incident.get('notes') or []
    if notes:
        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph('Analyst Notes', styles['Heading2']))
        for note in notes:
            story.append(Paragraph(
                f"<b>{note.get('timestamp', '')} [{note.get('author', '')}]</b><br/>"
                f"{note.get('text', '')}", styles['Normal']))

    doc.build(story)
    return buf.getvalue()
