"""
SOC summary report generation for CAID.

Produces a whole-analysis PDF (reportlab) and CSV export summarizing
the currently analyzed scenario: report scope, risk distribution,
calibration status, correlation findings, incidents, and the highest-
risk users, IPs, and events. Sibling to ``core/incident_report.py``,
which covers single incidents; this module covers the full pipeline
output stored in ``modules.analysis_state``.

Synthetic/defensive research data only — no real Canvas connectivity.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from modules.export import _sanitize_csv_frame


_DISCLAIMER = (
    'CAID is a defensive research prototype. All data is synthetic; '
    'scores are calibrated anomaly scores, not probabilities. Correlation '
    'strengths are not proof of coordination, and a shared IP alone is '
    'never treated as evidence of an attack.'
)

_RISK_ORDER = ['Low', 'Medium', 'High', 'Critical']

_TYPE_LABELS = {
    'repeated_pattern_across_sessions': 'Repeated Pattern Across Sessions',
    'similar_account_behavior': 'Similar Account Behavior',
    'repeated_sensitive_operations': 'Repeated Sensitive Operations',
    'long_term_behavior_change': 'Long-Term Behavior Change',
    'shared_ip_activity': 'Shared IP Activity',
}


def _df(analysis: Dict) -> pd.DataFrame:
    df = (analysis or {}).get('scored_events')
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _corr(analysis: Dict) -> pd.DataFrame:
    corr = (analysis or {}).get('correlations')
    return corr if isinstance(corr, pd.DataFrame) else pd.DataFrame()


def _risk_counts(df: pd.DataFrame) -> Dict[str, int]:
    if 'risk_level' not in df.columns or df.empty:
        return {}
    counts = (df['risk_level'].astype(str).str.title()
              .value_counts().to_dict())
    return {level: int(counts.get(level, 0)) for level in _RISK_ORDER
            if counts.get(level)}


def _data_window(df: pd.DataFrame) -> str:
    if 'timestamp' not in df.columns or df.empty:
        return '—'
    ts = pd.to_datetime(df['timestamp'], errors='coerce').dropna()
    if ts.empty:
        return '—'
    return f'{ts.min()} → {ts.max()}'


def _mode(series: Optional[pd.Series]) -> str:
    if series is None:
        return 'unknown'
    mode = series.dropna()
    if mode.empty:
        return 'unknown'
    return str(mode.mode().iloc[0])


def _top_entities(df: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:
    if df.empty or col not in df.columns or 'combined_score' not in df.columns:
        return pd.DataFrame()
    g = (df.groupby(col)['combined_score']
           .agg(mean_score='mean', max_score='max', events='size')
           .sort_values('mean_score', ascending=False).head(n))
    g.index.name = col
    return g.reset_index()


def _top_events(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    if df.empty or 'combined_score' not in df.columns:
        return pd.DataFrame()
    cols = [c for c in ['timestamp', 'user_id', 'session_id', 'event_type',
                        'combined_score', 'risk_level'] if c in df.columns]
    return (df.sort_values('combined_score', ascending=False)
              .head(n)[cols])


def _scenario_label(analysis: Dict) -> str:
    return str((analysis or {}).get('scenario_label')
               or (analysis or {}).get('scenario_key') or 'unknown')


def _generated_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _executive_summary(analysis: Dict,
                       incidents: List[Dict]) -> str:
    """Auto-generated posture narrative for the report header."""
    df, corr = _df(analysis), _corr(analysis)
    metrics = _metrics(analysis, incidents)
    risk = _risk_counts(df)
    high_crit = risk.get('High', 0) + risk.get('Critical', 0)
    high_crit_pct = (high_crit / len(df) * 100) if len(df) else 0.0
    shared_only = 0
    if not corr.empty and 'shared_ip_only' in corr.columns:
        shared_only = int(corr['shared_ip_only']
                          .fillna(False).astype(bool).sum())
    parts = [
        f"This report summarizes the CAID retrospective analysis of "
        f"{_scenario_label(analysis)}: {metrics['events_analyzed']} events "
        f"from {metrics['users']} users across {metrics['sessions']} "
        f"sessions ({_data_window(df)}).",
        f"{high_crit:,} events ({high_crit_pct:.1f}%) were classified High "
        f"or Critical risk; {metrics['ml_anomalies']} were flagged as ML "
        f"anomalies.",
        f"The correlation layer produced {metrics['correlation_findings']} "
        f"findings grouped into {metrics['incidents_total']} incidents "
        f"({metrics['incidents_open']} open, "
        f"{metrics['incidents_critical']} critical).",
    ]
    if shared_only:
        parts.append(
            f"{shared_only} finding(s) rest on shared-IP evidence only and "
            "remain capped/low-confidence pending corroboration.")
    if not incidents and corr.empty:
        parts.append("No cross-session correlation findings or incidents "
                     "were raised for this scenario.")
    return ' '.join(parts)


def report_outline(analysis: Dict,
                   incidents: Optional[List[Dict]] = None
                   ) -> List[Dict[str, str]]:
    """Section headings and one-line contents, for UI preview."""
    incidents = incidents or []
    df, corr = _df(analysis), _corr(analysis)
    risk = _risk_counts(df)
    n_types = (int(corr['correlation_type'].nunique())
               if not corr.empty and 'correlation_type' in corr.columns
               else 0)
    n_entities = sum(int(df[col].nunique()) for col in ('user_id', 'source_ip')
                     if col in df.columns) if not df.empty else 0
    outline = [
        {'section': 'Report details',
         'contents': f"{_scenario_label(analysis)} · {_data_window(df)}"},
        {'section': 'Executive summary',
         'contents': 'Auto-generated posture narrative'},
        {'section': 'Risk distribution',
         'contents': (f"{len(risk)} level(s) across {len(df):,} events"
                      if risk else 'No risk-level data')},
        {'section': 'Key metrics',
         'contents': f"{len(_metrics(analysis, incidents))} metrics"},
        {'section': 'Calibration & methodology',
         'contents': 'Calibration status, threshold version, caveats'},
        {'section': 'Incidents',
         'contents': f"{len(incidents)} incident(s)"},
    ]
    if n_types:
        outline.append({
            'section': 'Correlation findings by type',
            'contents': f"{n_types} type(s), {len(corr)} finding(s)"})
    if n_entities:
        outline.append({'section': 'Top suspicious entities',
                        'contents': 'Top users/IPs by mean combined score'})
    if len(df):
        outline.append({'section': 'Highest-risk events',
                        'contents': f"Top {min(15, len(df))} events by "
                                    "combined score"})
    outline.append({'section': 'Disclaimer',
                    'contents': 'Synthetic/defensive research data notice'})
    return outline


def _metrics(analysis: Dict, incidents: List[Dict]) -> Dict[str, str]:
    """Headline metrics shared by the PDF and CSV exports."""
    df, corr = _df(analysis), _corr(analysis)
    risk = _risk_counts(df)
    anomalies = (int(df['is_anomaly'].fillna(False).astype(bool).sum())
                 if 'is_anomaly' in df.columns and not df.empty else 0)
    high_crit = int(sum(risk.get(l, 0) for l in ('High', 'Critical')))
    shared_ip = corr[corr['correlation_type'] == 'shared_ip_activity'] \
        if not corr.empty and 'correlation_type' in corr.columns \
        else pd.DataFrame()
    metrics = {
        'events_analyzed': f'{len(df):,}',
        'users': f"{df['user_id'].nunique():,}"
                 if 'user_id' in df.columns else '—',
        'sessions': f"{df['session_id'].nunique():,}"
                    if 'session_id' in df.columns else '—',
        'source_ips': f"{df['source_ip'].nunique():,}"
                      if 'source_ip' in df.columns else '—',
        'ml_anomalies': f'{anomalies:,}',
        'high_critical_events': f'{high_crit:,}',
        'correlation_findings': f'{len(corr):,}',
        'shared_ip_findings': f'{len(shared_ip):,}',
        'corroborated_shared_ip':
            f"{int((~shared_ip['shared_ip_only']).sum()):,}"
            if len(shared_ip) and 'shared_ip_only' in shared_ip.columns
            else '0',
        'incidents_total': f'{len(incidents):,}',
        'incidents_open': f"{sum(1 for i in incidents if i.get('status') == 'Open'):,}",
        'incidents_critical':
            f"{sum(1 for i in incidents if i.get('severity') == 'Critical'):,}",
    }
    return metrics


# ----------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------

def generate_soc_summary_csv(analysis: Dict,
                             incidents: Optional[List[Dict]] = None,
                             max_entity_rows: int = 25,
                             max_event_rows: int = 50) -> bytes:
    """Serialize a SOC summary to a sanitized tidy-format CSV."""
    incidents = incidents or []
    df, corr = _df(analysis), _corr(analysis)
    rows: List[Dict] = []

    def add(section: str, item: str, detail=""):
        rows.append({'section': section, 'item': str(item),
                     'detail': '' if detail is None else str(detail)})

    add('report', 'generated_at', _generated_at())
    add('report', 'scenario', _scenario_label(analysis))
    add('report', 'scenario_key', (analysis or {}).get('scenario_key', ''))
    add('report', 'data_window', _data_window(df))

    for level, count in _risk_counts(df).items():
        pct = (count / len(df) * 100) if len(df) else 0.0
        add('risk_distribution', level, f'{count} ({pct:.1f}%)')

    for key, value in _metrics(analysis, incidents).items():
        add('metrics', key, value)

    if not df.empty:
        for col, label in [('risk_calibration_status', 'risk_calibration'),
                           ('detection_calibration_status',
                            'detection_calibration'),
                           ('threshold_version', 'threshold_version')]:
            if col in df.columns:
                add('calibration', label, _mode(df[col]))

    if not corr.empty and 'correlation_type' in corr.columns:
        for ctype, grp in corr.groupby('correlation_type'):
            avg = (grp['correlation_score'].mean()
                   if 'correlation_score' in grp.columns else float('nan'))
            shared_only = (int(grp['shared_ip_only'].sum())
                           if 'shared_ip_only' in grp.columns else 0)
            add('correlation_types', _TYPE_LABELS.get(str(ctype), str(ctype)),
                f'findings={len(grp)}; avg_score={avg:.3f}; '
                f'shared_ip_only={shared_only}')

    for inc in incidents:
        add('incident', inc.get('incident_id', ''),
            f"severity={inc.get('severity', '')}; "
            f"status={inc.get('status', '')}; "
            f"findings={len(inc.get('findings') or [])}; "
            f"title={inc.get('title', '')}; "
            f"updated={inc.get('updated_at', '')}")
        for f in (inc.get('findings') or [])[:10]:
            add('incident_finding', inc.get('incident_id', ''),
                json.dumps({
                    'correlation_id': f.get('correlation_id'),
                    'correlation_type': f.get('correlation_type'),
                    'entities': f.get('entities'),
                    'correlation_score': f.get('correlation_score'),
                    'shared_ip_only': f.get('shared_ip_only'),
                }, default=str))

    for col, label in [('user_id', 'user'), ('source_ip', 'ip')]:
        top = _top_entities(df, col, n=max_entity_rows)
        for _, r in top.iterrows():
            add(f'top_{label}s', r[col],
                f"mean_score={float(r['mean_score']):.3f}; "
                f"max_score={float(r['max_score']):.3f}; "
                f"events={int(r['events'])}")

    for _, r in _top_events(df, n=max_event_rows).iterrows():
        add('top_events',
            f"{r.get('timestamp', '')} | {r.get('user_id', '')} | "
            f"{r.get('session_id', '')} | {r.get('event_type', '')}",
            f"score={r.get('combined_score', '')} "
            f"risk={r.get('risk_level', '')}")

    add('disclaimer', 'notice', _DISCLAIMER)

    return _sanitize_csv_frame(pd.DataFrame(rows)).to_csv(
        index=False).encode('utf-8')


# ----------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------

_RISK_BAR_COLORS = {
    'Low': '#00cc66',
    'Medium': '#ffaa00',
    'High': '#ff4444',
    'Critical': '#990000',
}


def _risk_distribution_drawing(risk: Dict[str, int]):
    """Stacked horizontal bar + legend for the risk distribution."""
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.lib.colors import HexColor
    from reportlab.pdfbase.pdfmetrics import stringWidth

    width, height = 460, 74
    d = Drawing(width, height)
    total = sum(risk.values()) or 1
    bar_h, bar_y = 14, height - 30
    x = 0.0
    for level in _RISK_ORDER:
        if level not in risk:
            continue
        w = width * risk[level] / total
        d.add(Rect(x, bar_y, w, bar_h,
                   fillColor=HexColor(
                       _RISK_BAR_COLORS.get(level, '#64748B')),
                   strokeColor=HexColor('#ffffff'), strokeWidth=0.5))
        x += w
    cx = 0.0
    for level in _RISK_ORDER:
        if level not in risk:
            continue
        count = risk[level]
        label = f"{level} {count} ({count / total * 100:.1f}%)"
        d.add(Rect(cx, 4, 7, 7,
                   fillColor=HexColor(
                       _RISK_BAR_COLORS.get(level, '#64748B')),
                   strokeColor=None))
        d.add(String(cx + 10, 5, label, fontName='Helvetica', fontSize=7,
                     fillColor=HexColor('#333333')))
        cx += 14 + stringWidth(label, 'Helvetica', 7) + 10
    return d


def _pdf_table(data, col_widths, header_bg='#1a1a2e'):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    return t


def generate_soc_summary_pdf(analysis: Dict,
                             incidents: Optional[List[Dict]] = None,
                             max_entity_rows: int = 10,
                             max_event_rows: int = 15) -> bytes:
    """Render a SOC summary report to PDF bytes via reportlab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle)
    from xml.sax.saxutils import escape as _xml_escape

    incidents = incidents or []
    df, corr = _df(analysis), _corr(analysis)

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(0.75 * inch, 0.45 * inch,
                          'CAID — SOC Summary Report')
        canvas.drawRightString(letter[0] - 0.75 * inch, 0.45 * inch,
                               f'Page {doc.page}')
        canvas.setStrokeColor(colors.HexColor('#dddddd'))
        canvas.setLineWidth(0.5)
        canvas.line(0.75 * inch, 0.6 * inch,
                    letter[0] - 0.75 * inch, 0.6 * inch)
        canvas.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        title='CAID SOC Summary Report', author='CAID')
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle('h1c', parent=styles['Heading1'], alignment=1)
    small = ParagraphStyle('small', parent=styles['Normal'],
                           fontSize=8, textColor=colors.grey)

    story = []
    story.append(Paragraph('CAID — SOC Summary Report', h1))
    story.append(Paragraph(_DISCLAIMER, small))
    story.append(Spacer(1, 0.15 * inch))

    # ---- Report details -------------------------------------------------
    metrics = _metrics(analysis, incidents)
    details = [
        ['Scenario', _scenario_label(analysis)],
        ['Generated (UTC)', _generated_at()],
        ['Data window', _data_window(df)],
        ['Events analyzed', metrics['events_analyzed']],
        ['Users / sessions / IPs',
         f"{metrics['users']} / {metrics['sessions']} / {metrics['source_ips']}"],
        ['Incidents', metrics['incidents_total']],
    ]
    t = Table(details, colWidths=[1.6 * inch, 4.8 * inch])
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.15 * inch))

    # ---- Executive summary -------------------------------------------------
    story.append(Paragraph('Executive Summary', styles['Heading2']))
    story.append(Paragraph(
        _xml_escape(_executive_summary(analysis, incidents)),
        styles['Normal']))
    story.append(Spacer(1, 0.15 * inch))

    # ---- Risk distribution ----------------------------------------------
    story.append(Paragraph('Risk Distribution', styles['Heading2']))
    risk = _risk_counts(df)
    if risk:
        story.append(_risk_distribution_drawing(risk))
        story.append(Spacer(1, 0.05 * inch))
        rdata = [['Risk level', 'Events', '% of total']]
        for level in _RISK_ORDER:
            if level in risk:
                rdata.append([level, str(risk[level]),
                              f'{risk[level] / len(df) * 100:.1f}%'])
        story.append(_pdf_table(rdata, [2.0 * inch, 1.2 * inch, 1.2 * inch]))
    else:
        story.append(Paragraph('No risk-level data available.', styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))

    # ---- Key metrics ------------------------------------------------------
    story.append(Paragraph('Key Metrics', styles['Heading2']))
    mdata = [['Metric', 'Value']]
    labels = {
        'ml_anomalies': 'ML anomalies',
        'high_critical_events': 'High/Critical events',
        'correlation_findings': 'Correlation findings',
        'shared_ip_findings': 'Shared-IP findings',
        'corroborated_shared_ip': 'Corroborated shared-IP findings',
        'incidents_open': 'Open incidents',
        'incidents_critical': 'Critical incidents',
    }
    for key, label in labels.items():
        mdata.append([label, metrics[key]])
    story.append(_pdf_table(mdata, [3.2 * inch, 1.6 * inch]))
    story.append(Spacer(1, 0.15 * inch))

    # ---- Calibration ------------------------------------------------------
    story.append(Paragraph('Calibration & Methodology', styles['Heading2']))
    cal_lines = []
    if not df.empty:
        for col, label in [('risk_calibration_status', 'Risk calibration'),
                           ('detection_calibration_status',
                            'Detection calibration'),
                           ('threshold_version', 'Threshold version')]:
            if col in df.columns:
                cal_lines.append(f'{label}: {_mode(df[col])}')
    cal_lines.append(
        'Scores are calibrated anomaly scores, not probabilities. '
        'Correlation strengths indicate resemblance, not confirmed '
        'coordination. Findings rest on synthetic/defensive research data.')
    for line in cal_lines:
        story.append(Paragraph(_xml_escape(line), styles['Normal']))
    story.append(Spacer(1, 0.15 * inch))

    # ---- Incidents ---------------------------------------------------------
    story.append(Paragraph(f'Incidents ({len(incidents)})', styles['Heading2']))
    if incidents:
        idata = [['ID', 'Severity', 'Status', 'Findings', 'Title', 'Updated']]
        for inc in incidents[:25]:
            title = str(inc.get('title', ''))
            if len(title) > 60:
                title = title[:57] + '...'
            idata.append([
                str(inc.get('incident_id', '')),
                str(inc.get('severity', '')),
                str(inc.get('status', '')),
                str(len(inc.get('findings') or [])),
                title,
                str(inc.get('updated_at', ''))[:19],
            ])
        story.append(_pdf_table(
            idata, [0.9 * inch, 0.6 * inch, 0.8 * inch, 0.5 * inch,
                    2.6 * inch, 1.0 * inch]))
        if len(incidents) > 25:
            story.append(Paragraph(
                f'{len(incidents) - 25} additional incidents omitted.', small))
    else:
        story.append(Paragraph('No incidents recorded for this analysis.',
                               styles['Normal']))
    story.append(Spacer(1, 0.15 * inch))

    # ---- Correlation findings by type --------------------------------------
    if not corr.empty and 'correlation_type' in corr.columns:
        story.append(Paragraph('Correlation Findings by Type',
                               styles['Heading2']))
        cdata = [['Type', 'Findings', 'Avg score', 'Shared-IP only']]
        for ctype, grp in corr.groupby('correlation_type'):
            avg = (grp['correlation_score'].mean()
                   if 'correlation_score' in grp.columns else float('nan'))
            shared_only = (int(grp['shared_ip_only'].sum())
                           if 'shared_ip_only' in grp.columns else 0)
            cdata.append([
                _TYPE_LABELS.get(str(ctype), str(ctype)),
                str(len(grp)), f'{avg:.3f}', str(shared_only)])
        story.append(_pdf_table(cdata, [2.6 * inch, 0.9 * inch,
                                        0.9 * inch, 1.0 * inch]))
        story.append(Spacer(1, 0.15 * inch))

    # ---- Top entities -------------------------------------------------------
    ent_rows = []
    for col, label in [('user_id', 'user'), ('source_ip', 'ip')]:
        top = _top_entities(df, col, n=max_entity_rows)
        for _, r in top.iterrows():
            ent_rows.append([label, str(r[col]),
                             f"{float(r['mean_score']):.3f}",
                             f"{float(r['max_score']):.3f}",
                             str(int(r['events']))])
    if ent_rows:
        story.append(Paragraph('Top Suspicious Entities', styles['Heading2']))
        edata = [['Type', 'Entity', 'Mean score', 'Max score', 'Events']]
        edata.extend(ent_rows)
        story.append(_pdf_table(edata, [0.6 * inch, 2.4 * inch, 0.9 * inch,
                                        0.9 * inch, 0.7 * inch]))
        story.append(Paragraph(
            'Ranked by mean combined score; scores are calibrated anomaly '
            'scores, not probabilities.', small))
        story.append(Spacer(1, 0.15 * inch))

    # ---- Highest-risk events -------------------------------------------------
    top = _top_events(df, n=max_event_rows)
    if len(top):
        story.append(Paragraph('Highest-Risk Events', styles['Heading2']))
        evdata = [['Timestamp', 'User', 'Session', 'Event', 'Score', 'Risk']]
        for _, r in top.iterrows():
            score = r.get('combined_score')
            score_txt = (f'{float(score):.3f}'
                         if pd.notna(score) else '')
            evdata.append([
                str(r.get('timestamp', ''))[:19],
                str(r.get('user_id', '')),
                str(r.get('session_id', ''))[:18],
                str(r.get('event_type', '')),
                score_txt,
                str(r.get('risk_level', '')),
            ])
        story.append(_pdf_table(
            evdata, [1.4 * inch, 0.8 * inch, 1.0 * inch, 1.3 * inch,
                     0.5 * inch, 0.6 * inch]))
        story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(_DISCLAIMER, small))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
