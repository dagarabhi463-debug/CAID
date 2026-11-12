"""Unit tests for core.summary_report (SOC summary PDF/CSV export)."""

import pandas as pd
import pytest

from core.summary_report import (
    generate_soc_summary_csv,
    generate_soc_summary_pdf,
    report_outline,
)

reportlab = pytest.importorskip('reportlab')


def _analysis():
    df = pd.DataFrame({
        'timestamp': ['2026-01-01 10:00', '2026-01-01 11:00',
                      '2026-01-02 09:00'],
        'user_id': ['u1', 'u2', 'u1'],
        'session_id': ['s1', 's2', 's3'],
        'source_ip': ['10.0.0.1', '10.0.0.2', '10.0.0.1'],
        'event_type': ['login', 'file_download', 'login'],
        'risk_level': ['Low', 'High', 'Critical'],
        'is_anomaly': [False, True, True],
        'combined_score': [0.1, 0.8, 0.95],
        'risk_calibration_status': ['ok'] * 3,
        'detection_calibration_status': ['ok'] * 3,
        'threshold_version': ['v2.0'] * 3,
    })
    corr = pd.DataFrame({
        'correlation_id': ['c1'],
        'correlation_type': ['shared_ip_activity'],
        'entities': [['u1', 'u2']],
        'sessions': [['s1', 's2']],
        'correlation_score': [0.7],
        'shared_ip_only': [True],
        'operating_note': ['note'],
        'evidence': [[]],
    })
    return {'scored_events': df, 'correlations': corr,
            'scenario_key': 'test', 'scenario_label': 'Test Scenario'}


def _incident():
    return {'incident_id': 'inc_abc', 'title': '1 correlated finding',
            'severity': 'High', 'status': 'Open',
            'findings': [{'correlation_id': 'c1'}],
            'updated_at': '2026-01-02T00:00:00'}


def test_pdf_is_valid_pdf_bytes():
    pdf = generate_soc_summary_pdf(_analysis(), [_incident()])
    assert pdf[:5] == b'%PDF-'
    assert b'%%EOF' in pdf[-64:]


def test_pdf_handles_empty_analysis():
    pdf = generate_soc_summary_pdf(
        {'scored_events': None, 'correlations': None}, [])
    assert pdf[:5] == b'%PDF-'


def test_csv_contains_report_sections():
    csv = generate_soc_summary_csv(_analysis(), [_incident()]).decode('utf-8')
    for token in ('report,generated_at', 'risk_distribution,Critical',
                  'metrics,events_analyzed', 'incident,inc_abc',
                  'top_users', 'top_events', 'disclaimer'):
        assert token in csv


def test_csv_empty_analysis():
    csv = generate_soc_summary_csv({}, None).decode('utf-8')
    assert 'report,scenario' in csv
    assert 'disclaimer' in csv


def test_report_outline_sections():
    outline = report_outline(_analysis(), [_incident()])
    sections = [o['section'] for o in outline]
    assert 'Executive summary' in sections
    assert 'Risk distribution' in sections
    assert 'Incidents' in sections
    assert 'Correlation findings by type' in sections
    assert sections[-1] == 'Disclaimer'
