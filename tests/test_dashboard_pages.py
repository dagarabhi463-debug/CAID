"""Smoke tests: every dashboard page renders without exceptions."""

import os

import pytest

streamlit = pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = [
    'app_pages/overview.py',
    'app_pages/events.py',
    'app_pages/anomaly_detection.py',
    'app_pages/session_intelligence.py',
    'app_pages/correlation.py',
    'app_pages/investigation.py',
    'app_pages/mitre.py',
    'app_pages/ai_analyst.py',
    'app_pages/settings.py',
    'app_pages/log_analysis.py',
]


@pytest.mark.parametrize('page', PAGES)
def test_page_loads(page):
    at = AppTest.from_file(os.path.join(ROOT, page), default_timeout=60)
    at.run()
    assert not at.exception, \
        f'{page} raised: {[e.value for e in at.exception]}'


def test_entrypoint_navigation_loads():
    at = AppTest.from_file(os.path.join(ROOT, 'app.py'), default_timeout=60)
    at.run()
    assert not at.exception, \
        f'app.py raised: {[e.value for e in at.exception]}'
