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
    'app_pages/streaming_health.py',
    'app_pages/detection_quality.py',
    'app_pages/reports.py',
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


def test_mitre_page_uses_cached_bundle():
    """Regression: MITRE page must render coverage from the cached STIX
    bundle on first visit, without requiring the download button."""
    at = AppTest.from_file(os.path.join(ROOT, 'app_pages/mitre.py'),
                           default_timeout=60)
    at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert 'Techniques in framework' in labels, \
        'MITRE coverage metrics missing — cached bundle not loaded'


def test_overview_load_and_analyze_produces_metrics():
    """Functional: overview's Load-and-analyze flow yields real KPIs."""
    at = AppTest.from_file(os.path.join(ROOT, 'app_pages/overview.py'),
                           default_timeout=240)
    at.run()
    btns = [b for b in at.button if 'Load and analyze' in (b.label or '')]
    assert btns, 'Load and analyze button missing on overview'
    btns[0].click().run()
    click_msgs = [e.value for grp in (at.error, at.warning, at.info)
                  for e in grp]
    # st.rerun() inside the load handler can leave a pending rerun;
    # drain it (bounded) until the post-analysis render is observed.
    for _ in range(3):
        if any(m.label == 'Events analyzed' for m in at.metric):
            break
        at.run()
    assert not at.exception, \
        f'overview raised after analysis: {[e.value for e in at.exception]}'
    labels = {m.label for m in at.metric}
    if not {'Events analyzed', 'Anomalies (ML)', 'Open incidents'} <= labels:
        msgs = [e.value for grp in (at.error, at.warning, at.info)
                for e in grp]
        pytest.fail(f'metrics missing; click-run messages: {click_msgs}; '
                    f'final messages: {msgs}; '
                    f'session keys: {sorted(at.session_state.keys())}')
    assert {'Events analyzed', 'Anomalies (ML)', 'Open incidents'} <= labels
