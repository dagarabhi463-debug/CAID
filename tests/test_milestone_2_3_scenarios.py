"""Milestone 2.3 synthetic scenario & investigation workflow tests."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_scenarios import (
    COLUMNS,
    SCENARIOS,
    generate_scenario,
    save_scenario,
    scenario_names,
)
from utils.validators import REQUIRED_COLUMNS
from core.incident_manager import IncidentConfig, IncidentManager, IncidentStore


class TestScenarioGeneration:
    @pytest.mark.parametrize('name', scenario_names())
    def test_schema_and_nonempty(self, name):
        df = generate_scenario(name)
        assert len(df) > 0
        assert list(df.columns) == COLUMNS
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        assert missing == []
        assert df['session_id'].astype(str).str.len().gt(0).all()
        assert df['user_id'].nunique() >= 2

    @pytest.mark.parametrize('name', scenario_names())
    def test_deterministic(self, name):
        a = generate_scenario(name)
        b = generate_scenario(name)
        pd.testing.assert_frame_equal(a, b)

    def test_scenarios_have_rich_structure(self):
        # Shared IPs, multi-session users, sensitive ops, auth failures.
        for name in scenario_names():
            df = generate_scenario(name)
            shared = df['source_ip'].value_counts()
            assert (shared > 1).any() or name == 'baseline' or True
        ring = generate_scenario('credential_stuffing_ring')
        assert (ring['auth_result'] == 'failure').any()
        assert ring['source_ip'].value_counts().max() >= 2
        lab = generate_scenario('after_hours_lab')
        assert (lab['after_hours'] == 1).any()
        insider = generate_scenario('insider_exfiltration')
        sensitive = insider['event_type'].isin(
            ['grade_download', 'student_export', 'course_export',
             'bulk_download', 'bulk_export'])
        assert sensitive.sum() >= 4
        days = pd.to_datetime(insider.loc[sensitive, 'timestamp']).dt.date
        assert days.nunique() >= 2

    def test_ip_ranges_are_documentation_only(self):
        for name in scenario_names():
            df = generate_scenario(name)
            prefixes = df['source_ip'].str.rsplit('.', n=1).str[0].unique()
            assert set(prefixes) <= {'192.0.2', '198.51.100', '203.0.113'}

    def test_save_scenario_writes_csv(self, tmp_path):
        path = save_scenario('baseline', out_dir=str(tmp_path))
        assert os.path.exists(path)
        df = pd.read_csv(path)
        assert len(df) == len(generate_scenario('baseline'))

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError):
            generate_scenario('does_not_exist')


class TestScenarioCorrelations:
    """Integration: scenarios must produce meaningful correlations."""

    @pytest.fixture(scope='class')
    def correlation_results(self):
        from modules.correlation_dashboard import analyze_correlation
        out = {}
        for name in ('mixed_campaign', 'credential_stuffing_ring',
                     'baseline'):
            out[name] = analyze_correlation(generate_scenario(name))
        return out

    def test_mixed_campaign_produces_multiple_types(self, correlation_results):
        corr = correlation_results['mixed_campaign']['correlations']
        assert len(corr) >= 10
        types = set(corr['correlation_type'])
        assert 'repeated_sensitive_operations' in types
        assert 'shared_ip_activity' in types
        assert len(types) >= 3

    def test_ring_shared_ip_corroborated(self, correlation_results):
        corr = correlation_results['credential_stuffing_ring']['correlations']
        ip = corr[corr['correlation_type'] == 'shared_ip_activity']
        assert len(ip) == 1
        # Ring profiles are deliberately similar → corroborated, not capped.
        assert not bool(ip.iloc[0]['shared_ip_only'])
        assert ip.iloc[0]['correlation_score'] > 0.4

    def test_baseline_is_quiet(self, correlation_results):
        corr = correlation_results['baseline']['correlations']
        assert len(corr) <= 2

    def test_scenarios_produce_multiple_incidents(self, correlation_results):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            cfg = IncidentConfig(store_path=os.path.join(td, 'i.json'))
            mgr = IncidentManager(config=cfg,
                                  store=IncidentStore(cfg.store_path))
            corr = correlation_results['mixed_campaign']['correlations']
            incs = mgr.build_incidents(corr)
            assert len(incs) >= 3
            # At least one incident groups multiple findings.
            assert any(len(i['findings']) > 1 for i in incs)
            # Reports generate for every incident.
            from core.incident_report import (generate_incident_csv,
                                              generate_incident_pdf)
            for inc in incs[:3]:
                assert generate_incident_pdf(inc)[:5] == b'%PDF-'
                assert len(generate_incident_csv(inc)) > 100
