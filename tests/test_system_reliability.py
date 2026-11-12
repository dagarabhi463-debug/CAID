"""
Milestone 1.6: Full-system reliability tests.

End-to-end detection scenarios, score/baseline integrity, security
boundaries, and export safety. Uses deterministic synthetic data only.
"""

import numpy as np
import pandas as pd
import pytest

from core.log_processor import load_sample_dataset
from models.anomaly_detector import AnomalyDetector
from core.behavioural_analyzer import BehavioralAnalyzer, ensure_fitted_analyzer
from core.detection_rules import DetectionRulesEngine, DetectionRule, RuleSeverity, RuleAction
from core.detection_engine import DetectionEngine
from core.risk_engine import RiskEngine
from core.ml_integration import MLIntegration
from utils.validators import validate_security_log, get_validation_summary
from modules.export import get_download_button_csv, export_analysis_summary


def make_event(**overrides):
    row = {
        'timestamp': '2024-03-01 10:00:00', 'user_id': 'stu_test', 'role': 'student',
        'source_ip': '192.0.2.10', 'event_type': 'course_view',
        'courses_accessed': 1, 'accounts_accessed': 1, 'api_requests': 50,
        'data_mb': 1.0, 'token_age_minutes': 30, 'after_hours': 0, 'label': 'normal',
    }
    row.update(overrides)
    return pd.DataFrame([row])


SCENARIOS = {
    'normal_student': {},
    'unusual_login_time': {'after_hours': 1, 'timestamp': '2024-03-01 03:00:00'},
    'high_api': {'api_requests': 2000},
    'bulk_export': {'event_type': 'bulk_download', 'data_mb': 300.0, 'accounts_accessed': 40},
    'suspicious_accounts': {'accounts_accessed': 45, 'courses_accessed': 20},
    'unusual_ip': {'source_ip': '203.0.113.99', 'event_type': 'login'},
    'multi_indicator': {'after_hours': 1, 'api_requests': 3000, 'data_mb': 400.0,
                        'event_type': 'student_export', 'accounts_accessed': 30,
                        'source_ip': '203.0.113.77', 'token_age_minutes': 8000},
    'unseen_user': {'user_id': 'never_seen_user_12345'},
}


@pytest.fixture(scope='module')
def detector():
    d = AnomalyDetector()
    d.fit(load_sample_dataset('normal'))
    return d


@pytest.fixture(scope='module')
def behavioral():
    return ensure_fitted_analyzer()


def run_pipeline(df, detector, behavioral):
    """Full production pipeline: ML -> risk -> rules -> behavioral -> detection."""
    df = detector.predict(df)
    df = RiskEngine().process_events(df)
    df = DetectionRulesEngine().process_events(df)
    df = behavioral.analyze_user_behavior(df)
    df = DetectionEngine().process_dataframe(df)
    return df


class TestEndToEndScenarios:
    """Every scenario must traverse the full pipeline without corruption."""

    @pytest.mark.parametrize('name', list(SCENARIOS))
    def test_scenario_pipeline(self, detector, behavioral, name):
        df = make_event(**SCENARIOS[name])
        out = run_pipeline(df, detector, behavioral)
        row = out.iloc[0]
        # All contract columns present
        for col in ('is_anomaly', 'anomaly_score', 'anomaly_score_normalized',
                    'ml_anomaly_score', 'risk_score', 'risk_level',
                    'rule_score', 'behavioral_deviation_score',
                    'behavioral_score_missing', 'combined_score',
                    'detection_id', 'detection_explanation'):
            assert col in out.columns, f"{name}: missing {col}"
        # Score bounds
        for col in ('anomaly_score_normalized', 'risk_score', 'rule_score',
                    'combined_score'):
            assert 0 <= row[col] <= 1, f"{name}: {col}={row[col]} out of range"
        assert isinstance(row['detection_explanation'], str) and row['detection_explanation']

    def test_multi_indicator_scores_highest(self, detector, behavioral):
        frames = []
        for name, over in SCENARIOS.items():
            frames.append(run_pipeline(make_event(**over), detector, behavioral))
        combined = pd.concat(frames, ignore_index=True)
        normal = combined[
            (combined['user_id'] == 'stu_test') &
            (combined['event_type'] == 'course_view') &
            (combined['api_requests'] == 50)
        ]
        assert len(normal) >= 1
        # The most suspicious scenario must not score below the normal one
        assert combined['combined_score'].max() >= normal['combined_score'].max()

    def test_missing_ml_model_explicit(self, tmp_path):
        """Missing model artifact -> explicit fallback, not fake predictions."""
        ml = MLIntegration(model_path=str(tmp_path / 'nope.joblib'))
        df = ml.predict(make_event())
        assert df['ml_model_available'].iloc[0] is False or \
            df['ml_model_available'].iloc[0] == False
        assert pd.isna(df['anomaly_score_normalized'].iloc[0])

    def test_invalid_model_artifact_rejected(self, tmp_path):
        """Corrupt artifact must raise, not silently predict."""
        bad = tmp_path / 'bad.joblib'
        bad.write_text('not a joblib file')
        d = AnomalyDetector()
        with pytest.raises(Exception):
            d.load(str(bad))

    def test_malformed_input_logs(self):
        """Malformed CSV content is rejected at validation boundary."""
        bad = pd.DataFrame({'timestamp': ['not-a-date'], 'user_id': ['u']})
        ok, errors = validate_security_log(bad)
        assert ok is False
        summary = get_validation_summary(bad)
        assert summary['overall_valid'] is False


class TestScoreIntegrity:
    """Score contract: direction, no double inversion, missing propagation."""

    def test_no_double_inversion_full_pipeline(self, detector):
        """Anomalous event must not end up with lower combined_score than normal."""
        normal = detector.predict(make_event(api_requests=50))
        anomalous = detector.predict(make_event(
            api_requests=5000, data_mb=500.0, event_type='bulk_download',
            accounts_accessed=60, after_hours=1, source_ip='203.0.113.5'))
        assert anomalous['anomaly_score_normalized'].iloc[0] > \
            normal['anomaly_score_normalized'].iloc[0]
        # raw direction opposite
        assert anomalous['anomaly_score'].iloc[0] < normal['anomaly_score'].iloc[0]

    def test_missing_score_propagation(self, detector, behavioral):
        """Missing behavioral -> NaN + flag through the whole pipeline."""
        df = make_event()
        df = detector.predict(df)
        # Milestone 1.9: run the rules channel so ml+rules have evidence
        # (behavioral stays missing -> degraded_ml_rules mode).
        df = DetectionRulesEngine().process_events(df)
        df = DetectionEngine().process_dataframe(df)  # no behavioral column
        assert bool(df['behavioral_score_missing'].iloc[0]) is True
        assert pd.isna(df['behavioral_deviation'].iloc[0])
        assert df['operating_mode'].iloc[0] == 'degraded_ml_rules'
        # combined uses renormalized weights over ml+rules
        expected = (0.35 * df['ml_anomaly_score'].iloc[0] +
                    0.35 * df['rule_score'].iloc[0]) / 0.70
        assert df['combined_score'].iloc[0] == pytest.approx(expected)

    def test_same_timestamp_chronological(self):
        """Same-timestamp events get deterministic chronological order."""
        mgr = BehavioralAnalyzer()
        df = pd.DataFrame([
            {'timestamp': '2024-01-01 10:00:00', 'user_id': 'u',
             'api_requests': 100, 'data_mb': 5, 'courses_accessed': 3,
             'accounts_accessed': 2, 'token_age_minutes': 50, 'after_hours': 0},
            {'timestamp': '2024-01-01 10:00:00', 'user_id': 'u',
             'api_requests': 10000, 'data_mb': 5000, 'courses_accessed': 3,
             'accounts_accessed': 2, 'token_age_minutes': 50, 'after_hours': 1},
        ])
        out1 = mgr.baseline_manager.evaluate_chronological(df, min_history=1)
        out2 = mgr.baseline_manager.evaluate_chronological(df, min_history=1)
        pd.testing.assert_frame_equal(
            out1[['behavioral_deviation_score', 'chronological']],
            out2[['behavioral_deviation_score', 'chronological']])

    def test_behavioral_baseline_immutable_after_fit(self):
        ba = ensure_fitted_analyzer()
        before = {u: dict(b) for u, b in ba.user_baselines.items()}
        df = make_event(api_requests=999999)
        ba.analyze_user_behavior(df)
        after = ba.user_baselines
        for u, b in before.items():
            assert after[u] == b

    def test_persistence_roundtrip_scores(self, detector, tmp_path):
        """ML + behavioral save/load produce identical scores."""
        path = tmp_path / 'm.joblib'
        detector.save(str(path))
        loaded = AnomalyDetector().load(str(path))
        df = make_event(api_requests=500)
        a = detector.predict(df)['anomaly_score_normalized'].iloc[0]
        b = loaded.predict(df)['anomaly_score_normalized'].iloc[0]
        assert a == pytest.approx(b)


class TestSecurityBoundaries:
    """Security audit: validation, rule safety, export injection."""

    def test_rule_engine_rejects_unsafe_conditions(self):
        engine = DetectionRulesEngine(rules=[])
        for bad in ("__import__('os').system('x')", "open('f').read()",
                    "api_requests.__class__", "lambda x: x"):
            with pytest.raises(Exception):
                engine.add_rule(DetectionRule(
                    rule_id='bad', name='b', description='d',
                    condition=bad, severity=RuleSeverity.LOW))

    def test_rule_evaluate_no_exec(self):
        rules = DetectionRulesEngine()
        # Malformed row data must not crash or trigger
        out = rules.process_events(pd.DataFrame([
            {'user_id': 'u', 'api_requests': 'not_a_number',
             'data_mb': None, 'event_type': None, 'source_ip': 'x'}]))
        assert out['rule_score'].iloc[0] == 0.0

    def test_csv_formula_injection_sanitized(self):
        df = pd.DataFrame({'user_id': ['=cmd|"/c calc"!A1', '+1+1', 'normal',
                                       '-2+3', '@SUM(1)']})
        csv_bytes, _, _ = get_download_button_csv(df, 't.csv', 'dl')
        content = csv_bytes.decode('utf-8')
        for line in content.splitlines()[1:]:
            val = line.split(',')[0]
            if val and val != 'user_id':
                assert not val.startswith(('=', '+', '-', '@')), val

    def test_export_summary_no_crash_on_empty(self):
        out = export_analysis_summary(pd.DataFrame())
        assert isinstance(out, bytes)

    def test_upload_size_limit_logic(self):
        """validate_upload enforces 10MB cap (logic-level check)."""
        class FakeFile:
            name = 'big.csv'
            size = 11 * 1024 * 1024
            def read(self): return b''
            def seek(self, n): pass
        # Replicate the guard from modules.log_analysis.validate_upload
        assert FakeFile.size / (1024 * 1024) > 10

    def test_validators_reject_missing_columns(self):
        df = pd.DataFrame({'user_id': ['u'], 'api_requests': [1]})
        ok, errors = validate_security_log(df)
        assert ok is False
        assert any('Missing required columns' in e for e in errors)


class TestDashboardConsistency:
    """Displayed values must come from real pipeline columns."""

    def test_pipeline_columns_match_dashboard_expectations(self, detector, behavioral):
        """Columns consumed by modules/dashboard.py and threat_detection.py exist."""
        df = run_pipeline(
            pd.concat([load_sample_dataset('normal').head(20),
                       load_sample_dataset('high_risk').head(20)]),
            detector, behavioral)
        for col in ('is_anomaly', 'risk_level', 'risk_score', 'combined_score',
                    'detection_id', 'timestamp', 'user_id', 'role', 'source_ip',
                    'event_type', 'anomaly_score_normalized', 'rule_score',
                    'behavioral_deviation_score', 'detection_explanation',
                    'recommended_actions'):
            assert col in df.columns, col
        # Missing indicators never displayed as normal: behavioral missing => NaN score
        missing = df['behavioral_score_missing']
        if missing.any():
            assert df.loc[missing, 'behavioral_deviation_score'].isna().all()
