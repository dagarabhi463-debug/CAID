"""
Milestone 1.9 tests: independent datasets, leakage guards, calibration
lifecycle, missing-ML operating modes, structured evidence, artifact
validation.

All tests use small in-memory fixtures or the generated independent
datasets — no fabricated metrics.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from core.detection_engine import (
    DetectionEngine, Detection, RiskLevel, OperatingMode)
from core.detection_rules import DetectionRulesEngine
from core.risk_engine import RiskEngine, RiskLevel as EngineRiskLevel
from core.calibration import (
    CalibrationUnavailable, save_calibration, load_calibration,
    apply_to_engine, detector_fingerprint, dataset_manifest_hash,
    ARTIFACT_VERSION as CAL_ARTIFACT_VERSION)
from models.anomaly_detector import AnomalyDetector

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IND = os.path.join(BASE, 'data', 'independent')


def _row(user_id='u1', source_ip='192.0.2.10', event_type='login',
         api=50, data=1.0, courses=1, accounts=1, ts='2024-01-01 10:00:00',
         after_hours=0, label='normal'):
    return {
        'timestamp': ts, 'user_id': user_id, 'role': 'student',
        'source_ip': source_ip, 'event_type': event_type,
        'courses_accessed': courses, 'accounts_accessed': accounts,
        'api_requests': api, 'data_mb': data,
        'token_age_minutes': 30, 'after_hours': after_hours,
        'label': label,
    }


# ---------------------------------------------------------------------------
# Phase 1 — independent dataset integrity
# ---------------------------------------------------------------------------

class TestIndependentDatasets:
    @pytest.fixture(scope='class')
    def splits(self):
        out = {}
        for name in ('train', 'calibration', 'validation', 'holdout'):
            p = os.path.join(IND, f'{name}.csv')
            if os.path.exists(p):
                out[name] = pd.read_csv(p)
        return out

    def test_all_splits_exist(self, splits):
        assert set(splits) == {'train', 'calibration', 'validation',
                               'holdout'}

    def test_labels_do_not_determine_ip_prefix(self, splits):
        """Holdout label x /24-prefix crosstab must be roughly uniform —
        no prefix may be dominated by one label."""
        df = splits['holdout'].copy()
        df['prefix'] = df['source_ip'].str.extract(r'(\d+\.\d+\.\d+\.)')[0]
        ct = pd.crosstab(df['prefix'], df['label'])
        # Each prefix's modal label share must be well below 1.0
        purity = ct.max(axis=1) / ct.sum(axis=1)
        assert purity.max() < 0.80
        # And every label appears under every prefix family overall
        for lab in ('normal', 'suspicious', 'high_risk'):
            sub = df[df['label'] == lab]
            assert sub['prefix'].nunique() >= 2

    def test_holdout_only_entities_absent_from_train(self, splits):
        cohorts = json.load(open(os.path.join(IND, 'entity_cohorts.json')))
        train_users = set(splits['train']['user_id'])
        train_ips = set(splits['train']['source_ip'])
        holdout_only_users = {u for u, c in cohorts['users'].items()
                              if c == 'holdout_only'}
        holdout_only_ips = {i for i, c in cohorts['ips'].items()
                            if c == 'holdout_only'}
        assert not (holdout_only_users & train_users)
        assert not (holdout_only_ips & train_ips)
        # And they DO appear in holdout (scenario coverage)
        assert holdout_only_users & set(splits['holdout']['user_id'])
        assert holdout_only_ips & set(splits['holdout']['source_ip'])

    def test_shared_entities_span_splits(self, splits):
        shared_users = {u for u, c in
                        json.load(open(os.path.join(
                            IND, 'entity_cohorts.json')))['users'].items()
                        if c == 'shared'}
        for name in ('train', 'calibration', 'validation', 'holdout'):
            present = shared_users & set(splits[name]['user_id'])
            assert len(present) > 50, f'{name} missing shared users'

    def test_no_exact_duplicates_across_splits(self, splits):
        keys = ['user_id', 'source_ip', 'timestamp', 'event_type',
                'api_requests', 'data_mb', 'label']
        seen = set()
        for name, df in splits.items():
            for row in df[keys].itertuples(index=False, name=None):
                assert row not in seen, f'duplicate row in {name}'
                seen.add(row)

    def test_manifest_records_seed_and_roles(self):
        m = json.load(open(os.path.join(IND, 'dataset_manifest.json')))
        assert m['seed'] == 1900
        assert set(m['splits']) == {'train', 'calibration', 'validation',
                                    'holdout'}
        assert m['splits']['holdout']['role'] == 'holdout'


# ---------------------------------------------------------------------------
# Phase 2 — leakage guards in code
# ---------------------------------------------------------------------------

class TestLeakageGuards:
    def test_ip_prefix_rule_removed_from_defaults(self):
        engine = DetectionRulesEngine()
        assert 'suspicious_ip_class' not in engine.rules
        # ip_class must not appear in ANY default rule condition
        for rid, rule in engine.rules.items():
            assert 'ip_class' not in rule.condition

    def test_label_column_not_a_feature(self):
        from core.feature_engineering import engineer_features
        df = pd.DataFrame([_row(label='suspicious')])
        out, cols = engineer_features(df)
        assert 'label' not in cols
        # engineered ip_class exists but is not an ML feature
        assert 'ip_class' not in cols

    def test_is_sensitive_event_overlap_in_holdout(self):
        """is_sensitive_event must not be perfectly label-correlated in
        the independent holdout (unlike the legacy generator)."""
        df = pd.read_csv(os.path.join(IND, 'holdout.csv'))
        sensitive = {'grade_download', 'student_export', 'course_export',
                     'bulk_download'}
        df['sens'] = df['event_type'].isin(sensitive)
        pos = df['label'].isin(['suspicious', 'high_risk'])
        # Both classes must contain sensitive and non-sensitive events
        assert df.loc[pos, 'sens'].sum() > 0
        assert (~df.loc[pos, 'sens']).sum() > 0
        assert df.loc[~pos, 'sens'].sum() > 0   # normals can be 'sensitive'
        assert (~df.loc[~pos, 'sens']).sum() > 0


# ---------------------------------------------------------------------------
# Phase 4 — calibration lifecycle
# ---------------------------------------------------------------------------

class TestCalibrationLifecycle:
    def _detector(self):
        det = AnomalyDetector()
        det.fit(pd.DataFrame([_row(user_id=f'u{i}') for i in range(10)]))
        return det

    def test_roundtrip_preserves_metadata(self, tmp_path):
        det = self._detector()
        p = str(tmp_path / 'thr.joblib')
        art = save_calibration(
            {'medium': 0.1, 'high': 0.3, 'critical': 0.5},
            {'calibration_rows': 10, 'score_mean': 0.2},
            detector=det, dataset_role='calibration',
            manifest_hash='abc123', path=p, percentiles=(30, 65, 85))
        loaded = load_calibration(path=p, detector=det)
        assert loaded['thresholds'] == {'medium': 0.1, 'high': 0.3,
                                        'critical': 0.5}
        assert loaded['threshold_version'] == art['threshold_version']
        assert loaded['dataset_role'] == 'calibration'
        assert loaded['dataset_manifest_hash'] == 'abc123'
        assert loaded['detector_fingerprint'] == detector_fingerprint(det)

    def test_fingerprint_mismatch_rejected(self, tmp_path):
        det_a = self._detector()
        det_b = AnomalyDetector()
        det_b.fit(pd.DataFrame([_row(user_id=f'v{i}') for i in range(10)]))
        p = str(tmp_path / 'thr.joblib')
        save_calibration({'medium': 0.1, 'high': 0.3, 'critical': 0.5},
                         {}, detector=det_a, dataset_role='calibration',
                         path=p)
        # Same schema/hyperparams -> same fingerprint is possible; force
        # mismatch by checking the guard logic directly with a doctored fp
        with pytest.raises(CalibrationUnavailable):
            load_calibration(path=p, detector=object())  # no fingerprint
        # doctored artifact with wrong fingerprint
        import joblib
        data = joblib.load(p)
        data['detector_fingerprint'] = 'deadbeefdeadbeef'
        joblib.dump(data, p)
        with pytest.raises(CalibrationUnavailable):
            load_calibration(path=p, detector=det_a)

    def test_non_calibration_role_rejected(self, tmp_path):
        det = self._detector()
        with pytest.raises(ValueError):
            save_calibration({'medium': 0.1, 'high': 0.3, 'critical': 0.5},
                             {}, detector=det, dataset_role='holdout',
                             path=str(tmp_path / 'x.joblib'))

    def test_apply_to_engine_sets_status(self, tmp_path):
        det = self._detector()
        p = str(tmp_path / 'thr.joblib')
        save_calibration({'medium': 0.1, 'high': 0.3, 'critical': 0.5},
                         {'calibration_rows': 5}, detector=det,
                         dataset_role='calibration', path=p)
        engine = RiskEngine()
        art = apply_to_engine(engine, detector=det, path=p)
        assert art is not None
        assert engine.calibration_status == 'calibrated'
        assert engine.threshold_version == art['threshold_version']
        assert engine.fixed_thresholds['critical'] == 0.5
        # classify uses fixed thresholds now (0.4 >= high=0.3, < critical=0.5)
        assert engine.classify_risk(0.4) == EngineRiskLevel.HIGH
        assert engine.classify_risk(0.6) == EngineRiskLevel.CRITICAL

    def test_apply_to_engine_missing_artifact_uncalibrated(self, tmp_path):
        engine = RiskEngine()
        art = apply_to_engine(engine, path=str(tmp_path / 'nope.joblib'))
        assert art is None
        assert engine.calibration_status == 'uncalibrated_dataset_relative'
        assert engine.fixed_thresholds is None
        # dataset-relative fallback still functions
        engine.fit(pd.DataFrame({'risk_score': [0.0, 0.5, 1.0]}))
        assert engine.classify_risk(1.0) == EngineRiskLevel.CRITICAL

    def test_calibrate_script_rejects_holdout_role(self):
        """The CLI guard: manifest role must be 'calibration'."""
        import subprocess
        import sys
        script = os.path.join(BASE, 'scripts', 'calibrate_thresholds.py')
        r = subprocess.run(
            [sys.executable, script,
             '--detector', 'models/saved_models/anomaly_detector.joblib',
             '--data', os.path.join(IND, 'holdout.csv'),
             '--manifest', os.path.join(IND, 'dataset_manifest.json')],
            capture_output=True, text=True, cwd=BASE)
        assert r.returncode != 0
        assert 'REFUSED' in (r.stdout + r.stderr)


# ---------------------------------------------------------------------------
# Phase 5 — missing-ML operating modes
# ---------------------------------------------------------------------------

class TestOperatingModes:
    def _df(self, n=3, with_ml=True, with_behavioral=True):
        rows = []
        for i in range(n):
            r = _row(user_id=f'm{i}', api=50 + i * 100,
                     data=1.0 + i * 5, accounts=2 + i * 4)
            rows.append(r)
        df = pd.DataFrame(rows)
        if with_ml:
            df['anomaly_score_normalized'] = [0.1, 0.4, 0.9][:n]
            df['is_anomaly'] = [False, False, True][:n]
        if with_behavioral:
            df['behavioral_deviation_score'] = [0.0, 0.2, 0.8][:n]
            df['behavioral_score_missing'] = False
        return df

    def test_full_mode_all_channels(self):
        df = self._df()
        out = DetectionRulesEngine().process_events(df)
        out = DetectionEngine().process_dataframe(out)
        assert set(out['operating_mode']) == {
            OperatingMode.FULL_ML_RULES_BEHAVIORAL.value}
        assert out['combined_score'].notna().all()
        assert (out['risk_level'] != RiskLevel.INSUFFICIENT.value).all()

    def test_degraded_mode_ml_missing(self):
        """ML unavailable but rules+behavioral present -> degraded mode,
        weights renormalize, neutral placeholder NOT counted."""
        df = self._df(with_ml=False)
        df = DetectionRulesEngine().process_events(df)
        df['behavioral_deviation_score'] = [0.0, 0.2, 0.8]
        df['behavioral_score_missing'] = False
        out = DetectionEngine().process_dataframe(df)
        assert set(out['operating_mode']) == {
            OperatingMode.DEGRADED_RULES_BEHAVIORAL.value}
        assert out['ml_score_missing'].all()
        assert out['combined_score'].notna().all()
        # combined = (0.35*rule + 0.30*behavioral)/0.65 — no 0.5 ML term
        row = out.iloc[2]
        expected = (0.35 * row['rule_score'] + 0.30 * 0.8) / 0.65
        assert row['combined_score'] == pytest.approx(expected, abs=1e-6)

    def test_insufficient_evidence_mode(self):
        """Only one channel with real evidence -> INSUFFICIENT_EVIDENCE,
        NaN score, no HIGH/CRITICAL classification."""
        df = pd.DataFrame([_row(api=5000, data=500.0, accounts=90)])
        # no ML columns, no behavioral columns -> only rules channel
        out = DetectionEngine().process_dataframe(df)
        assert set(out['operating_mode']) == {
            OperatingMode.INSUFFICIENT_EVIDENCE.value}
        assert out['combined_score'].isna().all()
        assert (out['risk_level'] == RiskLevel.INSUFFICIENT.value).all()
        assert (out['risk_level'].isin(['High', 'Critical'])).sum() == 0
        ev = out.iloc[0]['structured_evidence']
        assert any(e['channel'] == 'combined'
                   and e['contribution_or_severity']
                   == 'insufficient_evidence' for e in ev)

    def test_missing_ml_not_counted_as_observation(self):
        """A row with ml_score_missing must not silently contribute 0.5."""
        df = pd.DataFrame([_row()])  # bare row: no ml/behavioral cols
        det = DetectionEngine()
        d = det.process_event(df.iloc[0])
        assert d.ml_score_missing is True
        assert d.operating_mode == OperatingMode.INSUFFICIENT_EVIDENCE.value
        assert np.isnan(d.combined_score)

    def test_operating_mode_documented_in_explanation(self):
        df = self._df(with_ml=False)
        df = DetectionRulesEngine().process_events(df)
        df['behavioral_deviation_score'] = [0.0, 0.2, 0.8]
        df['behavioral_score_missing'] = False
        out = DetectionEngine().process_dataframe(df)
        assert 'degraded' in out.iloc[0]['detection_explanation']


# ---------------------------------------------------------------------------
# Phase 7 — structured evidence
# ---------------------------------------------------------------------------

class TestStructuredEvidence:
    def test_evidence_objects_have_required_fields(self):
        df = pd.DataFrame([_row(api=500, data=50.0, accounts=30)])
        df['anomaly_score_normalized'] = [0.9]
        df['is_anomaly'] = [True]
        df['behavioral_deviation_score'] = [0.7]
        df['behavioral_score_missing'] = [False]
        df = DetectionRulesEngine().process_events(df)
        out = DetectionEngine().process_dataframe(df)
        ev = out.iloc[0]['structured_evidence']
        required = {'channel', 'evidence_name', 'observed_value',
                    'expected_baseline_or_threshold',
                    'contribution_or_severity', 'availability',
                    'explanation'}
        assert len(ev) >= 3
        for e in ev:
            assert required <= set(e.keys())
        channels = {e['channel'] for e in ev}
        assert {'ml', 'rules', 'behavioral', 'combined'} <= channels
        # explanation string preserved alongside structured evidence
        assert isinstance(out.iloc[0]['detection_explanation'], str)
        assert len(out.iloc[0]['detection_explanation']) > 0

    def test_missing_channel_evidence_marked(self):
        df = pd.DataFrame([_row()])
        out = DetectionEngine().process_dataframe(df)
        ev = out.iloc[0]['structured_evidence']
        ml_ev = [e for e in ev if e['channel'] == 'ml'][0]
        assert ml_ev['availability'] == 'missing'
        assert ml_ev['contribution_or_severity'] == 'excluded'


# ---------------------------------------------------------------------------
# Phase 8 — artifact validation
# ---------------------------------------------------------------------------

class TestArtifactValidation:
    def test_v2_artifact_loads_and_predicts_consistently(self):
        path = os.path.join(BASE, 'models', 'saved_models',
                            'anomaly_detector.joblib')
        if not os.path.exists(path):
            pytest.skip('production artifact not present')
        det = AnomalyDetector().load(path)
        df = pd.DataFrame([_row(), _row(user_id='u2', api=500)])
        a = det.predict(df)
        b = det.predict(df)
        pd.testing.assert_frame_equal(a, b)  # deterministic
        # save->load round trip preserves predictions
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'm.joblib')
            det.save(p)
            det2 = AnomalyDetector().load(p)
            c = det2.predict(df)
        pd.testing.assert_frame_equal(a, c)

    def test_incompatible_artifact_version_fails(self, tmp_path):
        det = AnomalyDetector()
        det.fit(pd.DataFrame([_row(user_id=f'x{i}') for i in range(10)]))
        p = str(tmp_path / 'old.joblib')
        det.save(p)
        import joblib
        data = joblib.load(p)
        data['artifact_version'] = 999
        joblib.dump(data, p)
        with pytest.raises(ValueError, match='Incompatible'):
            AnomalyDetector().load(p)

    def test_feature_schema_mismatch_rejected(self):
        det = AnomalyDetector()
        det.fit(pd.DataFrame([_row(user_id=f'y{i}') for i in range(10)]))
        det.feature_columns = det.feature_columns[:-1]  # doctored schema
        with pytest.raises(ValueError):
            det.predict(pd.DataFrame([_row()]))

    def test_calibration_status_default_uncalibrated(self):
        engine = RiskEngine()
        assert engine.calibration_status == 'uncalibrated_dataset_relative'
        assert engine.threshold_version is None
