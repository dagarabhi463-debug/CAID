"""Milestone 2.0.1 final regression tests."""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration import (
    CalibrationUnavailable,
    apply_detection_calibration,
    component_fingerprint,
    save_calibration,
)
from core.detection_engine import DetectionEngine
from core.feature_engineering import FeatureBaselineManager, engineer_features


def _log_row(user_id, api_requests, source_ip='198.51.100.10', ts='2026-01-05 08:00:00'):
    return {
        'timestamp': ts,
        'user_id': user_id,
        'role': 'student',
        'source_ip': source_ip,
        'event_type': 'course_view',
        'courses_accessed': 1,
        'accounts_accessed': 1,
        'api_requests': api_requests,
        'data_mb': 1.0,
        'token_age_minutes': 30,
        'after_hours': 0,
        'label': 'normal',
    }


class TestFeatureBaselineFingerprint:
    def test_fingerprint_deterministic_and_verified(self, tmp_path):
        train = pd.DataFrame([
            _log_row('user_a', 10, ts=f'2026-01-05 08:0{i}:00')
            for i in range(3)
        ] + [
            _log_row('user_b', 100, ts=f'2026-01-05 09:0{i}:00')
            for i in range(3)
        ])
        manager = FeatureBaselineManager().fit(train)
        fingerprint = manager.fingerprint()

        assert fingerprint == manager.fingerprint()
        assert component_fingerprint(manager, 'feature_baseline') == fingerprint

        path = tmp_path / 'baseline.joblib'
        joblib.dump(manager, path)
        loaded = joblib.load(path)
        assert loaded.fingerprint() == fingerprint

    def test_calibration_rejects_feature_baseline_mismatch(self, tmp_path):
        manager = FeatureBaselineManager().fit(pd.DataFrame([
            _log_row('user_a', 10), _log_row('user_a', 12),
        ]))
        fingerprint = manager.fingerprint()
        path = tmp_path / 'cal.joblib'
        save_calibration(
            {'medium': 0.1, 'high': 0.2, 'critical': 0.3},
            {'calibration_rows': 2},
            detector=None,
            dataset_role='calibration',
            path=str(path),
            artifact_version=2,
            fingerprints={'feature_baseline': fingerprint},
            channel_profile='v2_0',
            detection_thresholds={'medium': 0.2, 'high': 0.4, 'critical': 0.6},
        )

        from core.calibration import load_calibration
        assert load_calibration(str(path), fingerprints={
            'feature_baseline': fingerprint})['fingerprints']['feature_baseline'] == fingerprint
        with pytest.raises(CalibrationUnavailable):
            load_calibration(str(path), fingerprints={
                'feature_baseline': 'wrong-fingerprint'})


class TestDetectionCalibration:
    def _row(self):
        return pd.Series({
            'user_id': 'u1',
            'timestamp': '2026-01-05 08:00:00',
            'anomaly_score_normalized': 0.5,
            'rule_score': 0.5,
            'behavioral_deviation_score': 0.5,
            'session_deviation_score': 0.5,
            'sequence_deviation_score': 0.5,
            'session_score_missing': False,
            'sequence_score_missing': False,
        })

    def _artifact(self, path, critical):
        save_calibration(
            {'medium': 0.2, 'high': 0.4, 'critical': 0.3},
            {'calibration_rows': 2},
            detector=None,
            dataset_role='calibration',
            path=str(path),
            artifact_version=2,
            fingerprints={},
            channel_profile='v2_0',
            detection_thresholds={'medium': 0.2, 'high': 0.4,
                                  'critical': critical},
        )

    def test_calibrated_thresholds_control_final_risk_level(self, tmp_path):
        high_path = tmp_path / 'high.joblib'
        critical_path = tmp_path / 'critical.joblib'
        self._artifact(high_path, critical=0.6)
        self._artifact(critical_path, critical=0.45)

        engine = DetectionEngine(channel_profile='v2_0')
        artifact = apply_detection_calibration(engine, path=str(high_path))
        assert artifact is not None
        assert engine.calibration_status == 'calibrated'
        assert engine.threshold_version == artifact['threshold_version']
        assert engine.thresholds['critical'] == pytest.approx(0.6)
        assert engine.process_event(self._row()).risk_level.value == 'High'

        stricter = DetectionEngine(channel_profile='v2_0')
        apply_detection_calibration(stricter, path=str(critical_path))
        assert stricter.process_event(self._row()).risk_level.value == 'Critical'

    def test_fingerprint_mismatch_leaves_engine_uncalibrated(self, tmp_path):
        path = tmp_path / 'cal.joblib'
        save_calibration(
            {'medium': 0.2, 'high': 0.4, 'critical': 0.3},
            {'calibration_rows': 2},
            detector=None,
            dataset_role='calibration',
            path=str(path),
            artifact_version=2,
            fingerprints={'feature_baseline': 'expected'},
            channel_profile='v2_0',
            detection_thresholds={'medium': 0.1, 'high': 0.2, 'critical': 0.3},
        )

        engine = DetectionEngine(channel_profile='v2_0')
        assert apply_detection_calibration(
            engine, path=str(path),
            fingerprints={'feature_baseline': 'actual'}) is None
        assert engine.calibration_status == 'uncalibrated_default_thresholds'
        assert engine.thresholds['critical'] == pytest.approx(0.75)


class TestCommonIpBehavior:
    def test_shared_ip_preserves_distinct_user_baselines(self):
        train = pd.DataFrame(
            [_log_row('normal_user', 10, ts=f'2026-01-05 08:0{i}:00')
             for i in range(3)] +
            [_log_row('suspicious_user', 100, ts=f'2026-01-05 09:0{i}:00')
             for i in range(3)]
        )
        manager = FeatureBaselineManager().fit(train)
        eval_df = pd.DataFrame([
            _log_row('normal_user', 10, ts='2026-01-06 08:00:00'),
            _log_row('suspicious_user', 100, ts='2026-01-06 08:01:00'),
        ])

        out, _ = engineer_features(eval_df, manager)
        assert out['ip_event_count'].nunique() == 1
        assert out['ip_event_count'].iloc[0] == 6
        assert out['user_mean_api'].tolist() == [10.0, 100.0]
        assert out['user_api_z'].nunique() == 1

    def test_independent_v2_holdout_has_mixed_users_on_shared_ips(self):
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data', 'independent_v2')
        cohorts = json.load(open(os.path.join(base, 'entity_cohorts.json'),
                                 encoding='utf-8'))
        df = pd.read_csv(os.path.join(base, 'holdout.csv'))
        shared = set(cohorts['shared_ips'])
        sub = df[df['source_ip'].isin(shared)].copy()
        sub['positive'] = sub['label'].isin(['suspicious', 'high_risk'])

        mixed = False
        for _, group in sub.groupby('source_ip'):
            normal_users = set(group.loc[group['label'] == 'normal', 'user_id'])
            positive_users = set(group.loc[group['positive'], 'user_id'])
            if normal_users and positive_users and normal_users != positive_users:
                mixed = True
                break
        assert mixed

    def test_common_ip_scenario_masks_include_normal_and_positive_rows(self):
        from scripts.milestone_2_0_1_audit import scenario_masks
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data', 'independent_v2')
        df = pd.read_csv(os.path.join(base, 'holdout.csv'))
        manifest = json.load(open(os.path.join(base, 'dataset_manifest.json'),
                                  encoding='utf-8'))
        masks = scenario_masks(df, manifest)

        assert masks['common_ip_all'].sum() > 0
        assert (df.loc[masks['common_ip_normal'], 'label'] == 'normal').all()
        assert masks['common_ip_normal'].sum() > 0
        assert df.loc[masks['common_ip_positives'], 'label'].isin(
            ['suspicious', 'high_risk']).all()
        assert masks['common_ip_positives'].sum() > 0
