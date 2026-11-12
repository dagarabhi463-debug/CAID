"""Milestone 2.4 closure — AUD-002 artifact lifecycle tests.

Verifies the unified ensure_* policy:
- loaded artifact        -> status 'loaded', no refit
- missing artifact       -> fit on normal data + persist,
                            'refit_fallback_missing'
- corrupt artifact       -> quarantine (never silently overwritten) +
                            fit on normal data + persist fresh,
                            'refit_fallback_corrupt'
- fitting only ever uses designated training-normal data
"""

import os

import pytest

from core.artifact_lifecycle import quarantine_corrupt
from core.log_processor import load_sample_dataset


@pytest.fixture(scope='module')
def normal_df():
    return load_sample_dataset('normal')


def _corrupt(path):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'this is not a valid joblib artifact')


class TestQuarantineHelper:
    def test_quarantine_moves_file(self, tmp_path):
        p = tmp_path / 'artifact.joblib'
        p.write_bytes(b'corrupt')
        q = quarantine_corrupt(str(p))
        assert q is not None
        assert not p.exists()
        assert os.path.exists(q)
        assert 'corrupt-' in os.path.basename(q)
        with open(q, 'rb') as f:
            assert f.read() == b'corrupt'

    def test_quarantine_missing_is_noop(self, tmp_path):
        assert quarantine_corrupt(str(tmp_path / 'nope.joblib')) is None


class TestBehavioralBaselineLifecycle:
    def test_missing_refit_persisted_then_loaded(self, tmp_path):
        from core.behavioral_baseline import ensure_fitted_baseline
        path = str(tmp_path / 'behavioral_baselines.joblib')
        mgr = ensure_fitted_baseline(path)
        assert mgr.artifact_status == 'refit_fallback_missing'
        assert os.path.exists(path)
        mgr2 = ensure_fitted_baseline(path)
        assert mgr2.artifact_status == 'loaded'

    def test_corrupt_quarantined_and_repersisted(self, tmp_path):
        from core.behavioral_baseline import ensure_fitted_baseline
        path = str(tmp_path / 'behavioral_baselines.joblib')
        _corrupt(path)
        mgr = ensure_fitted_baseline(path)
        assert mgr.artifact_status == 'refit_fallback_corrupt'
        # Original bytes preserved under quarantine name, not overwritten
        quarantines = [f for f in os.listdir(tmp_path)
                       if '.corrupt-' in f]
        assert len(quarantines) == 1
        # Fresh valid artifact now loads
        mgr2 = ensure_fitted_baseline(path)
        assert mgr2.artifact_status == 'loaded'


class TestSessionAnalyzerLifecycle:
    def test_missing_refit_persisted_then_loaded(self, tmp_path, normal_df):
        from core.session_analyzer import ensure_fitted_session_analyzer
        path = str(tmp_path / 'session_baselines.joblib')
        analyzer = ensure_fitted_session_analyzer(path=path,
                                                  train_df=normal_df)
        assert analyzer.baseline_manager.artifact_status == \
            'refit_fallback_missing'
        assert os.path.exists(path)
        analyzer2 = ensure_fitted_session_analyzer(path=path,
                                                   train_df=normal_df)
        assert analyzer2.baseline_manager.artifact_status == 'loaded'

    def test_corrupt_quarantined_and_repersisted(self, tmp_path, normal_df):
        from core.session_analyzer import ensure_fitted_session_analyzer
        path = str(tmp_path / 'session_baselines.joblib')
        _corrupt(path)
        analyzer = ensure_fitted_session_analyzer(path=path,
                                                  train_df=normal_df)
        assert analyzer.baseline_manager.artifact_status == \
            'refit_fallback_corrupt'
        assert any('.corrupt-' in f for f in os.listdir(tmp_path))
        analyzer2 = ensure_fitted_session_analyzer(path=path,
                                                   train_df=normal_df)
        assert analyzer2.baseline_manager.artifact_status == 'loaded'


class TestSequenceAnalyzerLifecycle:
    def test_missing_refit_persisted_then_loaded(self, tmp_path, normal_df):
        from core.sequence_detection import ensure_fitted_sequence_analyzer
        path = str(tmp_path / 'sequence_model.joblib')
        analyzer = ensure_fitted_sequence_analyzer(path=path,
                                                   train_df=normal_df)
        assert analyzer.artifact_status == 'refit_fallback_missing'
        assert os.path.exists(path)
        analyzer2 = ensure_fitted_sequence_analyzer(path=path,
                                                    train_df=normal_df)
        assert analyzer2.artifact_status == 'loaded'

    def test_corrupt_quarantined_and_repersisted(self, tmp_path, normal_df):
        from core.sequence_detection import ensure_fitted_sequence_analyzer
        path = str(tmp_path / 'sequence_model.joblib')
        _corrupt(path)
        analyzer = ensure_fitted_sequence_analyzer(path=path,
                                                   train_df=normal_df)
        assert analyzer.artifact_status == 'refit_fallback_corrupt'
        assert any('.corrupt-' in f for f in os.listdir(tmp_path))
        analyzer2 = ensure_fitted_sequence_analyzer(path=path,
                                                    train_df=normal_df)
        assert analyzer2.artifact_status == 'loaded'


class TestDetectorLifecycle:
    def test_missing_refit_persisted_then_loaded(self, tmp_path, monkeypatch):
        import models.anomaly_detector as ad
        path = str(tmp_path / 'anomaly_detector.joblib')
        monkeypatch.setattr(ad, 'get_model_path', lambda: path)
        detector = ad.ensure_model_exists()
        assert detector.artifact_status == 'refit_fallback_missing'
        assert os.path.exists(path)
        detector2 = ad.ensure_model_exists()
        assert detector2.artifact_status == 'loaded'

    def test_corrupt_quarantined_and_repersisted(self, tmp_path, monkeypatch):
        import models.anomaly_detector as ad
        path = str(tmp_path / 'anomaly_detector.joblib')
        _corrupt(path)
        monkeypatch.setattr(ad, 'get_model_path', lambda: path)
        detector = ad.ensure_model_exists()
        assert detector.artifact_status == 'refit_fallback_corrupt'
        assert any('.corrupt-' in f for f in os.listdir(tmp_path))
        detector2 = ad.ensure_model_exists()
        assert detector2.artifact_status == 'loaded'


class TestFitDataProvenance:
    """Refits must only ever use designated training-normal data."""

    def test_ensure_helpers_fit_on_normal_only(self, tmp_path, monkeypatch):
        calls = []
        import core.log_processor as lp
        real = lp.load_sample_dataset

        def spy(dataset_type='all'):
            calls.append(dataset_type)
            return real(dataset_type)

        monkeypatch.setattr(lp, 'load_sample_dataset', spy)

        from core.behavioral_baseline import ensure_fitted_baseline
        from core.session_analyzer import ensure_fitted_session_analyzer
        from core.sequence_detection import ensure_fitted_sequence_analyzer

        normal_df = real('normal')
        ensure_fitted_baseline(str(tmp_path / 'b.joblib'))
        ensure_fitted_session_analyzer(
            path=str(tmp_path / 's.joblib'), train_df=normal_df)
        ensure_fitted_sequence_analyzer(
            path=str(tmp_path / 'q.joblib'), train_df=normal_df)

        # Every dataset load triggered by a refit must be 'normal'.
        # (train_df was supplied for session/sequence, so only the
        # behavioral fallback load is expected here.)
        assert all(c == 'normal' for c in calls)
