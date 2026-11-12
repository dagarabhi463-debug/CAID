"""Milestone 2.4 — benchmark split integrity & security-input tests."""

import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'scripts'))


class TestBenchmarkSplitIntegrity:
    """Leakage guards for scripts/benchmark_models.py splits."""

    @pytest.fixture(scope='class')
    def splits(self):
        from benchmark_models import load_labeled_data, make_splits
        return make_splits(load_labeled_data())

    def test_entity_disjoint_splits(self, splits):
        users = {k: set(v['user_id'].astype(str)) for k, v in splits.items()}
        keys = list(users)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                assert not (users[a] & users[b]), f'{a} leaks into {b}'

    def test_unsupervised_splits_are_normal_only(self, splits):
        for name in ('unsup_train', 'calibration'):
            labels = set(splits[name]['label'].unique())
            assert labels == {'normal'}, f'{name} has {labels}'

    def test_holdout_has_both_classes(self, splits):
        labels = set(splits['holdout']['label'].unique())
        assert {'normal', 'suspicious', 'high_risk'} <= labels

    def test_deterministic_splits(self):
        from benchmark_models import load_labeled_data, make_splits
        a = make_splits(load_labeled_data())
        b = make_splits(load_labeled_data())
        for k in a:
            pd.testing.assert_frame_equal(
                a[k].reset_index(drop=True),
                b[k].reset_index(drop=True))


class _FakeUpload:
    def __init__(self, data: bytes, name='x.csv'):
        self._bio = io.BytesIO(data)
        self.name = name
        self.size = len(data)

    def read(self, *a):
        return self._bio.read(*a)

    def seek(self, *a):
        return self._bio.seek(*a)


class TestUploadValidationSecurity:
    """Malformed-input handling for the CSV upload path."""

    def test_oversized_file_rejected(self):
        from modules.log_analysis import validate_upload
        big = _FakeUpload(b'a' * (11 * 1024 * 1024))
        ok, msg = validate_upload(big)
        assert not ok and '10MB' in msg

    def test_non_csv_extension_rejected(self):
        from modules.log_analysis import validate_upload
        ok, msg = validate_upload(_FakeUpload(b'a,b\n1,2\n', 'x.exe'))
        assert not ok and 'CSV' in msg

    def test_empty_csv_rejected(self):
        from modules.log_analysis import validate_upload
        ok, _ = validate_upload(_FakeUpload(b''))
        assert not ok

    def test_non_utf8_decode_fails_cleanly(self):
        from modules.log_analysis import validate_upload
        bad = _FakeUpload(b'\xff\xfe\x00bad,csv\n'.decode(
            'latin1').encode('latin1'))
        ok, msg = validate_upload(bad)
        # Must not crash — returns (False, message) or raises only
        # inside the guarded try which validate_upload catches.
        assert ok is False or 'Error' in str(msg)

    def test_process_uploaded_file_invalid_utf8_raises(self):
        """Documented behavior: strict decode raises (page catches it)."""
        from modules.log_analysis import process_uploaded_file
        bad = _FakeUpload('café'.encode('latin1'))
        with pytest.raises(UnicodeDecodeError):
            process_uploaded_file(bad)

    def test_formula_injection_neutralized_in_export(self):
        from modules.export import _sanitize_csv_frame
        df = pd.DataFrame({'a': ['=cmd()', '+1-2', '@x', 'safe',
                                 '-danger', 'normal']})
        out = _sanitize_csv_frame(df)
        for v in out['a']:
            if isinstance(v, str) and v and v[0] in '=-+@':
                pytest.fail(f'unsanitized formula cell: {v!r}')


class TestUploadV2Pipeline:
    """AUD-003: uploaded CSVs must use the same v2_0 detection pipeline
    as scenario analysis (session + sequence channels included)."""

    def test_uploaded_data_gets_v2_channels(self):
        from core.log_processor import load_sample_dataset
        from modules.log_analysis import analyze_uploaded_data
        df = load_sample_dataset('normal').head(30).copy()
        out = analyze_uploaded_data(df)
        for col in ('combined_score', 'risk_level', 'operating_mode',
                    'missing_channels', 'session_deviation_score',
                    'sequence_deviation_score',
                    'detection_calibration_status',
                    'risk_calibration_status', 'threshold_version'):
            assert col in out.columns, col
        assert out['combined_score'].between(0, 1).all()

    def test_uploaded_data_without_optional_columns(self):
        """Uploads lacking session_id/auth_result must still score via
        reconstructed sessions (missing channels reported honestly)."""
        from core.log_processor import load_sample_dataset
        from modules.log_analysis import analyze_uploaded_data
        df = load_sample_dataset('normal').head(20).copy()
        df = df.drop(columns=[c for c in ('session_id', 'auth_result',
                                          'label') if c in df.columns])
        out = analyze_uploaded_data(df)
        assert 'session_deviation_score' in out.columns
        assert 'sequence_deviation_score' in out.columns
        assert 'operating_mode' in out.columns
