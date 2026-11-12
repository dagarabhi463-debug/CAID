"""Streaming pipeline tests — parity, recovery, backpressure (M2.5)."""

import asyncio
import json
import os

import numpy as np
import pandas as pd
import pytest

from core.log_processor import load_sample_dataset
from core.streaming.event_schema import StreamingEvent, validate_event
from core.streaming.event_queue import AsyncioEventQueue
from core.streaming.event_ingestion import EventIngestor
from core.streaming.result_sink import JsonlResultSink
from core.streaming.event_replay import DeadLetterQueue, replay_dlq_async
from core.streaming.event_processor import StreamingDetectionProcessor

SCORE_COLS = [
    'combined_score', 'ml_anomaly_score', 'anomaly_score_normalized',
    'behavioral_deviation', 'session_deviation', 'sequence_deviation',
    'risk_score', 'rule_score', 'operating_mode', 'risk_level',
]


def _events_from_df(df, source='parity'):
    events = []
    for _, r in df.iterrows():
        raw = {c: r[c] for c in df.columns}
        raw.pop('label', None)
        ev, reason = validate_event(raw, source=source)
        assert ev is not None, reason
        events.append(ev)
    return events


def _batch_df(df):
    """Batch input matching stream normalization: ISO ts, float numerics,
    all log columns present (no label)."""
    out = df.copy()
    out['timestamp'] = pd.to_datetime(out['timestamp'], errors='coerce') \
        .dt.strftime('%Y-%m-%dT%H:%M:%S')
    for c in ('courses_accessed', 'accounts_accessed', 'api_requests',
              'data_mb', 'token_age_minutes', 'after_hours'):
        out[c] = pd.to_numeric(out[c], errors='coerce').astype(float)
    out['after_hours'] = out['after_hours'].astype(int)
    for c in ('role', 'source_ip', 'session_id', 'auth_result'):
        if c not in out.columns:
            out[c] = ''
        out[c] = out[c].fillna('').astype(str)
    out = out.drop(columns=['label'], errors='ignore')
    return out


@pytest.fixture(scope='module')
def artifacts():
    art = StreamingDetectionProcessor.load_artifacts(
        tracker_kwargs={'reorder_window_seconds': 0.0,
                        'max_hold_seconds': 0.0})
    return StreamingDetectionProcessor(art)


class TestParity:
    def test_stream_scores_match_batch(self, artifacts):
        raw = load_sample_dataset('all')
        batch_in = _batch_df(raw)
        # Feed events in per-user timestamp order (the order the batch
        # reconstructor effectively uses); stable sort keeps df order for
        # ties so batch _orig tie-breaks still align.
        batch_in = batch_in.sort_values(['user_id', 'timestamp'],
                                        kind='mergesort').reset_index(drop=True)
        events = _events_from_df(batch_in)

        tracker = artifacts.artifacts.tracker
        rows = []
        for ev in events:
            r, _late = tracker.ingest(ev)
            rows.extend(r)
        r2, _ = tracker.flush_all()
        rows.extend(r2)
        assert len(rows) == len(events)

        stream_scored = artifacts.process_rows(rows)
        stream_by_id = {r['event_id']: r for r in stream_scored}

        # Batch pipeline on identical input.
        from modules.session_intelligence import analyze_session_intelligence
        batch_out = analyze_session_intelligence(batch_in)

        # Compare per-event on score-contract columns.
        mismatches = []
        for i, ev in enumerate(events):
            brow = batch_out.iloc[i]
            srow = stream_by_id[ev.event_id]
            for col in SCORE_COLS:
                bv = brow.get(col)
                sv = srow.get(col)
                if isinstance(bv, float) and np.isnan(bv):
                    bv = None
                if isinstance(sv, float) and (sv is None or
                                              not np.isfinite(sv)):
                    sv = None
                if isinstance(bv, (int, float, np.floating)) and \
                        isinstance(sv, (int, float)):
                    if not np.isclose(float(bv), float(sv), atol=1e-9,
                                      equal_nan=True):
                        mismatches.append((i, col, bv, sv))
                elif str(bv) != str(sv):
                    mismatches.append((i, col, bv, sv))
        assert not mismatches, mismatches[:10]


class TestIngestDedup:
    def test_duplicate_event_id_skipped(self, tmp_path):
        async def run():
            q = AsyncioEventQueue(max_depth=100)
            ing = EventIngestor(q, source='t')
            raw = {'timestamp': '2024-01-01 00:00:00', 'user_id': 'u1',
                   'role': 'student', 'source_ip': '192.0.2.1',
                   'event_type': 'login', 'courses_accessed': 1,
                   'accounts_accessed': 1, 'api_requests': 1,
                   'data_mb': 1.0, 'token_age_minutes': 1, 'after_hours': 0}
            r1 = await ing.ingest(raw)
            r2 = await ing.ingest(raw)
            assert r1.status == 'accepted' and r2.status == 'duplicate'
            await q.close()
        asyncio.run(run())

    def test_label_rejected_end_to_end(self, tmp_path):
        async def run():
            q = AsyncioEventQueue(max_depth=100)
            ing = EventIngestor(q, source='t')
            raw = {'timestamp': '2024-01-01 00:00:00', 'user_id': 'u1',
                   'role': 'student', 'source_ip': '192.0.2.1',
                   'event_type': 'login', 'courses_accessed': 1,
                   'accounts_accessed': 1, 'api_requests': 1,
                   'data_mb': 1.0, 'token_age_minutes': 1, 'after_hours': 0,
                   'label': 'high_risk'}
            r = await ing.ingest(raw)
            assert r.status == 'rejected'
            assert await q.depth() == 0
            await q.close()
        asyncio.run(run())


class TestBackpressure:
    def test_producer_rejects_when_saturated(self, tmp_path):
        async def run():
            q = AsyncioEventQueue(max_depth=1)
            ing = EventIngestor(q, source='t', backpressure_timeout_s=0.05)
            raw = {'timestamp': '2024-01-01 00:00:00', 'user_id': 'u1',
                   'role': 'student', 'source_ip': '192.0.2.1',
                   'event_type': 'login', 'courses_accessed': 1,
                   'accounts_accessed': 1, 'api_requests': 1,
                   'data_mb': 1.0, 'token_age_minutes': 1, 'after_hours': 0}
            r1 = await ing.ingest(raw)
            r2 = await ing.ingest(dict(raw, user_id='u2'))
            assert r1.status == 'accepted'
            assert r2.status == 'rejected' and r2.reason == 'backpressure'
            await q.close()
        asyncio.run(run())


class TestWorkerIdempotency:
    def test_duplicate_delivery_scored_once(self, artifacts, tmp_path):
        """At-least-once redelivery of the same event_id must not produce
        a second scored row (Milestone 2.5.1)."""
        from core.streaming.event_queue import QueuedEvent
        from core.streaming.event_schema import BoundedDedupLRU

        async def run():
            q = AsyncioEventQueue(max_depth=10)
            sink = JsonlResultSink(str(tmp_path / 'sink.jsonl'))
            dlq = DeadLetterQueue(str(tmp_path / 'dlq.jsonl'))
            committed = BoundedDedupLRU()
            pending_acks = {}
            raw = {'timestamp': '2024-01-01 00:00:00', 'user_id': 'u_dup',
                   'role': 'student', 'source_ip': '192.0.2.1',
                   'event_type': 'login', 'courses_accessed': 1,
                   'accounts_accessed': 1, 'api_requests': 1,
                   'data_mb': 1.0, 'token_age_minutes': 1, 'after_hours': 0}
            ev, reason = validate_event(raw, source='t')
            assert ev is not None, reason
            payload = ev.to_dict()
            # Two distinct queue deliveries of the SAME event payload.
            m1 = QueuedEvent(queue_id='q1', payload=payload)
            m2 = QueuedEvent(queue_id='q2', payload=payload)
            await artifacts._handle_messages(
                [m1], q, sink, dlq, pending_acks, committed)
            await artifacts._handle_messages(
                [m2], q, sink, dlq, pending_acks, committed)
            assert sink.records_written == 1
            assert await q.pending_count() == 0
            snap = artifacts.monitor.snapshot()
            assert snap['counters'].get(
                'caid_events_duplicate_total') == 1
            sink.close()
            dlq.close()
            await q.close()
        asyncio.run(run())


class TestRecovery:
    def test_checkpoint_restore_tracker(self, artifacts, tmp_path):
        from core.streaming.event_checkpoint import StreamCheckpoint
        raw = load_sample_dataset('normal').head(30)
        events = _events_from_df(_batch_df(raw))
        tr = type(artifacts.artifacts.tracker)(
            artifacts.artifacts.session_analyzer.baseline_manager,
            artifacts.artifacts.sequence_analyzer,
            reorder_window_seconds=0.0, max_hold_seconds=0.0)
        for ev in events:
            tr.ingest(ev)
        cp = StreamCheckpoint(str(tmp_path / 'ckpt.json'))
        cp.save(tracker=tr,
                artifact_fingerprints={'detector': 'x'})

        tr2 = type(artifacts.artifacts.tracker)(
            artifacts.artifacts.session_analyzer.baseline_manager,
            artifacts.artifacts.sequence_analyzer,
            reorder_window_seconds=0.0, max_hold_seconds=0.0)
        assert cp.restore_tracker(tr2)
        assert tr2._arrival_seq == tr._arrival_seq
        assert len(tr2._sessions) == len(tr._sessions)

    def test_dlq_write_and_replay(self, artifacts, tmp_path):
        async def run():
            dlq_path = str(tmp_path / 'dlq.jsonl')
            dlq = DeadLetterQueue(dlq_path)
            q = AsyncioEventQueue(max_depth=100)
            ing = EventIngestor(q, source='t')
            ev_dict = {'timestamp': '2024-01-01 00:00:00', 'user_id': 'u1',
                       'role': 'student', 'source_ip': '192.0.2.1',
                       'event_type': 'login', 'courses_accessed': 1,
                       'accounts_accessed': 1, 'api_requests': 1,
                       'data_mb': 1.0, 'token_age_minutes': 1,
                       'after_hours': 0, 'event_id': 'orig1'}
            dlq.write({'event': ev_dict, 'reason': 'late_event'})
            dlq.commit()
            results = await replay_dlq_async(dlq_path, ing)
            assert len(results) == 1
            assert results[0].status == 'accepted'
            msgs = await q.dequeue(5, timeout_s=0.2)
            assert len(msgs) == 1
            assert msgs[0].payload['replay_of'] == 'orig1'
            await q.close()
            dlq.close()
        asyncio.run(run())
