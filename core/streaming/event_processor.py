"""
Streaming detection processor — Milestone 2.5.

Integrates the streaming pipeline with the EXISTING v2_0 detection stack:
the same persisted artifacts (loaded via the AUD-002 ``ensure_*`` helpers),
the same fixed fusion weights, the same fingerprint-checked calibration
artifact, and the same score semantics as the batch pipeline
(``modules/session_intelligence.py::analyze_session_intelligence``).

Guarantees:
- No model is ever refit during streaming inference.
- Scores are calibrated anomaly scores, NOT probabilities.
- Retrospective correlation never feeds back into per-event scores.
- The batch pipeline is untouched.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.log_processor import load_sample_dataset
from core.detection_rules import DetectionRulesEngine
from core.detection_engine import DetectionEngine
from core.risk_engine import RiskEngine
from core.calibration import (
    apply_to_engine as apply_risk_calibration,
    apply_detection_calibration,
    component_fingerprint,
    detector_fingerprint,
)
from core.behavioural_analyzer import ensure_fitted_analyzer
from core.session_analyzer import ensure_fitted_session_analyzer
from core.sequence_detection import ensure_fitted_sequence_analyzer
from models.anomaly_detector import ensure_model_exists
from core.streaming.session_state import IncrementalSessionTracker
from core.streaming.stream_monitor import MetricsRegistry


def _parse_iso(ts) -> Optional[datetime]:
    if isinstance(ts, datetime):
        return ts
    try:
        text = str(ts)
        if text.endswith('Z'):
            text = text[:-1] + '+00:00'
        return datetime.fromisoformat(text)
    except Exception:
        pass
    try:
        return pd.to_datetime(ts).to_pydatetime()
    except Exception:
        return None


def _delivery_key(event_id: str, replay_of: str = '') -> str:
    """Queue-processing dedup key: event identity plus replay lineage."""
    return f'{event_id or ""}|{replay_of or ""}'


def _event_delivery_key(ev) -> str:
    return _delivery_key(ev.event_id, getattr(ev, 'replay_of', ''))


@dataclass
class DetectionArtifacts:
    """Loaded, frozen detection stack shared by streaming inference."""
    detector: object
    risk_engine: RiskEngine
    rules_engine: DetectionRulesEngine
    behavioral: object
    session_analyzer: object
    sequence_analyzer: object
    detection_engine: DetectionEngine
    tracker: IncrementalSessionTracker
    artifact_statuses: Dict[str, Optional[str]] = field(default_factory=dict)
    calibration_statuses: Dict[str, str] = field(default_factory=dict)


class StreamingDetectionProcessor:
    """
    Per-event (micro-batch) detection dispatch over tracker-emitted rows.

    Channel execution order mirrors the batch pipeline. Batch-invariant
    channels (ML, rules, behavioral, risk) run over the micro-batch
    DataFrame; session/sequence scores arrive precomputed from the
    IncrementalSessionTracker (causal state); fusion runs per event via
    ``DetectionEngine.process_event`` semantics through
    ``process_dataframe`` on the micro-batch.
    """

    def __init__(self, artifacts: DetectionArtifacts,
                 monitor: Optional[MetricsRegistry] = None,
                 late_event_policy: str = 'dlq'):
        """
        Args:
            late_event_policy: 'dlq' (default) dead-letters late events;
                'rescore' additionally scores them out-of-band as
                isolated rows flagged ``is_late_event`` — committed
                scores are never rewritten, the rescored row is purely
                additive evidence for the analyst.
        """
        self.artifacts = artifacts
        self.monitor = monitor or MetricsRegistry()
        self.late_event_policy = (late_event_policy
                                  if late_event_policy in ('dlq', 'rescore')
                                  else 'dlq')

    # ------------------------------------------------------------------
    # Artifact loading (identical to analyze_session_intelligence)
    # ------------------------------------------------------------------

    @classmethod
    def load_artifacts(cls,
                       tracker_kwargs: Optional[Dict] = None
                       ) -> DetectionArtifacts:
        detector = ensure_model_exists()

        risk = RiskEngine()
        apply_risk_calibration(risk, detector=detector)
        rules_engine = DetectionRulesEngine()
        behavioral = ensure_fitted_analyzer()

        normal = load_sample_dataset('normal')
        session = ensure_fitted_session_analyzer(train_df=normal)
        sequence = ensure_fitted_sequence_analyzer(train_df=normal)

        tracker = IncrementalSessionTracker(
            session.baseline_manager, sequence,
            **(tracker_kwargs or {}))

        artifact_statuses = {
            'detector': getattr(detector, 'artifact_status', None),
            'feature_baseline': getattr(
                getattr(detector, 'baseline_manager', None),
                'artifact_status', None),
            'behavioral_baseline': getattr(
                getattr(behavioral, 'baseline_manager', None),
                'artifact_status', None),
            'session_baseline': getattr(
                getattr(session, 'baseline_manager', None),
                'artifact_status', None),
            'sequence_model': getattr(sequence, 'artifact_status', None),
        }

        engine = DetectionEngine(channel_profile='v2_0')
        cal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', '..', 'models', 'saved_models',
                               'calibration')
        cal_path = os.path.join(
            cal_dir,
            f'v2_0_thresholds_{detector_fingerprint(detector)}.joblib')
        fingerprints = {
            'detector': detector_fingerprint(detector),
            'feature_baseline': component_fingerprint(
                detector.baseline_manager, 'feature_baseline'),
            'behavioral_baseline': component_fingerprint(
                behavioral.baseline_manager, 'behavioral_baseline'),
            'session_baseline': component_fingerprint(
                session.baseline_manager, 'session_baseline'),
            'sequence_model': component_fingerprint(sequence,
                                                    'sequence_model'),
        }
        apply_detection_calibration(engine, detector=detector, path=cal_path,
                                    fingerprints=fingerprints)

        return DetectionArtifacts(
            detector=detector, risk_engine=risk, rules_engine=rules_engine,
            behavioral=behavioral, session_analyzer=session,
            sequence_analyzer=sequence, detection_engine=engine,
            tracker=tracker, artifact_statuses=artifact_statuses,
            calibration_statuses={
                'risk': risk.calibration_status,
                'detection': engine.calibration_status,
                'threshold_version': engine.threshold_version
                or risk.threshold_version,
            })

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    _OUTPUT_COLUMNS = (
        'event_id', 'ingested_at_utc', 'source', 'replay_of',
        'scored_at_utc', 'detection_latency_ms',
        'timestamp', 'user_id', 'session_id', 'event_type',
        'combined_score', 'risk_level', 'operating_mode',
        'channels_available', 'missing_channels', 'channel_weights',
        'ml_anomaly', 'ml_anomaly_score', 'anomaly_score_normalized',
        'is_anomaly', 'rule_score', 'rules_triggered', 'triggered_rules',
        'behavioral_deviation', 'behavioral_score_missing',
        'session_deviation', 'session_score_missing',
        'sequence_deviation', 'sequence_score_missing',
        'detection_id', 'detection_explanation', 'recommended_actions',
        'structured_evidence', 'risk_score', 'risk_calibration_status',
        'detection_calibration_status', 'threshold_version',
        'artifact_statuses',
        'is_late_event', 'late_ordering_detail',
    )

    def process_rows(self, rows: List[Dict]) -> List[Dict]:
        """
        Score tracker-emitted rows through the full channel stack.

        Returns one output record per input row (same order).
        """
        if not rows:
            return []
        art = self.artifacts
        scored_at = datetime.now(timezone.utc).isoformat()

        df = pd.DataFrame(rows)

        # --- ML channel (batch-invariant: stored baselines only) ---------
        df = art.detector.predict(df)

        # --- Risk channel (per-row; fixed calibrated thresholds) ---------
        from utils.row_view import RowView
        risk_scores, risk_factors, risk_levels, risk_evidence = [], [], [], []
        for rec in df.to_dict('records'):
            row = RowView(rec)
            score, factors = art.risk_engine.calculate_risk_score(row)
            risk_scores.append(score)
            risk_factors.append(factors)
            risk_levels.append(
                art.risk_engine.classify_risk(score).value)
            row_with_factors = row.copy()
            row_with_factors['risk_factors'] = factors
            risk_evidence.append(
                art.risk_engine._generate_evidence(row_with_factors))
        df['risk_score'] = risk_scores
        df['risk_factors'] = risk_factors
        df['risk_level'] = risk_levels
        df['risk_evidence'] = risk_evidence

        # --- Rules channel (per-row) --------------------------------------
        df = art.rules_engine.process_events(df)

        # --- Behavioral channel (per-row vs frozen baselines) -------------
        df = art.behavioral.analyze_user_behavior(df)

        # --- Fusion: DetectionEngine v2_0 with fingerprint-checked --------
        # calibration (identical weights/thresholds as batch).
        df = art.detection_engine.process_dataframe(df)
        df['detection_id'] = [
            f"DET-{eid}" for eid in df.get('event_id',
                                           pd.Series([''] * len(df)))]
        df['risk_calibration_status'] = art.risk_engine.calibration_status
        df['detection_calibration_status'] = \
            art.detection_engine.calibration_status
        df['threshold_version'] = (art.detection_engine.threshold_version
                                   or art.risk_engine.threshold_version)

        out: List[Dict] = []
        scored_dt = _parse_iso(scored_at)
        for rec in df.to_dict('records'):
            row = RowView(rec)
            rec = {c: row.get(c) for c in self._OUTPUT_COLUMNS if c in row.index}
            ingested = _parse_iso(rec.get('ingested_at_utc'))
            scored = scored_dt
            if ingested is not None and scored is not None:
                rec['detection_latency_ms'] = (
                    scored - ingested).total_seconds() * 1000.0
                self.monitor.detection_latency(
                    rec['detection_latency_ms'] / 1000.0)
            else:
                rec['detection_latency_ms'] = None
            rec['scored_at_utc'] = scored_at
            rec['artifact_statuses'] = dict(art.artifact_statuses)
            # NaN -> None for clean JSON serialization
            for k, v in list(rec.items()):
                if isinstance(v, float) and not np.isfinite(v):
                    rec[k] = None
            out.append(rec)
        return out

    # ------------------------------------------------------------------
    # Worker loop (reliable processing: retry/backoff/DLQ/ack-after-commit)
    # ------------------------------------------------------------------

    async def run_worker(self, queue, sink, dlq,
                         batch_size: int = 100,
                         dequeue_timeout_s: float = 0.5,
                         sweep_interval_s: float = 0.05,
                         max_attempts: int = 5,
                         base_backoff_s: float = 0.1,
                         max_backoff_s: float = 30.0,
                         claim_pending_interval_s: float = 30.0,
                         committed_ids=None,
                         checkpoint_path: Optional[str] = None,
                         checkpoint_interval_s: float = 30.0,
                         artifact_fingerprints: Optional[Dict[str, str]] = None,
                         max_batch_latency_s: float = 0.1,
                         stop_event=None):
        """
        Reliable consumer loop.

        - Dequeue -> tracker.ingest (ordering/late policy) -> process_rows
          -> sink.write_batch + sink.commit -> queue.ack (ack ONLY after
          durable commit).
        - Events still buffered in the tracker are tracked in
          ``pending_acks`` and acked when their row is emitted+committed.
        - Transient failures (sink/commit) retry with exponential backoff;
          exhausted retries and deterministic processing errors go to the
          DLQ (which is itself the durable commit for failures -> then ack).
        - At-least-once delivery is made effectively-once by
          ``committed_ids``: a bounded LRU of event_ids already accepted
          into the pipeline. Redelivered duplicates are acked immediately
          without re-scoring (deterministic event_id identity). The set is
          in-memory; callers needing cross-restart dedup may inject a
          persisted instance — redelivery after restart otherwise
          re-scores deterministically (identical rows; consumers dedup by
          ``event_id``).
        - Sweep-emitted rows are accumulated into a micro-batch buffer and
          committed at ``batch_size`` rows or ``max_batch_latency_s``
          whichever comes first — bounding added latency while avoiding
          per-sweep pandas/fsync overhead (throughput fix, M2.5.1).
        """
        import asyncio
        import random

        from core.streaming.event_schema import (
            BoundedDedupLRU, StreamingEvent)

        if committed_ids is None:
            committed_ids = BoundedDedupLRU()

        from core.streaming.event_checkpoint import StreamCheckpoint

        checkpoint = (StreamCheckpoint(checkpoint_path)
                      if checkpoint_path else None)
        pending_acks: Dict[str, str] = {}   # delivery_key -> queue_id
        last_sweep = time.monotonic()
        last_claim = time.monotonic()
        last_checkpoint = time.monotonic()
        score_buf: List[Dict] = []
        score_buf_since: Optional[float] = None

        async def _flush_score_buf(force: bool = False) -> None:
            nonlocal score_buf, score_buf_since
            if not score_buf:
                return
            if (not force and len(score_buf) < batch_size
                    and score_buf_since is not None
                    and (time.monotonic() - score_buf_since)
                    < max_batch_latency_s):
                return
            rows = score_buf
            score_buf = []
            score_buf_since = None
            await self._commit_rows(
                rows, queue, sink, dlq, pending_acks,
                max_attempts=max_attempts, base_backoff_s=base_backoff_s,
                max_backoff_s=max_backoff_s)
        if checkpoint is not None:
            self._restore_checkpoint(
                checkpoint, committed_ids, pending_acks)
            last_checkpoint = time.monotonic()

        async def _maybe_checkpoint(force: bool = False) -> None:
            nonlocal last_checkpoint
            if checkpoint is None:
                return
            now = time.monotonic()
            if not force and now - last_checkpoint < checkpoint_interval_s:
                return
            await self._save_checkpoint(
                checkpoint, queue, committed_ids, pending_acks,
                artifact_fingerprints=artifact_fingerprints)
            last_checkpoint = time.monotonic()

        while True:
            if stop_event is not None and stop_event.is_set():
                break
            now = time.monotonic()

            # Periodic pending-entry recovery (Redis XAUTOCLAIM / asyncio
            # inflight re-delivery) and idle tracker sweep.
            if now - last_claim >= claim_pending_interval_s:
                last_claim = now
                try:
                    claimed = await queue.claim_pending()
                except Exception:
                    claimed = []
                if claimed:
                    await self._handle_messages(
                        claimed, queue, sink, dlq, pending_acks,
                        committed_ids, max_attempts=max_attempts,
                        base_backoff_s=base_backoff_s,
                        max_backoff_s=max_backoff_s)
                    await _maybe_checkpoint()
            if now - last_sweep >= sweep_interval_s:
                last_sweep = now
                rows, _late = self.artifacts.tracker.sweep()
                if rows:
                    if score_buf_since is None:
                        score_buf_since = time.monotonic()
                    score_buf.extend(rows)
                await _flush_score_buf()
                self.monitor.queue_depth(await self._safe_depth(queue))
                await _maybe_checkpoint()

            try:
                msgs = await queue.dequeue(batch_size, dequeue_timeout_s)
            except Exception:
                msgs = []
            if not msgs:
                continue
            await self._handle_messages(
                msgs, queue, sink, dlq, pending_acks, committed_ids,
                max_attempts=max_attempts, base_backoff_s=base_backoff_s,
                max_backoff_s=max_backoff_s)
            await _maybe_checkpoint()

        # Shutdown: flush tracker, commit, ack everything remaining.
        rows, _late = self.artifacts.tracker.flush_all()
        if rows:
            score_buf.extend(rows)
        await _flush_score_buf(force=True)
        if pending_acks:
            # Events dequeued but never emitted — redeliver by NOT acking;
            # they remain pending for the next consumer.
            pending_acks.clear()
        await _maybe_checkpoint(force=True)

    @staticmethod
    def _worker_state(committed_ids, pending_acks: Dict[str, str]) -> Dict:
        return {
            'committed_ids': committed_ids.snapshot_keys(),
            'pending_acks': dict(pending_acks),
        }

    async def _queue_meta(self, queue) -> Dict:
        meta = {'implementation': type(queue).__name__}
        for name in ('depth', 'pending_count'):
            fn = getattr(queue, name, None)
            if fn is None:
                continue
            try:
                value = fn()
                if hasattr(value, '__await__'):
                    value = await value
                meta[name] = value
            except Exception:
                meta[name] = None
        return meta

    async def _save_checkpoint(self, checkpoint, queue, committed_ids,
                               pending_acks, *, artifact_fingerprints=None):
        try:
            checkpoint.save(
                queue_meta=await self._queue_meta(queue),
                tracker=self.artifacts.tracker,
                artifact_fingerprints=artifact_fingerprints or {},
                monitor_snapshot=self.monitor.snapshot(),
                extra={'worker_state': self._worker_state(
                    committed_ids, pending_acks)})
        except Exception:
            # Checkpointing is operational state; never break detection.
            self.monitor.gauge('caid_checkpoint_save_errors', 1.0)

    def _restore_checkpoint(self, checkpoint, committed_ids, pending_acks):
        try:
            restored = checkpoint.restore_tracker(self.artifacts.tracker)
            meta = checkpoint.load() or {}
            worker_state = (meta.get('extra') or {}).get('worker_state') or {}
            committed_ids.restore_keys(worker_state.get('committed_ids'))
            pending_acks.update(worker_state.get('pending_acks') or {})
            if restored:
                self.monitor.gauge('caid_checkpoint_restored', 1.0)
        except Exception:
            self.monitor.gauge('caid_checkpoint_restore_errors', 1.0)

    async def _handle_messages(self, msgs, queue, sink, dlq,
                               pending_acks, committed_ids,
                               max_attempts: int = 5,
                               base_backoff_s: float = 0.1,
                               max_backoff_s: float = 30.0):
        import asyncio
        import random

        from core.streaming.event_schema import StreamingEvent

        events = []
        bad = []
        for m in msgs:
            try:
                events.append((m, StreamingEvent.from_dict(m.payload)))
            except Exception as e:
                bad.append((m, f'payload_decode:{e}'))

        # Duplicate delivery (at-least-once queues): the deterministic
        # event_id was already accepted into the pipeline -> ack without
        # re-scoring. This also prevents pending_acks key collisions.
        dup_qids = []
        fresh = []
        for m, ev in events:
            dkey = _event_delivery_key(ev)
            if committed_ids.check_and_add(dkey):
                dup_qids.append(m.queue_id)
                self.monitor.event_duplicate()
            else:
                fresh.append((m, ev, dkey))
        if dup_qids:
            await queue.ack(dup_qids)

        # Undecodable payloads -> DLQ immediately.
        for m, reason in bad:
            dlq.write({'queue_id': m.queue_id, 'payload': m.payload,
                       'reason': reason, 'stage': 'decode'})
            dlq.commit()
            await queue.ack([m.queue_id])
            self.monitor.dlq_depth(dlq.depth())
            self.monitor.dead_letter(reason.split(':')[0])

        rows, late_events = [], []
        event_queue_ids = {}
        for m, ev, dkey in fresh:
            event_queue_ids[dkey] = m.queue_id
            # Register the delivery mapping up-front: the event may stay
            # buffered in the tracker for ordering; it is acked only when
            # its scored row is durably committed (zero-loss contract).
            pending_acks[dkey] = m.queue_id
            try:
                r, l = self.artifacts.tracker.ingest(ev)
            except Exception as e:
                pending_acks.pop(dkey, None)
                dlq.write({'event': ev.to_dict(),
                           'reason': 'processing_error',
                           'stage': 'tracker', 'error': str(e),
                           'queue_id': m.queue_id})
                dlq.commit()
                await queue.ack([m.queue_id])
                self.monitor.dlq_depth(dlq.depth())
                self.monitor.dead_letter('processing_error')
                continue
            rows.extend(r)
            late_events.extend(l)

        # Late events: default policy dead-letters them (reason
        # late_event), then acks. The tracker records the emitted
        # ordering frontier for traceability. With late_event_policy=
        # 'rescore' the event is ALSO scored out-of-band as an isolated
        # row flagged is_late_event — additive evidence only; the
        # already-committed causal position is never rewritten. The
        # pending_acks entry is kept so the queue ack still happens
        # only after the rescored row is durably committed.
        late_ids = []
        late_details = getattr(self.artifacts.tracker, 'late_details', {})
        rescored_late = []
        for ev in late_events:
            dkey = _event_delivery_key(ev)
            qid = event_queue_ids.get(dkey)
            detail = late_details.pop(ev.event_id, {})
            if self.late_event_policy == 'rescore':
                row = ev.to_dict()
                row['is_late_event'] = True
                row['late_ordering_detail'] = detail
                rescored_late.append(row)
                continue
            pending_acks.pop(dkey, None)
            dlq.write({'event': ev.to_dict(), 'reason': 'late_event',
                       'stage': 'ordering',
                       'detail': detail,
                       'queue_id': qid})
            if qid:
                late_ids.append(qid)
        if late_ids:
            dlq.commit()
            await queue.ack(late_ids)
            self.monitor.dlq_depth(dlq.depth())
            self.monitor.dead_letter('late_event')
        if rescored_late:
            rows.extend(rescored_late)
            self.monitor.incr('caid_late_events_rescored_total',
                              len(rescored_late))

        if not rows:
            # All events buffered for ordering — nothing to commit/ack yet.
            return

        await self._commit_rows(
            rows, queue, sink, dlq, pending_acks,
            max_attempts=max_attempts, base_backoff_s=base_backoff_s,
            max_backoff_s=max_backoff_s)

    async def _commit_rows(self, rows, queue, sink, dlq, pending_acks,
                           max_attempts: int = 5,
                           base_backoff_s: float = 0.1,
                           max_backoff_s: float = 30.0):
        """Score -> durable sink commit -> ack, with retry/backoff/DLQ.

        Shared by the dequeue path (``_handle_messages``) and the
        micro-batch sweep buffer. Acks exactly the queue entries whose
        rows were emitted+committed; queue ids are resolved through
        ``pending_acks`` (delivery_key -> queue_id).
        """
        import asyncio
        import random

        scored = None
        last_err = None
        attempt = 0
        for attempt in range(1, max(1, int(max_attempts)) + 1):
            try:
                scored = self.process_rows(rows)
                sink.write_batch(scored)
                sink.commit()
                last_err = None
                break
            except Exception as e:  # transient sink/processing failure
                last_err = e
                self.monitor.retry('processing')
                await asyncio.sleep(min(
                    float(max_backoff_s),
                    float(base_backoff_s) * (2 ** (attempt - 1))
                    * (0.5 + random.random())))
        if last_err is not None:
            # Retry exhausted -> DLQ the affected events, then ack.
            ids = []
            for row in rows:
                dkey = _delivery_key(row.get('event_id'),
                                     row.get('replay_of', ''))
                qid = pending_acks.get(dkey)
                dlq.write({'event': row, 'reason': 'retry_exhausted',
                           'stage': 'processing', 'error': str(last_err),
                           'attempts': attempt, 'queue_id': qid})
                if qid:
                    ids.append(qid)
                    pending_acks.pop(dkey, None)
            dlq.commit()
            if ids:
                await queue.ack(ids)
            self.monitor.dlq_depth(dlq.depth())
            self.monitor.dead_letter('retry_exhausted')
            return None

        # Durable commit succeeded -> ack exactly the queue entries whose
        # rows were emitted+committed. Events still buffered in the tracker
        # remain in pending_acks and are acked by a later sweep/flush.
        emitted_qids = []
        for row in scored:
            dkey = _delivery_key(row.get('event_id'),
                                 row.get('replay_of', ''))
            if dkey in pending_acks:
                emitted_qids.append(pending_acks.pop(dkey))
        if emitted_qids:
            await queue.ack(emitted_qids)
        self.monitor.queue_depth(await self._safe_depth(queue))
        return scored

    async def _safe_depth(self, queue) -> int:
        try:
            return await queue.depth()
        except Exception:
            return -1
