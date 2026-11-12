"""
Artifact lifecycle policy for CAID persisted model/baseline artifacts.

Unified policy for every ``ensure_fitted_*`` / ``ensure_model_exists``
helper (Milestone 2.4 closure, AUD-002):

- artifact loads cleanly            -> status ``loaded``
- artifact missing                  -> fit on designated training-normal
                                       data, persist, status
                                       ``refit_fallback_missing``
- artifact present but corrupt or   -> quarantine the corrupt file to
  incompatible                        ``<name>.corrupt-<UTC timestamp>``,
                                       fit on designated training-normal
                                       data, persist a fresh artifact,
                                       status ``refit_fallback_corrupt``

Corrupt artifacts are NEVER silently overwritten: the original bytes are
preserved under a quarantine name before any new artifact is written, and
the status is always reported to the caller (``artifact_status``
attribute) so operators can distinguish "loaded artifact" from
"just-fit fallback".

Fitting is only ever performed on designated training-normal data
(``load_sample_dataset('normal')``) — never on validation, holdout, or
inference data (leakage contract unchanged).
"""

import os
from datetime import datetime, timezone
from typing import Optional

STATUS_LOADED = 'loaded'
STATUS_REFIT_MISSING = 'refit_fallback_missing'
STATUS_REFIT_CORRUPT = 'refit_fallback_corrupt'


def quarantine_corrupt(path: str) -> Optional[str]:
    """
    Move a corrupt/incompatible artifact aside so it is never silently
    overwritten by an automatic refit.

    Args:
        path: artifact path that failed to load

    Returns:
        The quarantine path written, or None if there was nothing to move.
    """
    if not path or not os.path.exists(path):
        return None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    base, ext = os.path.splitext(path)
    quarantine = f'{base}.corrupt-{stamp}{ext}'
    # Avoid clobbering an existing quarantine from the same second.
    counter = 1
    while os.path.exists(quarantine):
        quarantine = f'{base}.corrupt-{stamp}-{counter}{ext}'
        counter += 1
    os.replace(path, quarantine)
    return quarantine
