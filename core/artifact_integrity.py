"""Artifact integrity helpers for CAID persisted model/baseline files.

Provides SHA-256 checksum sidecars (``<artifact>.sha256``) so tampered
or corrupted artifacts are detected at load time instead of silently
producing wrong scores. Verification is best-effort additive: artifacts
without a sidecar (written before this mechanism existed) still load;
artifacts WITH a sidecar must match or the caller decides how to fail
(the ``ensure_*`` helpers treat load failures as corrupt -> quarantine
+ designated-data refit, which is the desired integrity response).

Filesystem protection: ``restrict_permissions`` tightens artifact file
permissions where the OS supports it (best-effort, no-op on failure).
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Optional

_SIGN_ENV = 'CAID_ARTIFACT_KEY'


def _signing_key() -> Optional[bytes]:
    """HMAC key for artifact signing; None when CAID_ARTIFACT_KEY unset.

    When a key IS configured, sidecars become signatures: an attacker
    who can write the artifact file but not the key cannot forge a
    valid checksum. Without a key the sidecar remains a plain
    corruption/tamper-evidence checksum (still detects accidental
    corruption and casual modification).
    """
    key = os.environ.get(_SIGN_ENV, '')
    return key.encode('utf-8') if key else None


def _mac(path: str) -> str:
    return hmac.new(_signing_key() or b'caid-artifact-integrity',
                    sha256_file(path).encode('ascii'),
                    hashlib.sha256).hexdigest()


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def checksum_path(path: str) -> str:
    return path + '.sha256'


def write_checksum(path: str) -> str:
    """Write ``<path>.sha256`` with the file digest (+HMAC when keyed)."""
    digest = sha256_file(path)
    payload = digest
    if _signing_key() is not None:
        payload = f'{digest}:{_mac(path)}'
    with open(checksum_path(path), 'w', encoding='utf-8') as f:
        f.write(payload + '\n')
    restrict_permissions(checksum_path(path))
    return digest


def verify_checksum(path: str) -> bool:
    """True when no sidecar exists (unprotected legacy artifact) or the
    sidecar verifies. With CAID_ARTIFACT_KEY set the HMAC signature is
    mandatory; without it only the plain digest is compared (legacy
    sidecars remain valid)."""
    sidecar = checksum_path(path)
    if not os.path.exists(sidecar):
        return True
    try:
        with open(sidecar, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        parts = content.split(':')
        expected = parts[0]
        digest = sha256_file(path)
        if expected != digest:
            return False
        if len(parts) > 1:
            # Signed sidecar: signature must verify under the key.
            key = _signing_key()
            if key is None:
                # Artifact was signed but no key is configured here —
                # fail closed rather than silently downgrade to unsigned.
                return False
            mac = hmac.new(key, digest.encode('ascii'),
                           hashlib.sha256).hexdigest()
            return hmac.compare_digest(parts[1], mac)
        # Unsigned sidecar: acceptable only when no key is configured
        # (mixed deployments would otherwise fail closed on legacy files).
        return _signing_key() is None
    except (OSError, IndexError):
        return False


def restrict_permissions(path: str,
                         mode: Optional[int] = None) -> None:
    """Best-effort permission tightening for sensitive files.

    Files get owner-only read/write (0o600); on Windows chmod only
    toggles the read-only flag, which is still applied harmlessly.
    Never raises — permission hardening must not break artifact I/O.
    """
    try:
        if mode is None:
            mode = 0o600 if os.path.isfile(path) else 0o700
        os.chmod(path, mode)
    except (OSError, NotImplementedError):
        pass
