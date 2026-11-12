"""Optional application-level access token for the CAID dashboard.

When ``CAID_UI_TOKEN`` is set, ``app.py`` requires the token before
rendering any page — a lightweight single-user gate for deployments
that expose the Streamlit UI beyond localhost. When unset (default),
no authentication is required and behavior is unchanged.

This is NOT multi-user authN/authZ: there are no roles, sessions are
per-browser, and the token is shared. For stronger requirements put the
UI behind a real reverse proxy (e.g. oauth2-proxy, nginx auth_request).
"""

from __future__ import annotations

import hmac
import os

ENV_VAR = 'CAID_UI_TOKEN'
_SESSION_KEY = '_caid_ui_token_ok'


def ui_token_required() -> bool:
    """True when CAID_UI_TOKEN is configured (non-empty)."""
    return bool(os.environ.get(ENV_VAR, '').strip())


def check_ui_token(token: str) -> bool:
    """Constant-time comparison against the configured token."""
    expected = os.environ.get(ENV_VAR, '')
    if not expected:
        return True
    return hmac.compare_digest(str(token or ''), expected)


def token_verified(session_state) -> bool:
    return bool(session_state.get(_SESSION_KEY))


def mark_verified(session_state) -> None:
    session_state[_SESSION_KEY] = True
