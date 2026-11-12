"""User-facing error sanitization for the CAID dashboard.

Raw exception strings can leak filesystem paths, internal class names,
and library details. ``safe_error_message`` returns a generic message
by default; the exception type name is included only when
``CAID_DEBUG=1`` is set (development troubleshooting).
"""

from __future__ import annotations

import os
from typing import Optional


def safe_error_message(exc: BaseException,
                       generic: str = 'Operation failed',
                       debug: Optional[bool] = None) -> str:
    """Return a UI-safe error string.

    Args:
        exc: the caught exception (never rendered verbatim by default)
        generic: fallback message shown to the user
        debug: force debug output on/off; defaults to CAID_DEBUG env
    """
    if debug is None:
        debug = os.environ.get('CAID_DEBUG', '').lower() in ('1', 'true', 'yes')
    if debug:
        return f"{generic}: {type(exc).__name__}: {exc}"
    return generic
