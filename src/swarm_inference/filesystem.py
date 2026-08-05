"""Small cross-platform filesystem durability primitives."""

from __future__ import annotations

import errno
import os
import time
from pathlib import Path

_TRANSIENT_ERRNOS = {errno.EACCES, errno.EBUSY, errno.EPERM}
_TRANSIENT_WINDOWS_ERRORS = {5, 32, 33}


def _transient_replace_error(exc: OSError) -> bool:
    return exc.errno in _TRANSIENT_ERRNOS or getattr(exc, "winerror", None) in (
        _TRANSIENT_WINDOWS_ERRORS
    )


def replace_atomically(
    source: Path,
    destination: Path,
    *,
    retry_timeout_s: float = 1.0,
    initial_backoff_s: float = 0.005,
) -> None:
    """Replace a file, retrying only transient sharing/permission denials.

    Windows virus scanners and synchronized folders can briefly open a just-written
    destination without delete sharing. Retrying that specific condition preserves the
    atomic replace contract; all other errors still fail immediately.
    """

    if retry_timeout_s < 0 or initial_backoff_s <= 0:
        raise ValueError("atomic replace retry settings are invalid")
    deadline = time.monotonic() + retry_timeout_s
    backoff = initial_backoff_s
    while True:
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            remaining = deadline - time.monotonic()
            if not _transient_replace_error(exc) or remaining <= 0:
                raise
            time.sleep(min(backoff, remaining))
            backoff = min(backoff * 2, 0.1)


__all__ = ["replace_atomically"]
