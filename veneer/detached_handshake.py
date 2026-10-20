"""On-disk handshake protocol between the parent process and a detached
Veneer launcher. Shared by ``veneer.manage.start(detached=True)`` and
``veneer.detached_start.main``. Kept small and dependency-free so both
sides can import it without pulling in the rest of veneer-py."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


class HandshakeStatus:
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"


def write_handshake(path, payload):
    """Write ``payload`` as JSON to ``path`` atomically.

    Uses a temp file in the same directory followed by ``os.replace`` so a
    concurrent reader never sees a half-written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_handshake(path):
    """Return the handshake payload, or ``None`` if the file is absent or
    currently mid-write (invalid JSON). Never raises on missing/partial."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def wait_for_status(path, accept, timeout, poll_interval=0.25):
    """Poll ``path`` until the handshake's ``status`` is in ``accept``.

    Raises ``TimeoutError`` if the deadline is reached first.
    """
    accept = set(accept)
    deadline = time.monotonic() + timeout
    while True:
        payload = read_handshake(path)
        if payload is not None and payload.get("status") in accept:
            return payload
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Handshake at {path} did not reach {sorted(accept)} within "
                f"{timeout}s (last payload: {payload!r})"
            )
        time.sleep(poll_interval)
