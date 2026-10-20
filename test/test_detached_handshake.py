import json
import os
import time
from pathlib import Path

import pytest

from veneer.detached_handshake import (
    HandshakeStatus,
    read_handshake,
    wait_for_status,
    write_handshake,
)


def test_write_handshake_is_atomic_and_roundtrips(tmp_path: Path):
    path = tmp_path / "handshake.json"
    payload = {
        "status": HandshakeStatus.READY,
        "launcher_pid": 111,
        "veneer_pids": [222, 333],
        "actual_ports": [9876, 9877],
        "error": None,
    }
    write_handshake(path, payload)
    assert read_handshake(path) == payload
    # No leftover temp files next to the target
    assert {p.name for p in tmp_path.iterdir()} == {"handshake.json"}


def test_read_handshake_missing_file_returns_none(tmp_path: Path):
    assert read_handshake(tmp_path / "nope.json") is None


def test_read_handshake_tolerates_partial_write(tmp_path: Path):
    path = tmp_path / "handshake.json"
    path.write_text("{not valid json")
    assert read_handshake(path) is None


def test_wait_for_status_returns_when_status_matches(tmp_path: Path):
    path = tmp_path / "handshake.json"
    write_handshake(path, {"status": HandshakeStatus.STARTING})

    # Background: flip to READY after a brief delay
    import threading

    def flip():
        time.sleep(0.2)
        write_handshake(
            path,
            {"status": HandshakeStatus.READY, "actual_ports": [9876]},
        )

    threading.Thread(target=flip, daemon=True).start()
    result = wait_for_status(
        path, {HandshakeStatus.READY, HandshakeStatus.FAILED}, timeout=5.0
    )
    assert result["status"] == HandshakeStatus.READY
    assert result["actual_ports"] == [9876]


def test_wait_for_status_times_out(tmp_path: Path):
    path = tmp_path / "handshake.json"
    write_handshake(path, {"status": HandshakeStatus.STARTING})
    with pytest.raises(TimeoutError):
        wait_for_status(
            path, {HandshakeStatus.READY}, timeout=0.3, poll_interval=0.05
        )
