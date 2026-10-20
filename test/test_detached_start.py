import json
import os
from pathlib import Path

from veneer.detached_handshake import HandshakeStatus, read_handshake
from veneer.detached_start import LauncherConfig, run


class _FakeProc:
    def __init__(self, pid):
        self.pid = pid

    def kill(self):
        self.killed = True

    def wait(self):
        return 0


def test_run_writes_ready_then_blocks_then_tears_down(tmp_path: Path):
    handshake = tmp_path / "handshake.json"
    config = LauncherConfig(
        handshake_path=str(handshake),
        start_kwargs={"n_instances": 2, "ports": [9990, 9990]},
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))

    captured = {}

    def fake_start(**kwargs):
        captured["start_kwargs"] = kwargs
        return [_FakeProc(111), _FakeProc(222)], [9990, 9991]

    blocks = {"called": False}

    def fake_block():
        blocks["called"] = True

    run(config_path, _start_fn=fake_start, _block_fn=fake_block)

    assert captured["start_kwargs"] == {"n_instances": 2, "ports": [9990, 9990]}
    assert blocks["called"]

    payload = read_handshake(handshake)
    assert payload["status"] == HandshakeStatus.READY
    assert payload["veneer_pids"] == [111, 222]
    assert payload["actual_ports"] == [9990, 9991]
    assert payload["launcher_pid"] == os.getpid()


def test_run_writes_failed_when_start_raises(tmp_path: Path):
    handshake = tmp_path / "handshake.json"
    config = LauncherConfig(
        handshake_path=str(handshake), start_kwargs={"n_instances": 1}
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))

    def failing_start(**_):
        raise RuntimeError("boom")

    import pytest
    with pytest.raises(RuntimeError):
        run(config_path, _start_fn=failing_start, _block_fn=lambda: None)

    payload = read_handshake(handshake)
    assert payload["status"] == HandshakeStatus.FAILED
    assert "boom" in payload["error"]
