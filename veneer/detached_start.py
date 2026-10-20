"""Entry point for a detached Veneer command-line launcher process.

Invoked as ``python -m veneer.detached_start <config.json>`` from a new
terminal window by ``veneer.manage.start(detached=True)``. Reads the config,
calls ``veneer.manage.start`` in-process, writes the actual bound ports to a
handshake file, and then blocks until interrupted so the user can see output
and the child Veneer processes stay alive."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .detached_handshake import HandshakeStatus, write_handshake


@dataclasses.dataclass
class LauncherConfig:
    handshake_path: str
    start_kwargs: dict

    @classmethod
    def from_dict(cls, d):
        return cls(
            handshake_path=d["handshake_path"],
            start_kwargs=dict(d["start_kwargs"]),
        )

    def to_dict(self):
        return {
            "handshake_path": self.handshake_path,
            "start_kwargs": self.start_kwargs,
        }


def _create_time(pid):
    """The process's create_time, or None if it cannot be read."""
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:
        return None


def _default_block():
    """Block until SIGINT. Prints a hint so the interactive user knows how
    to shut the launcher down."""
    print("Veneer launcher running. Press Ctrl-C to shut down.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def run(config_path, _start_fn=None, _block_fn=None):
    """Execute the launcher lifecycle. Split out of ``main`` for testability.

    ``_start_fn`` and ``_block_fn`` are injection seams used by tests; the
    production defaults are ``veneer.manage.start`` and an interruptible
    infinite loop.
    """
    if _start_fn is None:
        from .manage import start as _start_fn
    if _block_fn is None:
        _block_fn = _default_block

    config = LauncherConfig.from_dict(json.loads(Path(config_path).read_text()))

    write_handshake(
        config.handshake_path,
        {
            "status": HandshakeStatus.STARTING,
            "launcher_pid": os.getpid(),
            "veneer_pids": [],
            "actual_ports": [],
            "error": None,
        },
    )

    # BaseException, not Exception: a Ctrl-C in the launcher's window must still tell the
    # parent, which would otherwise wait out its whole detached_timeout.
    try:
        processes, actual_ports = _start_fn(**config.start_kwargs)
    except BaseException as exc:
        write_handshake(
            config.handshake_path,
            {
                "status": HandshakeStatus.FAILED,
                "launcher_pid": os.getpid(),
                "veneer_pids": [],
                "actual_ports": [],
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise

    # Everything after start() returns is inside the try, so the instances are killed even
    # if reporting them fails.
    try:
        veneer_pids = [getattr(p, "pid", None) for p in processes]
        write_handshake(
            config.handshake_path,
            {
                "status": HandshakeStatus.READY,
                "launcher_pid": os.getpid(),
                "veneer_pids": veneer_pids,
                # Lets the parent tell these processes from any later one reusing a PID.
                "veneer_create_times": [_create_time(pid) for pid in veneer_pids],
                "actual_ports": list(actual_ports),
                "error": None,
            },
        )

        print(
            f"Veneer ready. PIDs={veneer_pids}, ports={list(actual_ports)}",
            flush=True,
        )

        _block_fn()
    finally:
        for p in processes:
            try:
                p.kill()
            except Exception:
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Detached Veneer command-line launcher (internal)."
    )
    parser.add_argument("config", help="Path to launcher config JSON")
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    sys.exit(main())
