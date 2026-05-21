"""Small post-mortem helpers used by the run_on_cluster wrapper.

Kept separate from cluster.py so the unit tests don't have to import dask.
"""
from __future__ import annotations
import os
import psutil


def safe_exit_code(pid: 'int | None') -> 'int | None':
    """Return the process exit code if the OS still has it, else None.

    psutil cleans up the zombie shortly after exit, so this is best-effort.
    """
    if pid is None:
        return None
    try:
        p = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return None
    try:
        return p.wait(timeout=0)
    except (psutil.TimeoutExpired, psutil.NoSuchProcess):
        return None


def tail(path: 'str | None', max_lines: int = 200) -> str:
    """Return the last `max_lines` lines of a UTF-8 file, never raising.

    Used to attach VeneerCmd's stdout/stderr tail to incident reports.
    """
    if not path or not os.path.exists(path):
        return '(no log file captured)'
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except OSError as exc:
        return f'(could not read {path}: {exc})'
    return ''.join(lines[-max_lines:])
