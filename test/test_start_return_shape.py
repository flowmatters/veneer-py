"""Verify the back-compat return shape of manage.start.

The legacy return is a 2-tuple (processes, ports). When the new
capture-output-dir feature was added, start() grew a third element
(log_paths), which was a silent breaking change for direct callers.

The compromise: keep the default 2-tuple shape, emit DeprecationWarning
unless the caller explicitly opts in either way, and let callers pass
return_log_paths=True to receive the new 3-tuple shape.

We don't actually spawn VeneerCmd here — that would need Source. We
inject a fake _start_detached path which exercises the return-shape
selection logic in start() without touching the real Popen path.
"""
import warnings
import pytest

import veneer.manage as manage


def _fake_detached(start_kwargs, detached_timeout, leave_open):
    n = start_kwargs.get('n_instances', 1)
    return [object()] * n, list(range(9876, 9876 + n)), [None] * n


def test_default_emits_deprecation_warning_and_returns_2_tuple(monkeypatch):
    monkeypatch.setattr(manage, '_start_detached', _fake_detached)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = manage.start(detached=True, n_instances=1)
    assert len(result) == 2
    assert any(issubclass(w.category, DeprecationWarning)
               and 'return_log_paths' in str(w.message)
               for w in caught), [str(w.message) for w in caught]


def test_explicit_false_returns_2_tuple_no_warning(monkeypatch):
    monkeypatch.setattr(manage, '_start_detached', _fake_detached)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = manage.start(detached=True, n_instances=1, return_log_paths=False)
    assert len(result) == 2
    assert not any(issubclass(w.category, DeprecationWarning)
                   and 'return_log_paths' in str(w.message)
                   for w in caught)


def test_explicit_true_returns_3_tuple_no_warning(monkeypatch):
    monkeypatch.setattr(manage, '_start_detached', _fake_detached)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = manage.start(detached=True, n_instances=1, return_log_paths=True)
    assert len(result) == 3
    procs, ports, log_paths = result
    assert len(procs) == 1
    assert ports == [9876]
    assert log_paths == [None]
    assert not any(issubclass(w.category, DeprecationWarning)
                   and 'return_log_paths' in str(w.message)
                   for w in caught)
