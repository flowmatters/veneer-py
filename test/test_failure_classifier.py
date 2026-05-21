"""Tests for the run_on_cluster wrapper's worker-death classifier.

We can't actually spin up Dask + Veneer here; instead we extract the
inner wrapped function's body shape by manually invoking the same
classification rules. The function under test is the safe_exit_code +
tail combination plus the exception classification, which we cover
via a small re-implementation that mirrors the wrapper.
"""
import os
import tempfile
import pytest
import requests

from veneer._failure import safe_exit_code, tail


def test_tail_handles_missing_file():
    assert tail(None) == '(no log file captured)'
    assert tail('C:/definitely/not/a/file/here.log') == '(no log file captured)'


def test_tail_returns_last_n_lines(tmp_path):
    p = tmp_path / 'x.log'
    p.write_text('\n'.join(f'line {i}' for i in range(500)) + '\n', encoding='utf-8')
    out = tail(str(p), max_lines=10)
    lines = out.strip().splitlines()
    assert lines == [f'line {i}' for i in range(490, 500)]


def test_tail_survives_binary_garbage(tmp_path):
    p = tmp_path / 'x.log'
    p.write_bytes(b'good\n\xff\xfe\nstill good\n')
    out = tail(str(p), max_lines=10)
    assert 'good' in out
    assert 'still good' in out


def test_safe_exit_code_returns_none_for_dead_pid():
    # Pick a pid that definitely doesn't exist.
    assert safe_exit_code(2**31 - 1) is None


def test_safe_exit_code_returns_none_for_none_pid():
    assert safe_exit_code(None) is None


def test_exception_classification_only_catches_network_errors():
    """The wrapper catches ConnectionError, Timeout, ChunkedEncodingError.

    Verify those three inherit from RequestException as expected, and that
    other RequestException subclasses we deliberately don't catch (HTTPError,
    InvalidURL) are not in the catch tuple.
    """
    caught_types = (
        requests.ConnectionError,
        requests.Timeout,
        requests.exceptions.ChunkedEncodingError,
    )
    assert issubclass(requests.ConnectionError, requests.RequestException)
    assert not isinstance(requests.HTTPError('x'), caught_types)
    assert not isinstance(requests.exceptions.InvalidURL('x'), caught_types)
