"""Tests for HTTP-based readiness detection in veneer.manage.start().

S4a (Veneer side, not yet landed) adds GET /ready to VeneerCmd: 200 when a
scenario is loaded, 503 otherwise, 404 on servers built before protocol
20261001. These tests cover the pure decision logic and the port-ownership
guard without needing a live VeneerCmd process.
"""
from veneer.manage import interpret_readiness, VeneerStartupTimeout


def test_ready_endpoint_200_is_ready():
    assert interpret_readiness(200, {'Ready': True}) is True


def test_ready_endpoint_503_is_not_ready():
    assert interpret_readiness(503, {'Ready': False, 'Reason': 'no scenario loaded'}) is False


def test_old_server_without_ready_falls_back_to_root_scenario():
    assert interpret_readiness(404, None, root_status=200, root_body={'Scenario': 'Main'}) is True
    assert interpret_readiness(404, None, root_status=200, root_body={'Scenario': None}) is False


def test_method_not_allowed_also_falls_back_to_root_scenario():
    # Found live against a real production 6.1 VeneerCmd: its CoreWCF routing answers
    # GET /ready with 405 Method Not Allowed, not the 404 Program.cs's lack of a /ready
    # route was assumed (by reading the source alone) to produce. Any status that isn't a
    # modern server's own 200/503 must still fall back, not just 404.
    assert interpret_readiness(405, None, root_status=200, root_body={'Scenario': 'Main'}) is True
    assert interpret_readiness(405, None, root_status=200, root_body={'Scenario': None}) is False


def test_connection_refused_is_not_ready():
    assert interpret_readiness(None, None) is False


def test_timeout_error_carries_captured_output():
    err = VeneerStartupTimeout(600.0, ['line one', 'line two'])
    assert '600' in str(err)
    assert 'line two' in str(err)


def test_popen_with_stdin_pipe_raises_no_buffering_warning():
    """start() opens stdin/stdout/stderr as PIPE for every instance (stdin is used for
    the exit-on-stdin-eof die-with-parent behaviour). bufsize=1 (line buffering) is
    only meaningful for text-mode streams; on these binary pipes Python raises
    RuntimeWarning and substitutes the default buffer size anyway, three times per
    instance. _POPEN_BUFSIZE is what start() actually passes to Popen; prove it here
    with a real (trivial, short-lived) subprocess rather than mocking Popen, since the
    warning is raised deep inside Popen's own pipe setup, not something callable code
    triggers directly."""
    import subprocess
    import sys
    import warnings
    from veneer.manage import _POPEN_BUFSIZE

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with subprocess.Popen(
            [sys.executable, '-c', 'pass'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=_POPEN_BUFSIZE,
        ) as proc:
            proc.wait(timeout=10)

    buffering_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)
                          and 'buffering' in str(w.message)]
    assert buffering_warnings == []


def test_probe_rejects_a_foreign_instance_on_the_port(monkeypatch):
    import veneer.manage as m

    class FakeResponse:
        ok = True
        status_code = 200
        headers = {'content-type': 'application/json'}
        def __init__(self, body): self._body = body
        def json(self): return self._body

    calls = []
    def fake_get(url, timeout):
        calls.append(url)
        return FakeResponse({'PID': 999, 'Scenario': 'Main'})
    monkeypatch.setattr('requests.get', fake_get)
    assert m._probe_ready(9876, expected_pid=123) is False
    assert calls == ['http://localhost:9876/']   # stopped at the PID check, never asked /ready


def test_probe_falls_back_to_root_when_ready_answers_405(monkeypatch):
    import veneer.manage as m

    class FakeResponse:
        def __init__(self, status_code, body, ok=True):
            self.status_code = status_code
            self.ok = ok
            self.headers = {'content-type': 'application/json'}
            self._body = body

        def json(self):
            return self._body

    responses = {
        'http://localhost:9876/ready': FakeResponse(405, None),
        'http://localhost:9876/': FakeResponse(200, {'Scenario': 'Main'}),
    }

    def fake_get(url, timeout):
        return responses[url]

    monkeypatch.setattr('requests.get', fake_get)
    assert m._probe_ready(9876) is True
