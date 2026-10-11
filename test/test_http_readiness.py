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


def test_connection_refused_is_not_ready():
    assert interpret_readiness(None, None) is False


def test_timeout_error_carries_captured_output():
    err = VeneerStartupTimeout(600.0, ['line one', 'line two'])
    assert '600' in str(err)
    assert 'line two' in str(err)


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
