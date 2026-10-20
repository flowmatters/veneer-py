"""HTTP error responses from Veneer must carry their details into the Python exception.

Veneer reports a failure (e.g. a failed run) as an HTTP 500 whose JSON body holds the
message, the server-side stack trace and, for runs, the Source log captured during the
run. requests' raise_for_status() discards the body, so the caller used to see only
"500 Server Error: Internal Server Error for url: ..." with none of that detail.
"""
import json

import pytest
import requests

from veneer.general import Veneer, VeneerHTTPError


def _response(status_code, body, reason=None, url='http://localhost:9876/runs', method='POST'):
    resp = requests.Response()
    resp.status_code = status_code
    resp.reason = reason
    resp.url = url
    resp._content = body.encode('utf-8') if isinstance(body, str) else json.dumps(body).encode('utf-8')
    resp.request = requests.Request(method, url).prepare()
    return resp


class _FixedSession:
    def __init__(self, resp):
        self.resp = resp

    def post(self, url, **kwargs):
        return self.resp

    def get(self, url, **kwargs):
        return self.resp

    def request(self, method, url, **kwargs):
        return self.resp

    def delete(self, url, **kwargs):
        return self.resp


def _veneer(resp):
    v = Veneer(host='localhost', port=9876)
    v._session = _FixedSession(resp)
    return v


RUN_FAULT = {
    'Message': 'The simulation finished without producing a run.',
    'StackTrace': None,
    'Log': ['ERROR: Could not read input file inflow.csv', 'Run aborted'],
}


def test_run_model_500_includes_message_and_log():
    v = _veneer(_response(500, RUN_FAULT, reason='Internal Server Error'))
    with pytest.raises(VeneerHTTPError) as excinfo:
        v.run_model()

    err = excinfo.value
    text = str(err)
    assert '500' in text
    assert 'The simulation finished without producing a run.' in text
    assert 'Could not read input file inflow.csv' in text
    assert 'Run aborted' in text
    assert err.status_code == 500
    assert err.message == RUN_FAULT['Message']
    assert err.log == RUN_FAULT['Log']
    assert err.fault == RUN_FAULT


def test_run_model_500_includes_server_stack_trace():
    fault = {'Message': 'Object reference not set to an instance of an object.',
             'StackTrace': '   at Some.Source.Method()\n   at Another.Method()'}
    v = _veneer(_response(500, fault))
    with pytest.raises(VeneerHTTPError) as excinfo:
        v.run_model()

    text = str(excinfo.value)
    assert 'Object reference not set' in text
    assert 'at Some.Source.Method()' in text
    assert excinfo.value.stack_trace == fault['StackTrace']


def test_error_is_still_a_requests_http_error():
    # Existing callers that catch requests.HTTPError must keep working.
    v = _veneer(_response(500, RUN_FAULT))
    with pytest.raises(requests.HTTPError) as excinfo:
        v.run_model()
    assert excinfo.value.response.status_code == 500


def test_non_json_error_body_is_included_verbatim():
    v = _veneer(_response(500, '<html>Request Error: something broke</html>'))
    with pytest.raises(VeneerHTTPError) as excinfo:
        v.run_model()
    assert 'Request Error: something broke' in str(excinfo.value)
    assert excinfo.value.fault is None


def test_conflict_from_run_model_reports_the_reason():
    fault = {'Message': 'A simulation is already running. Cancel the current run before starting a new one.',
             'StackTrace': None}
    v = _veneer(_response(409, fault, reason='Conflict'))
    with pytest.raises(VeneerHTTPError) as excinfo:
        v.run_model()
    assert 'already running' in str(excinfo.value)
    assert excinfo.value.status_code == 409


def test_nested_inner_exception_details_are_included():
    fault = {'ExceptionType': 'System.InvalidOperationException',
             'Message': 'Outer failure',
             'StackTrace': '   at Outer()',
             'InnerException': {'ExceptionType': 'System.IO.IOException',
                                'Message': 'Inner failure',
                                'StackTrace': '   at Inner()'}}
    v = _veneer(_response(500, fault, url='http://localhost:9876/network', method='GET'))
    with pytest.raises(VeneerHTTPError) as excinfo:
        v.retrieve_json('/network')
    text = str(excinfo.value)
    assert 'System.InvalidOperationException' in text
    assert 'Outer failure' in text
    assert 'System.IO.IOException' in text
    assert 'Inner failure' in text
    assert 'at Inner()' in text


@pytest.mark.parametrize('call', [
    lambda v: v.post_json('/ironpython', {'Script': 'x'}),
    lambda v: v.update_json('/recorders', {}),
    lambda v: v.retrieve_csv('/runs/1/location/a/element/b/variable/c'),
    lambda v: v.drop_run(1),
    lambda v: v.delete_data_source('group'),
])
def test_other_http_calls_also_carry_details(call):
    v = _veneer(_response(500, {'Message': 'Server-side detail', 'StackTrace': '   at X()'}))
    with pytest.raises(VeneerHTTPError) as excinfo:
        call(v)
    assert 'Server-side detail' in str(excinfo.value)


def test_long_run_log_is_truncated_in_message_but_kept_in_full():
    log = ['line %d' % i for i in range(500)]
    v = _veneer(_response(500, {'Message': 'Run failed', 'StackTrace': None, 'Log': log}))
    with pytest.raises(VeneerHTTPError) as excinfo:
        v.run_model()
    text = str(excinfo.value)
    assert 'line 499' in text
    assert 'line 0\n' not in text
    assert excinfo.value.log == log


class _FakeHTTPClientResponse:
    def __init__(self, code, body):
        self._code = code
        self._body = body
        self.headers = {}

    def getcode(self):
        return self._code

    def read(self):
        return self._body.encode('utf-8')


class _FakeConnection:
    def __init__(self, resp):
        self._resp = resp

    def getresponse(self):
        return self._resp


def test_async_run_500_carries_details():
    v = Veneer(host='localhost', port=9876)
    conn = _FakeConnection(_FakeHTTPClientResponse(500, json.dumps(RUN_FAULT)))
    with pytest.raises(VeneerHTTPError) as excinfo:
        v._wait_for_run(conn)
    assert 'Could not read input file inflow.csv' in str(excinfo.value)


def test_async_run_302_returns_location():
    v = Veneer(host='localhost', port=9876)
    resp = _FakeHTTPClientResponse(302, '')
    resp.headers = {'Location': '/runs/3'}
    assert v._wait_for_run(_FakeConnection(resp)) == (302, '/runs/3')
