"""Regression test for a prefix-doubling bug in Veneer.retrieve_json.

general.py's url() already applies self.prefix exactly once when building a request
URL. retrieve_json used to ALSO pre-apply self.prefix itself before handing the result
to url(), doubling it whenever prefix was non-empty. This never showed up with the
common prefix='' (''+anything==anything), so it went unnoticed until a caller set a
real per-instance prefix (veneer-pool's S6 front-door proxy) and got back a path with
the prefix repeated, e.g. /sessions/X/instances/Y/sessions/X/instances/Y/ready.
"""
from veneer.general import Veneer


class _FakeResponse:
    def __init__(self, text='{}', status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        pass


class _RecordingSession:
    def __init__(self):
        self.requested_urls = []

    def get(self, url, **kwargs):
        self.requested_urls.append(url)
        return _FakeResponse()


def _veneer_with_recording_session(prefix):
    v = Veneer(host='localhost', port=8124, prefix=prefix)
    v._session = _RecordingSession()
    return v


def test_retrieve_json_applies_a_non_empty_prefix_exactly_once():
    v = _veneer_with_recording_session('/sessions/abc/instances/def')
    v.retrieve_json('/ready')
    assert v._session.requested_urls == [
        'http://localhost:8124/sessions/abc/instances/def/ready'
    ]


def test_retrieve_json_with_no_prefix_is_unaffected():
    v = _veneer_with_recording_session('')
    v.retrieve_json('/ready')
    assert v._session.requested_urls == ['http://localhost:8124/ready']
