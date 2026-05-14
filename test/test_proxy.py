"""Tests for veneer._proxy.AttributeChainProxy.

These tests do not require a running Veneer or Source instance — the
resolver is a plain Python callable that records its arguments.
"""
import pytest

from veneer._proxy import AttributeChainProxy


def test_single_attribute_call():
    calls = []
    proxy = AttributeChainProxy(lambda p, a, k: calls.append((p, a, k)))
    proxy.foo(1, x=2)
    assert calls == [(('foo',), (1,), {'x': 2})]


def test_nested_chain():
    calls = []
    proxy = AttributeChainProxy(lambda p, a, k: calls.append((p, a, k)))
    proxy.model.operations.get_overrides()
    assert calls == [(('model', 'operations', 'get_overrides'), (), {})]


def test_chain_reuse_is_immutable():
    """Regression for the BulkVeneerApplication mutation bug.

    Holding an intermediate proxy and reusing it for multiple terminal
    calls must produce independent paths.
    """
    calls = []
    proxy = AttributeChainProxy(lambda p, a, k: calls.append(p))
    m = proxy.model
    m.x()
    m.y()
    assert calls == [('model', 'x'), ('model', 'y')]


def test_underscore_attrs_raise():
    """Internal/dunder lookups must NOT be fabricated into remote calls."""
    proxy = AttributeChainProxy(lambda *args: None)
    with pytest.raises(AttributeError):
        proxy._foo
    with pytest.raises(AttributeError):
        proxy.__wrapped__


def test_returns_resolver_return_value():
    proxy = AttributeChainProxy(lambda p, a, k: 'sentinel')
    assert proxy.anything() == 'sentinel'


def test_bulk_veneer_walks_chain_on_each_client():
    """BulkVeneer dispatches a nested chain to every underlying client.

    Uses stub clients (no HTTP), so this test is safe in any environment.
    """
    from veneer.manage import BulkVeneer

    class StubLeaf:
        def __init__(self):
            self.calls = []
        def get_overrides(self, *a, **k):
            self.calls.append((a, k))
            return ('result', id(self))

    def make_stub_client():
        leaf = StubLeaf()
        client = type('StubClient', (), {})()
        client.port = 9876
        client.model = type('StubModel', (), {})()
        client.model.operations = type('StubOps', (), {})()
        client.model.operations.get_overrides = leaf.get_overrides
        client._leaf = leaf
        return client

    c1, c2 = make_stub_client(), make_stub_client()
    bulk = BulkVeneer(clients=[c1, c2])
    out = bulk.model.operations.get_overrides(42, flag=True)

    # call_on_all filters Nones; both leaves returned tuples so we get two results.
    assert len(out) == 2
    assert c1._leaf.calls == [((42,), {'flag': True})]
    assert c2._leaf.calls == [((42,), {'flag': True})]
