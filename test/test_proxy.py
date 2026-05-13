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
