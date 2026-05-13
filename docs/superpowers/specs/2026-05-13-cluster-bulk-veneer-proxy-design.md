# Nested Attribute Chain Proxy for `VeneerCluster.v` and `BulkVeneer`

**Status:** Draft
**Date:** 2026-05-13
**Author:** Joel Rahman (with Claude)

## Background

`veneer-py` exposes two objects that look like a `Veneer` HTTP client but fan calls out to multiple underlying clients:

- `VeneerCluster.v` (an instance of `ClusterVeneerClient`, in `veneer/cluster.py`) — runs a method on every Veneer instance in a dask-managed cluster.
- `BulkVeneer` (in `veneer/manage.py`) — runs a method on every Veneer instance in an in-process list of clients.

Both are intended to mimic the regular `Veneer` client surface so that user code can be written once and target either a single instance or a pool.

## The Problem

The single-`Veneer` API has nested helpers: notably `v.model` (a `VeneerIronPython` instance from `veneer/server_side.py`) which itself exposes `v.model.ui`, `v.model.catchment`, `v.model.functions`, `v.model.operations`, etc., each with their own methods. Real-world calls therefore look like:

```python
v.model.operations.get_overrides()
v.model.functions.create_modelled_variable(...)
```

Neither proxy currently supports this:

- `ClusterVeneerClient.__getattr__` (`cluster.py:105`) intercepts the first attribute lookup and returns a *callable*. So `cluster.v.model` is already a callable that tries to remotely invoke a method named `model` — there is nowhere to attach `.operations.get_overrides()`. Nested calls fail outright.
- `BulkVeneer` uses `BulkVeneerApplication` (`manage.py:759`) which *does* try to accumulate an attribute chain in `self.names`. However it mutates the same list on every `__getattr__`, so reusing an intermediate proxy gives the wrong path:

  ```python
  m = bulk.model     # BulkVeneerApplication, names=['model']
  m.x()              # call: names=['model', 'x']
  m.y()              # call: names=['model', 'x', 'y']  -- WRONG
  ```

  In addition to the bug, the underlying chain-proxy idea is not shared with `ClusterVeneerClient`, so the cluster proxy doesn't benefit from it at all.

## Goals

1. Allow nested attribute chains on `VeneerCluster.v` — `cluster.v.model.operations.get_overrides()` must dispatch correctly to each worker.
2. Fix the chain-reuse mutation bug in `BulkVeneer`.
3. DRY the chain-proxy mechanics between the two call sites.
4. Preserve each call site's existing return-shape semantics (explicitly *not* a goal to unify them).
5. Add unit tests that exercise the proxy without requiring a running Veneer instance.

## Non-goals

- Aligning the return shapes of `BulkVeneer.call_on_all` (which strips `None` results and returns `None` for an all-`None` list) and `ClusterVeneerClient` (which returns the raw per-worker list from `run_on_each`).
- Removing the existing double `@dask.delayed` wrapping in `ClusterVeneerClient` + `run_on_each`. It is pre-existing, harmless, and out of scope.
- Providing `__dir__`/tab completion for nested chains. Top-level `__dir__` via a dummy `Veneer(0)` remains as today; nested completion is not added.
- Wider API redesign of `BulkVeneer` (e.g. its `verbose`/`run_async` quirks). Keep behaviour identical apart from the chain-proxy bug fix.

## Design

### 1. New module `veneer/_proxy.py`

A small, dependency-free module containing a single class:

```python
class AttributeChainProxy:
    """Immutable chain-of-attributes proxy.

    Each ``__getattr__`` returns a NEW proxy with the looked-up name appended
    to the path. ``__call__`` hands ``(path, args, kwargs)`` to a caller-
    supplied resolver, which is responsible for actually executing the call
    (e.g. walking the path on a real client, fanning out to dask workers).

    Attribute names beginning with ``_`` are NOT proxied — they raise
    ``AttributeError`` — so internal state, dunders, pickling, repr and
    IPython inspection do not accidentally fabricate paths.
    """
    __slots__ = ('_resolver', '_path')

    def __init__(self, resolver, path=()):
        object.__setattr__(self, '_resolver', resolver)
        object.__setattr__(self, '_path', tuple(path))

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return AttributeChainProxy(self._resolver, self._path + (name,))

    def __call__(self, *args, **kwargs):
        return self._resolver(self._path, args, kwargs)

    def __repr__(self):
        return 'AttributeChainProxy<%s>' % '.'.join(self._path)
```

Key properties:

- **Immutable path.** Each chain step returns a new instance, so holding an intermediate proxy and reusing it for multiple terminal calls works correctly.
- **Underscore guard.** `__getattr__` only fires for attributes Python didn't already find. `_resolver` and `_path` are set via `object.__setattr__` and listed in `__slots__`, so they resolve normally. The explicit `if name.startswith('_')` guard means that lookups Python *does* route through `__getattr__` (e.g. `__getstate__`, `__wrapped__`, IPython probing of `_ipython_canary_method_should_not_exist_`) raise `AttributeError` rather than silently extending the path.
- **No `__dir__`.** Nested tab completion is out of scope. The top-level call sites can provide their own `__dir__` for the first level.

### 2. `ClusterVeneerClient` refactor (`veneer/cluster.py`)

```python
from ._proxy import AttributeChainProxy

class ClusterVeneerClient(object):
    def __init__(self, cluster):
        self._cluster = cluster
        self._dummy_v = Veneer(0)

    def __getattr__(self, attrname):
        if attrname.startswith('_'):
            raise AttributeError(attrname)
        return AttributeChainProxy(self._make_resolver(), (attrname,))

    def _make_resolver(self):
        cluster = self._cluster
        def resolve(path, args, kwargs):
            @dask.delayed
            def fn(p):
                target = Veneer(p)
                for n in path:
                    target = getattr(target, n)
                return target(*args, **kwargs)
            return cluster.run_on_each(fn)
        return resolve

    def __dir__(self):
        return self._dummy_v.__dir__()
```

Behaviour preserved:

- `cluster.v.scenario_info()` — `path=('scenario_info',)`, the inner `fn` does `getattr(Veneer(p), 'scenario_info')()`, identical to the existing single-level path.
- `cluster.v.model.operations.get_overrides()` — `path=('model','operations','get_overrides')`, `fn` walks the chain on a fresh `Veneer(p)` inside each dask worker, then calls it. This is the bug fix.
- `__dir__` still returns top-level Veneer attributes for tab completion.

The existing double `@dask.delayed` (inner `fn` is delayed; `run_on_each` wraps with `dask.delayed` again) is intentionally preserved to minimise behaviour change.

### 3. `BulkVeneer` refactor (`veneer/manage.py`)

```python
from ._proxy import AttributeChainProxy

class BulkVeneer(object):
    def __init__(self, ports=[], clients=[], verbose=False):
        self.veneers = [Veneer(port) for port in ports]
        self.veneers += clients
        self.verbose = verbose

    def call_path(self, client, path, *pargs, **kwargs):
        if self.verbose:
            print('Calling %s on port %d' % ('.'.join(path), client.port))
        target = client
        for p in path:
            target = getattr(target, p)
        return target(*pargs, **kwargs)

    def call_on_all(self, path, *pargs, **kwargs):
        result = [self.call_path(v, path, *pargs, **kwargs) for v in self.veneers]
        result = [r for r in result if r is not None]
        if kwargs.get('run_async'):
            return [r.getresponse().getcode() for r in result]
        if len(result):
            return result
        return None

    def __getattr__(self, attrname):
        if attrname.startswith('_'):
            raise AttributeError(attrname)
        return AttributeChainProxy(self._make_resolver(), (attrname,))

    def _make_resolver(self):
        def resolve(path, args, kwargs):
            return self.call_on_all(list(path), *args, **kwargs)
        return resolve
```

`BulkVeneerApplication` is removed — it was an internal helper now superseded by `AttributeChainProxy`. `call_path` and `call_on_all` are kept (the existing semantics: filter `None`, return `None` when empty, handle `run_async`) so the proxy is the only thing that changes.

Behaviour preserved: any existing caller that used `bulk.some_method(...)` still works the same way. Nested chains now also work and are safe to reuse.

### 4. Tests

New file `test/test_proxy.py`. None of these tests require a running Veneer or Source — the resolver / stub clients are plain Python objects.

```python
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
    calls = []
    proxy = AttributeChainProxy(lambda p, a, k: calls.append(p))
    m = proxy.model
    m.x()
    m.y()
    assert calls == [('model', 'x'), ('model', 'y')]

def test_underscore_attrs_raise():
    proxy = AttributeChainProxy(lambda *args: None)
    import pytest
    with pytest.raises(AttributeError):
        proxy._foo
    with pytest.raises(AttributeError):
        proxy.__wrapped__
```

Plus one integration-ish test for `BulkVeneer` using stub clients (no HTTP):

```python
def test_bulk_veneer_walks_chain_on_each_client():
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
```

The cluster's resolver depends on `dask.delayed` and a real worker pool, so it is not unit-tested here. Its mechanics are covered indirectly by the four `AttributeChainProxy` tests plus existing integration tests against a live cluster.

## Files Changed

- **New:** `veneer/_proxy.py` — `AttributeChainProxy`.
- **Modified:** `veneer/cluster.py` — `ClusterVeneerClient` rewritten to use `AttributeChainProxy`.
- **Modified:** `veneer/manage.py` — `BulkVeneer` rewritten to use `AttributeChainProxy`; `BulkVeneerApplication` deleted.
- **New:** `test/test_proxy.py` — five unit tests.

## Risk and Compatibility

- `BulkVeneerApplication` is removed. It is undocumented and (based on the `__getattr__` plumbing) was only used as an internal step in `BulkVeneer.__getattr__`. If any external caller imported it directly, they would break. Grep before merging; if found, keep a deprecated alias.
- The underscore guard in `__getattr__` changes one edge case: previously `cluster.v._anything` returned a wrapped callable; now it raises `AttributeError`. This is the desired behaviour (pickling, repr, IPython probes would otherwise fabricate remote calls) and unlikely to be relied on.
- The cluster's resolver constructs a fresh `Veneer(p)` per call inside the dask worker — identical to the current code. No new lifecycle concerns.

## Rollout

Single PR; no migration step required. Existing user code that uses `cluster.v.method()` or `bulk.method()` continues to work unchanged.
