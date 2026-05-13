"""Immutable attribute-chain proxy used by VeneerCluster.v and BulkVeneer.

Each ``__getattr__`` returns a NEW proxy with the looked-up name appended
to the path. ``__call__`` hands ``(path, args, kwargs)`` to a caller-
supplied resolver, which is responsible for actually executing the call
(e.g. walking the path on a real client, fanning out to dask workers).

Attribute names beginning with ``_`` raise ``AttributeError`` so that
pickling, repr, and IPython attribute probing do not silently fabricate
remote calls.
"""


class AttributeChainProxy(object):
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
