"""run_jobs reports completions as they land.

Run from `vendor/veneer-py`, where cwd shadows any editable veneer install so
`import veneer.cluster` is the vendored file this change touches. Same idiom as
test_existing_temp_directories.py.
"""
import unittest
from unittest import mock

import veneer.cluster as cluster_mod


class FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


def _fake_client(futures):
    """A dask_client stand-in. `submit(fn, args)` resolves the futures in args
    before calling, the way dask's does."""
    return type('C', (), {
        'compute': staticmethod(lambda jobs, sync=True:
                                [f.result() for f in futures] if sync
                                else list(futures)),
        'submit': staticmethod(lambda fn, args:
                               FakeFuture(fn([f.result() for f in args]))),
    })()


class RunJobsProgressTests(unittest.TestCase):
    def setUp(self):
        # as_completed is imported at function level in the implementation
        # (the constructor's project-copy loop does the same), so patch it at
        # the source module, where each call re-resolves it.
        patcher = mock.patch('dask.distributed.as_completed',
                             side_effect=lambda fs: iter(list(fs)))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _cluster(self, futures):
        cluster = cluster_mod.VeneerCluster.__new__(cluster_mod.VeneerCluster)
        cluster.dask_client = _fake_client(futures)
        return cluster

    def test_the_default_changes_nothing_and_starts_no_thread(self):
        # progress_callback=None is the historical path exactly: no
        # as_completed loop, no thread, no emit.
        cluster = self._cluster([FakeFuture(1), FakeFuture(2)])
        out = cluster.run_jobs(['a', 'b'], sync=False, partial_results=True)
        self.assertEqual(out.result(), [1, 2])

    def test_it_emits_once_per_completion_with_a_running_count(self):
        emitted = []
        cluster = self._cluster([FakeFuture(1), FakeFuture(2), FakeFuture(3)])
        cluster.run_jobs(['a', 'b', 'c'], sync=True, partial_results=True,
                         progress_callback=lambda *a: emitted.append(a))
        self.assertEqual([e[1] for e in emitted], [1, 2, 3])
        self.assertTrue(all(e[2] == 3 for e in emitted))
        self.assertTrue(all(e[0] == 'run-jobs' for e in emitted))

    def test_the_sync_result_order_survives_out_of_order_completion(self):
        # as_completed yields in completion order; the result order is part of
        # this method's contract, so the sync branch reassembles by index.
        futures = [FakeFuture(1), FakeFuture(2), FakeFuture(3)]
        cluster = self._cluster(futures)
        with mock.patch('dask.distributed.as_completed',
                        side_effect=lambda fs: iter(list(fs)[::-1])):
            out = cluster.run_jobs(['a', 'b', 'c'], sync=True,
                                   partial_results=True,
                                   progress_callback=lambda *a: None)
        self.assertEqual(list(out), [1, 2, 3])

    def test_a_raising_callback_never_reaches_the_caller(self):
        # _make_emitter already swallows; pinned here because a progress
        # annotation must never be able to fail a run.
        cluster = self._cluster([FakeFuture(1)])

        def boom(*a):
            raise RuntimeError('ui gone')

        cluster.run_jobs(['a'], sync=True, partial_results=True,
                         progress_callback=boom)   # no raise

    def test_no_jobs_emits_nothing_and_returns_empty(self):
        emitted = []
        cluster = self._cluster([])
        out = cluster.run_jobs([], sync=True, partial_results=True,
                               progress_callback=lambda *a: emitted.append(a))
        self.assertEqual(list(out), [])
        self.assertEqual(emitted, [])


if __name__ == '__main__':
    unittest.main()
