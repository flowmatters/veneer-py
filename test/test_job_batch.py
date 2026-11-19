"""JobBatch: one batch of cluster jobs that can be cancelled and drained.

Run from the repository root so cwd shadows any installed veneer.
"""
import threading
import unittest
from unittest import mock

import veneer.cluster as cluster_mod


class FakeFuture:
    def __init__(self, value=None, exc=None, key=None):
        self._value, self._exc = value, exc
        self.key = key or f'job-{id(self)}'
        self._cancelled = False

    def done(self):
        return True

    def cancelled(self):
        return self._cancelled

    def cancel(self):
        self._cancelled = True

    def result(self):
        if self._cancelled:
            raise cluster_mod.CancelledError()
        if self._exc:
            raise self._exc
        return self._value


def _client(futures):
    calls = {'submitted': None}

    def submit(fn, futs):
        calls['submitted'] = fn
        try:
            return FakeFuture(fn([f.result() for f in futs]))
        except BaseException as e:      # the aggregate carries a job's exception
            return FakeFuture(exc=e)
    return type('C', (), {
        'compute': staticmethod(lambda jobs, sync=False: list(futures)),
        'submit': staticmethod(submit),
        'cancel': staticmethod(lambda fs: [f.cancel() for f in fs]),
    })(), calls


def _cluster(futures):
    c = cluster_mod.VeneerCluster.__new__(cluster_mod.VeneerCluster)
    c.dask_client, c.calls = _client(futures)
    c.veneer_ports, c.worker_affinity = [], {}
    return c


class SubmitJobsTests(unittest.TestCase):
    def test_futures_are_the_per_job_futures_in_submission_order(self):
        fs = [FakeFuture(1), FakeFuture(2)]
        batch = _cluster(fs).submit_jobs(['a', 'b'])
        self.assertEqual(batch.futures, fs)

    def test_result_is_in_order_and_partial_results_selects_the_shape(self):
        ok = {'status': 'ok', 'worker': ('w', 1), 'value': 'v'}
        batch = _cluster([FakeFuture(ok)]).submit_jobs(['a'], partial_results=True)
        self.assertEqual(batch.result(), [ok])

    def test_without_partial_results_the_legacy_shape_is_returned(self):
        ok = {'status': 'ok', 'port': 9101, 'directory': 'C:/example', 'value': 'v'}
        batch = _cluster([FakeFuture(ok)]).submit_jobs(['a'])
        self.assertEqual(batch.result(), [((9101, 'C:/example'), 'v')])

    def test_a_job_exception_is_re_raised_unchanged(self):
        import requests
        err = requests.ConnectionError('boom')
        batch = _cluster([FakeFuture(exc=err)]).submit_jobs(['a'], partial_results=True)
        with self.assertRaises(requests.ConnectionError) as ctx:
            batch.result()
        self.assertIs(ctx.exception, err)

    def test_cancel_cancels_every_future_and_the_aggregate_and_is_idempotent(self):
        fs = [FakeFuture(1), FakeFuture(2)]
        batch = _cluster(fs).submit_jobs(['a', 'b'], partial_results=True)
        batch.cancel()
        batch.cancel()
        self.assertTrue(all(f.cancelled() for f in fs))
        self.assertTrue(batch._aggregate.cancelled())
        self.assertTrue(batch.cancelled())
        with self.assertRaises(cluster_mod.CancelledError):
            batch.result()

    def test_zero_jobs_is_already_done_with_an_empty_result_and_starts_no_watcher(self):
        cb = mock.Mock()
        with mock.patch.object(threading, 'Thread') as thread:
            batch = _cluster([]).submit_jobs([], progress_callback=cb)
        thread.assert_not_called()
        self.assertTrue(batch.done())
        self.assertEqual(batch.result(), [])
        cb.assert_not_called()

    def test_no_report_is_made_after_cancel(self):
        fs = [FakeFuture(1), FakeFuture(2)]
        second = threading.Event()      # the test has cancelled: yield the second future

        def as_completed(futs):
            futs = list(futs)
            yield futs[0]
            second.wait(5)
            yield futs[1]

        emitted = []
        first = threading.Event()

        def cb(stage, current, total, message):
            emitted.append((stage, current, total, message))
            first.set()

        before = set(threading.enumerate())
        with mock.patch('dask.distributed.as_completed', side_effect=as_completed):
            batch = _cluster(fs).submit_jobs(['a', 'b'], partial_results=True,
                                             progress_callback=cb)
            watcher, = [t for t in set(threading.enumerate()) - before
                        if t.name == 'veneer-run-jobs-progress']
            self.assertTrue(first.wait(5))
            batch.cancel()
            second.set()
            watcher.join(5)
        self.assertFalse(watcher.is_alive())
        self.assertEqual(emitted, [('run-jobs', 1, 2, 'Simulations complete (1/2)')])


class RunJobsSyncFalseTests(unittest.TestCase):
    def test_sync_false_returns_the_aggregate_future(self):
        ok = {'status': 'ok', 'port': 9101, 'directory': 'C:/example', 'value': 'v'}
        fut = _cluster([FakeFuture(ok)]).run_jobs(['a'], sync=False)
        self.assertEqual(fut.result(), [((9101, 'C:/example'), 'v')])

    def test_sync_false_with_zero_jobs_keeps_both_empty_shapes(self):
        c = _cluster([])
        self.assertEqual(c.run_jobs([], sync=False).result(), [])
        self.assertIs(c.calls['submitted'], cluster_mod._unwrap_or_raise)
        self.assertEqual(c.run_jobs([], sync=False, partial_results=True).result(), [])
        self.assertIs(c.calls['submitted'], list)


if __name__ == '__main__':
    unittest.main()
