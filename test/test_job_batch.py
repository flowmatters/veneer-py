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


class DrainRuleTests(unittest.TestCase):
    PORTS = [9101, 9102]

    def _batch(self, veneer, dask_busy, alive=None):
        c = _cluster([FakeFuture(1)])
        c.veneer_ports = list(self.PORTS)
        c._workers = [object(), object()]          # one client per port, unused by the fakes
        batch = c.submit_jobs(['a'], partial_results=True)
        batch._probe_veneer = lambda port, client, timeout: veneer(port)
        batch._dask_busy_ports = lambda timeout: set(dask_busy())
        batch._process_gone = lambda port: (alive is not None and port not in alive)
        return batch

    def test_idle_everywhere_drains_at_once(self):
        b = self._batch(lambda p: 'idle', lambda: [])
        self.assertEqual(b.drain(timeout=5, poll=0), [])

    def test_a_zero_job_batch_drains_at_once(self):
        self.assertEqual(_cluster([]).submit_jobs([]).drain(), [])

    def test_a_task_that_starts_after_its_port_was_judged_stopped_is_caught(self):
        # cancel() returns once the SCHEDULER has the cancel; the worker hears later, and a
        # task already queued there can start in that gap, after a round judged its port
        # idle. drain confirms with one more Dask read before it answers [].
        reads = iter([[], [9101], []])
        b = self._batch(lambda p: 'idle', lambda: next(reads, []))
        rounds = []
        self.assertEqual(b.drain(timeout=5, poll=0, on_poll=rounds.append), [])
        self.assertEqual(rounds, [1, 0])             # it went round again for 9101

    def test_a_port_whose_dask_worker_still_executes_is_busy(self):
        b = self._batch(lambda p: 'idle', lambda: [9102])
        self.assertEqual(b.drain(timeout=0.2, poll=0.05), [9102])

    def test_an_idle_veneer_whose_dask_worker_executes_is_busy_in_that_round(self):
        # Not only after the confirming read: while another port is still busy, the
        # round itself must count this one.
        counts = []
        b = self._batch(lambda p: 'busy' if p == 9101 else 'idle', lambda: [9102])
        self.assertEqual(b.drain(timeout=0.2, poll=0.05, on_poll=counts.append), self.PORTS)
        self.assertTrue(counts)
        self.assertTrue(all(n == 2 for n in counts))

    def test_a_busy_veneer_is_busy(self):
        # _probe_veneer folds a running model, and a 200 or 400 from /runs/cancel, to
        # 'busy' (pinned against real HTTP in the probe tests).
        b = self._batch(lambda p: 'busy' if p == 9101 else 'idle', lambda: [])
        self.assertEqual(b.drain(timeout=0.2, poll=0.05), [9101])

    def test_unreachable_is_busy_unless_dead_or_process_gone(self):
        b = self._batch(lambda p: 'unreachable', lambda: [])
        self.assertEqual(b.drain(timeout=0.2, poll=0.05), [9101, 9102])
        b = self._batch(lambda p: 'unreachable', lambda: [])
        self.assertEqual(b.drain(timeout=0.2, poll=0.05, dead_ports=[9101]), [9102])
        b = self._batch(lambda p: 'unreachable', lambda: [], alive={9102})
        self.assertEqual(b.drain(timeout=0.2, poll=0.05), [9102])

    def test_a_dead_port_stays_stopped_even_if_the_confirming_read_says_busy(self):
        # A gone Veneer answers nothing, so a dead port put back to busy would stay busy.
        gone = lambda p: 'unreachable' if p == 9101 else 'idle'
        reads = iter([[], [9101, 9102]])
        b = self._batch(gone, lambda: next(reads, [9102]))
        self.assertEqual(b.drain(timeout=0.2, poll=0.05, dead_ports=[9101]), [9102])
        reads = iter([[], [9101, 9102]])
        b = self._batch(gone, lambda: next(reads, [9102]), alive={9102})
        self.assertEqual(b.drain(timeout=0.2, poll=0.05), [9102])

    def test_a_stopped_port_is_not_probed_again(self):
        seen = []

        def veneer(p):
            seen.append(p)
            return 'idle' if p == 9101 else 'busy'
        b = self._batch(veneer, lambda: [])
        b.drain(timeout=0.3, poll=0.05)
        self.assertEqual(seen.count(9101), 1)
        self.assertGreater(seen.count(9102), 1)

    def test_on_poll_receives_the_busy_count_each_round(self):
        counts = []
        b = self._batch(lambda p: 'busy' if p == 9102 else 'idle', lambda: [])
        b.drain(timeout=0.2, poll=0.05, on_poll=counts.append)
        self.assertTrue(counts)
        self.assertTrue(all(n == 1 for n in counts))

    def test_an_old_veneer_raises(self):
        def veneer(p):
            raise cluster_mod.VeneerTooOldForDrain(p)
        b = self._batch(veneer, lambda: [])
        with self.assertRaises(cluster_mod.VeneerTooOldForDrain) as ctx:
            b.drain(timeout=1, poll=0)
        self.assertEqual(ctx.exception.port, 9101)

    def test_the_deadline_holds_against_a_hung_veneer(self):
        import time
        b = self._batch(lambda p: 'idle', lambda: [])
        # The probe sleeps exactly the timeout it is given, so drain must hand it
        # min(request_timeout, remaining) for the deadline to hold.
        b._probe_veneer = lambda port, client, timeout: (time.sleep(timeout), 'unreachable')[1]
        t0 = time.monotonic()
        busy = b.drain(timeout=0.5, poll=0.05, request_timeout=5)
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertEqual(busy, self.PORTS)

    def test_the_deadline_holds_against_a_hung_dask_read(self):
        import time
        b = self._batch(lambda p: 'idle', lambda: [])
        b._dask_busy_ports = lambda timeout: (time.sleep(timeout), set(self.PORTS))[1]
        t0 = time.monotonic()
        busy = b.drain(timeout=0.5, poll=0.05, request_timeout=5)
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertEqual(busy, self.PORTS)


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
