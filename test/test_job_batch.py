"""JobBatch: one batch of cluster jobs that can be cancelled and drained.

Run from the repository root so cwd shadows any installed veneer.
"""
import http.server
import os
import socket
import subprocess
import sys
import threading
import time
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

    def test_a_process_that_exits_during_the_drain_stops_its_port(self):
        alive = {9101, 9102}

        def gone_after_first_round(n):
            alive.discard(9102)
        b = self._batch(lambda p: 'idle' if p == 9101 else 'unreachable', lambda: [],
                        alive=alive)
        t0 = time.monotonic()
        self.assertEqual(b.drain(timeout=5, poll=0, on_poll=gone_after_first_round), [])
        self.assertLess(time.monotonic() - t0, 2)

    def test_every_port_dead_answers_at_once_without_reading_dask(self):
        def no_read():
            raise AssertionError('no Dask read is needed')
        b = self._batch(lambda p: 'unreachable', no_read)
        counts = []
        self.assertEqual(b.drain(timeout=5, dead_ports=self.PORTS, on_poll=counts.append), [])
        self.assertEqual(counts, [0])

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


class _FakeVeneerHandler(http.server.BaseHTTPRequestHandler):
    # Set per server: status_code, status_body, cancel_code, delay, requests (a list).
    def log_message(self, *args):
        pass

    def _answer(self, code, body=b''):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        cfg = self.server.cfg
        cfg['requests'].append(('GET', self.path))
        if self.path == '/runs/status':
            time.sleep(cfg.get('delay', 0))
            self._answer(cfg['status_code'], cfg.get('status_body', b''))
        else:
            self._answer(404)

    def do_POST(self):
        cfg = self.server.cfg
        cfg['requests'].append(('POST', self.path))
        self._answer(cfg['cancel_code'] if self.path == '/runs/cancel' else 404)


class ProbeVeneerTests(unittest.TestCase):
    '''_probe_veneer against real HTTP servers on ephemeral ports.'''

    def _serve(self, **cfg):
        cfg.setdefault('requests', [])
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _FakeVeneerHandler)
        server.daemon_threads = True
        server.block_on_close = False
        server.cfg = cfg
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1], cfg

    def _probe(self, port, timeout=2):
        from veneer.general import Veneer
        batch = _cluster([FakeFuture(1)]).submit_jobs(['a'])
        return batch._probe_veneer(port, Veneer(port, host='127.0.0.1'), timeout)

    IDLE = b'{"IsRunning": false}'
    RUNNING = b'{"IsRunning": true}'

    def _case(self, status_code, status_body, cancel_code):
        port, cfg = self._serve(status_code=status_code, status_body=status_body,
                                cancel_code=cancel_code)
        return self._probe(port), cfg['requests']

    def test_not_running_and_nothing_to_cancel_is_idle(self):
        answer, reqs = self._case(200, self.IDLE, 404)
        self.assertEqual(answer, 'idle')
        self.assertEqual(reqs, [('GET', '/runs/status'), ('POST', '/runs/cancel')])

    def test_running_is_busy_and_is_asked_to_cancel(self):
        answer, reqs = self._case(200, self.RUNNING, 200)
        self.assertEqual(answer, 'busy')
        self.assertIn(('POST', '/runs/cancel'), reqs)

    def test_a_run_claimed_but_not_started_is_busy(self):
        answer, reqs = self._case(200, self.IDLE, 400)
        self.assertEqual(answer, 'busy')
        self.assertIn(('POST', '/runs/cancel'), reqs)

    def test_a_cancel_that_found_something_is_busy(self):
        answer, reqs = self._case(200, self.IDLE, 200)
        self.assertEqual(answer, 'busy')

    def test_an_unreadable_status_is_busy_and_is_still_asked_to_cancel(self):
        for body in (b'not json', b'[1, 2]'):
            answer, reqs = self._case(200, body, 404)
            self.assertEqual(answer, 'busy', body)
            self.assertIn(('POST', '/runs/cancel'), reqs)

    def test_an_error_status_is_busy_and_is_still_asked_to_cancel(self):
        # A 404 from /runs/cancel is idle only after a readable status said no run.
        answer, reqs = self._case(500, b'', 404)
        self.assertEqual(answer, 'busy')
        self.assertEqual(reqs, [('GET', '/runs/status'), ('POST', '/runs/cancel')])

    def test_a_status_without_is_running_is_busy(self):
        answer, reqs = self._case(200, b'{}', 404)
        self.assertEqual(answer, 'busy')
        self.assertIn(('POST', '/runs/cancel'), reqs)

    def test_an_old_veneer_raises_and_is_not_asked_to_cancel(self):
        port, cfg = self._serve(status_code=404, cancel_code=404)
        with self.assertRaises(cluster_mod.VeneerTooOldForDrain) as ctx:
            self._probe(port)
        self.assertEqual(ctx.exception.port, port)
        self.assertEqual(cfg['requests'], [('GET', '/runs/status')])

    def test_nobody_listening_is_unreachable(self):
        s = socket.socket()
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
        s.close()
        self.assertEqual(self._probe(port), 'unreachable')

    def test_a_hung_veneer_is_unreachable_within_the_timeout(self):
        port, _ = self._serve(status_code=200, status_body=self.IDLE, cancel_code=404,
                              delay=2)
        t0 = time.monotonic()
        self.assertEqual(self._probe(port, timeout=0.3), 'unreachable')
        self.assertLess(time.monotonic() - t0, 1.5)


class DaskReadTests(unittest.TestCase):
    '''_dask_busy_ports on a fake client: placing answers, and failing toward busy.'''

    PORTS = [9101, 9102]

    def _batch(self, run):
        c = _cluster([FakeFuture(1, key='k1'), FakeFuture(2, key='k2')])
        c.veneer_ports = list(self.PORTS)
        c.worker_affinity = {
            'w1': cluster_mod.WorkerInfo(port=9101, directory='C:/example/a'),
            'w2': cluster_mod.WorkerInfo(port=9102, directory='C:/example/b'),
        }
        c.dask_client.run = run
        return c.submit_jobs(['a', 'b'], partial_results=True)

    def test_answers_are_placed_by_worker_name(self):
        seen = {}

        def run(fn, keys, on_error=None):
            seen.update(fn=fn, keys=keys, on_error=on_error)
            return {'tcp://a': ('w1', []), 'tcp://b': ('w2', ['k2'])}
        self.assertEqual(self._batch(run)._dask_busy_ports(1), {9102})
        self.assertIs(seen['fn'], cluster_mod._batch_task_keys)
        self.assertEqual(seen['keys'], ['k1', 'k2'])
        self.assertEqual(seen['on_error'], 'return')

    def test_a_busy_worker_that_cannot_be_placed_holds_every_port(self):
        run = lambda fn, keys, on_error=None: {'tcp://c': ('stranger', ['k1'])}
        self.assertEqual(self._batch(run)._dask_busy_ports(1), set(self.PORTS))

    def test_a_worker_error_holds_every_port(self):
        run = lambda fn, keys, on_error=None: {'tcp://a': ('w1', []), 'tcp://b': OSError('x')}
        self.assertEqual(self._batch(run)._dask_busy_ports(1), set(self.PORTS))

    def test_a_failed_read_holds_every_port(self):
        def run(fn, keys, on_error=None):
            raise OSError('scheduler gone')
        self.assertEqual(self._batch(run)._dask_busy_ports(1), set(self.PORTS))

    def test_a_hung_read_holds_every_port_and_starts_no_second_thread(self):
        release = threading.Event()
        entered = []

        def run(fn, keys, on_error=None):
            entered.append(1)
            release.wait(5)
            return {}
        b = self._batch(run)
        try:
            t0 = time.monotonic()
            self.assertEqual(b._dask_busy_ports(0.2), set(self.PORTS))
            self.assertEqual(b._dask_busy_ports(0.2), set(self.PORTS))
            self.assertLess(time.monotonic() - t0, 1.5)
            self.assertEqual(len(entered), 1)
        finally:
            release.set()


class ProcessGoneTests(unittest.TestCase):
    def _batch(self, pid):
        c = _cluster([FakeFuture(1)])
        c.worker_affinity = {'w1': cluster_mod.WorkerInfo(port=9101, directory='C:/example',
                                                          veneer_pid=pid)}
        return c.submit_jobs(['a'])

    def test_a_recorded_pid_that_has_exited_is_gone(self):
        p = subprocess.Popen([sys.executable, '-c', 'pass'])
        p.wait()
        self.assertTrue(self._batch(p.pid)._process_gone(9101))

    def test_a_live_pid_is_not_gone(self):
        self.assertFalse(self._batch(os.getpid())._process_gone(9101))

    def test_an_unknown_pid_or_port_is_not_gone(self):
        self.assertFalse(self._batch(None)._process_gone(9101))
        self.assertFalse(self._batch(os.getpid())._process_gone(9999))


class CancelAgainstReportTests(unittest.TestCase):
    def test_no_report_follows_the_return_of_cancel(self):
        fs = [FakeFuture(1), FakeFuture(2)]
        in_report = threading.Event()
        release = threading.Event()
        emitted = []

        def cb(stage, current, total, message):
            emitted.append(current)
            if current == 1:
                in_report.set()
                release.wait(5)

        before = set(threading.enumerate())
        with mock.patch('dask.distributed.as_completed', side_effect=lambda futs: iter(list(futs))):
            batch = _cluster(fs).submit_jobs(['a', 'b'], partial_results=True,
                                             progress_callback=cb)
            watcher, = [t for t in set(threading.enumerate()) - before
                        if t.name == 'veneer-run-jobs-progress']
            self.assertTrue(in_report.wait(5))
            at_return = []

            def cancel():
                batch.cancel()
                at_return.append(list(emitted))
            canceller = threading.Thread(target=cancel)
            canceller.start()
            canceller.join(0.3)
            self.assertTrue(canceller.is_alive())   # held until the report in progress ends
            release.set()
            canceller.join(5)
            watcher.join(5)
        self.assertFalse(canceller.is_alive())
        self.assertFalse(watcher.is_alive())
        # The watcher may win the lock for one more report before cancel() takes it,
        # but nothing is reported once cancel() has returned.
        self.assertEqual(emitted, at_return[0])

    def test_a_progress_callback_may_cancel_its_own_batch(self):
        fs = [FakeFuture(1), FakeFuture(2)]
        holder = {}
        emitted = []

        def cb(stage, current, total, message):
            emitted.append(current)
            holder['batch'].cancel()

        before = set(threading.enumerate())
        gate = threading.Event()

        def as_completed(futs):
            gate.wait(5)
            return iter(list(futs))
        with mock.patch('dask.distributed.as_completed', side_effect=as_completed):
            holder['batch'] = _cluster(fs).submit_jobs(['a', 'b'], partial_results=True,
                                                       progress_callback=cb)
            watcher, = [t for t in set(threading.enumerate()) - before
                        if t.name == 'veneer-run-jobs-progress']
            gate.set()
            watcher.join(5)
        self.assertFalse(watcher.is_alive())
        self.assertEqual(emitted, [1])
        self.assertTrue(holder['batch'].cancelled())


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
