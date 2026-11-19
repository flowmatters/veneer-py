"""JobBatch on a real, in-process Dask cluster: what cancel and drain's Dask read
actually see.

One worker with one thread makes "queued" deterministic, and processes=False keeps
it light and needs no pickling of the test functions. Run from the repository root
so cwd shadows any installed veneer.
"""
import threading
import time
import unittest

import dask
import requests
from distributed import Client, LocalCluster

import veneer.cluster as cluster_mod

STARTED = []
GATE = threading.Event()


def job(i):
    STARTED.append(i)
    GATE.wait(30)              # every job holds the only thread until the test releases it
    return i


def failing_job(i):
    raise requests.ConnectionError('connection refused', i)


def wait_for(predicate, timeout=10):
    deadline = time.time() + timeout
    while not predicate():
        if time.time() > deadline:
            raise AssertionError('timed out waiting')
        time.sleep(0.01)


def _key(name):
    return f'{name}-{time.time_ns()}'


class LocalClusterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lc = LocalCluster(n_workers=1, threads_per_worker=1, processes=False,
                              dashboard_address=None)
        cls.client = Client(cls.lc)

    @classmethod
    def tearDownClass(cls):
        GATE.set()
        cls.client.close()
        cls.lc.close()

    def setUp(self):
        STARTED.clear()
        GATE.clear()
        self.addCleanup(GATE.set)
        name = list(self.client.scheduler_info()['workers'].values())[0]['name']
        self.cluster = cluster_mod.VeneerCluster.__new__(cluster_mod.VeneerCluster)
        self.cluster.dask_client = self.client
        self.cluster.veneer_ports = [9201]
        self.cluster.worker_affinity = {
            str(name): cluster_mod.WorkerInfo(port=9201, directory='C:/example')}

    def _worker_heard(self, batch):
        worker, = self.lc.workers.values()

        def heard():
            tasks = worker.state.tasks
            return all(f.key not in tasks or tasks[f.key].state == 'cancelled'
                       for f in batch.futures)
        wait_for(heard)

    def _idle_worker(self):
        # A test must not leave a job running into the next one.
        wait_for(lambda: not any(self.client.processing().values()))

    def test_cancel_leaves_queued_jobs_unstarted(self):
        jobs = [dask.delayed(job)(i, dask_key_name=_key(f'job-{i}')) for i in range(4)]
        batch = self.cluster.submit_jobs(jobs, partial_results=True)
        wait_for(lambda: len(STARTED) == 1)     # whichever job the scheduler started first
        batch.cancel()
        # cancel() returns once the scheduler has the cancel. A job the worker already
        # holds can still start before the worker hears (drain's confirming read covers
        # that gap), so release the running job only once the worker has heard.
        self._worker_heard(batch)
        GATE.set()
        time.sleep(0.5)
        self.assertEqual(len(STARTED), 1)
        self.assertTrue(batch.cancelled())
        with self.assertRaises(cluster_mod.CancelledError):
            batch.result()
        self._idle_worker()

    def test_the_executing_set_holds_a_cancelled_running_task(self):
        jobs = [dask.delayed(job)(0, dask_key_name=_key('job-0'))]
        batch = self.cluster.submit_jobs(jobs, partial_results=True)
        wait_for(lambda: len(STARTED) == 1)
        batch.cancel()
        time.sleep(0.2)                         # let the worker hear of the cancel
        self.assertEqual(batch._dask_busy_ports(timeout=5), {9201})
        GATE.set()
        wait_for(lambda: batch._dask_busy_ports(timeout=5) == set(), timeout=5)
        self._idle_worker()

    def test_the_read_ignores_a_task_from_another_batch(self):
        other = self.cluster.submit_jobs([dask.delayed(job)(0, dask_key_name=_key('other'))],
                                         partial_results=True)
        wait_for(lambda: len(STARTED) == 1)
        mine = self.cluster.submit_jobs([dask.delayed(job)(1, dask_key_name=_key('mine'))],
                                        partial_results=True)
        self.assertEqual(mine._dask_busy_ports(timeout=5), set())
        self.assertEqual(other._dask_busy_ports(timeout=5), {9201})
        GATE.set()
        other.result()
        mine.result()

    def test_run_jobs_sync_false_is_unchanged(self):
        GATE.set()
        jobs = [dask.delayed(job)(i, dask_key_name=_key(f'job-{i}')) for i in (1, 2)]
        fut = self.cluster.run_jobs(jobs, sync=False, partial_results=True)
        self.assertEqual(fut.result(timeout=10), [1, 2])
        jobs = [dask.delayed(job)(i, dask_key_name=_key(f'job-{i}')) for i in (1, 2)]
        self.assertEqual(self.cluster.run_jobs(jobs, sync=True, partial_results=True), [1, 2])

    def test_run_jobs_sync_false_still_fails_fast_with_the_job_exception(self):
        GATE.set()
        jobs = [dask.delayed(failing_job)(7, dask_key_name=_key('fail')),
                dask.delayed(job)(1, dask_key_name=_key('ok'))]
        fut = self.cluster.run_jobs(jobs, sync=False, partial_results=True)
        wait_for(fut.done)
        # Real Dask serialises exceptions, so compare type and args, not identity.
        with self.assertRaises(requests.ConnectionError) as ctx:
            fut.result(timeout=10)
        self.assertEqual(ctx.exception.args, ('connection refused', 7))


if __name__ == '__main__':
    unittest.main()
