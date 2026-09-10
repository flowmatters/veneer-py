"""The affinity map must survive a dask nanny restarting a worker process.

`worker_affinity` used to be keyed by the dask worker's OS pid, captured once
at cluster startup. A nanny restart (memory limit, crash, external kill) gives
the replacement worker a NEW pid, and nothing repopulated the map — so every
subsequent task landing on that slot died with a bare `KeyError(<pid>)` while
its VeneerCmd process sat there alive and unreachable.

The dask worker *name* is stable across a nanny restart; the worker *address*
is not (the replacement binds a fresh port). So the map is keyed by name.
"""
import json
import os
import unittest

import pytest

import veneer.cluster as cluster_mod
from veneer.cluster import (
    ClusterAffinityError,
    WorkerInfo,
    build_worker_affinity,
    resolve_worker_info,
)


def _info(port, directory='C:/tmp/p', pid=1000):
    return WorkerInfo(port=port, directory=directory, veneer_pid=pid, log_path=None)


class BuildWorkerAffinityTests(unittest.TestCase):
    def test_keys_are_the_dask_worker_names_as_strings(self):
        # LocalCluster names workers 0..n-1 (ints); JSON keys are strings, so
        # the map is keyed by str throughout to keep one key type everywhere.
        affinity = build_worker_affinity([0, 1], [_info(9876), _info(9877)])
        self.assertEqual(sorted(affinity), ['0', '1'])
        self.assertEqual(affinity['0'].port, 9876)

    def test_names_are_paired_with_veneer_instances_in_a_stable_order(self):
        # Any bijection is correct (workers and Veneer instances are
        # interchangeable), but it must be deterministic so that reconnecting
        # to the same cluster twice produces the same assignment.
        first = build_worker_affinity(['w-2', 'w-1'], [_info(9876), _info(9877)])
        second = build_worker_affinity(['w-1', 'w-2'], [_info(9876), _info(9877)])
        self.assertEqual({k: v.port for k, v in first.items()},
                         {k: v.port for k, v in second.items()})

    def test_each_worker_gets_a_distinct_veneer_instance(self):
        affinity = build_worker_affinity([0, 1, 2], [_info(9876), _info(9877), _info(9878)])
        ports = [i.port for i in affinity.values()]
        self.assertEqual(sorted(ports), [9876, 9877, 9878])

    def test_a_count_mismatch_is_refused_rather_than_silently_sharing(self):
        # Two dask workers pointed at one Veneer instance would run concurrent
        # simulations against the same Source process.
        with self.assertRaises(ClusterAffinityError):
            build_worker_affinity([0, 1, 2], [_info(9876), _info(9877)])


class ResolveWorkerInfoTests(unittest.TestCase):
    def test_resolves_by_worker_name(self):
        affinity = build_worker_affinity([0, 1], [_info(9876), _info(9877)])
        self.assertEqual(resolve_worker_info(affinity, '1').port, 9877)

    def test_resolves_after_the_worker_process_has_been_replaced(self):
        # The regression: same worker name, brand new pid. Under the old
        # pid-keyed map this raised KeyError(<new pid>) forever.
        affinity = build_worker_affinity([0, 1], [_info(9876), _info(9877)])
        self.assertEqual(resolve_worker_info(affinity, '0', pid=36376).port, 9876)

    def test_an_unknown_worker_raises_an_actionable_error_not_a_keyerror(self):
        affinity = build_worker_affinity([0, 1], [_info(9876), _info(9877)])
        with self.assertRaises(ClusterAffinityError) as ctx:
            resolve_worker_info(affinity, '7', pid=36376)
        message = str(ctx.exception)
        self.assertIn('7', message)          # the worker that has no entry
        self.assertIn('36376', message)      # ...and its pid, for cross-checking
        self.assertIn("'0', '1'", message)   # the workers that DO have entries
        self.assertIn('restart', message.lower())

    def test_a_missing_worker_key_raises_the_same_actionable_error(self):
        # get_worker() fails outside a distributed worker (eg the synchronous
        # scheduler). That must not surface as `KeyError(None)` either.
        affinity = build_worker_affinity([0], [_info(9876)])
        with self.assertRaises(ClusterAffinityError):
            resolve_worker_info(affinity, None, pid=36376)


class CurrentWorkerKeyTests(unittest.TestCase):
    def test_returns_none_when_not_running_on_a_dask_worker(self):
        self.assertIsNone(cluster_mod.current_worker_key())


class ClusterConfigTests(unittest.TestCase):
    def _cluster(self, affinity):
        c = cluster_mod.VeneerCluster.__new__(cluster_mod.VeneerCluster)
        c.original_project_file = 'p.rsproj'
        c.n_workers = len(affinity)
        c.name = 'test cluster'
        c.veneer_ports = [i.port for i in affinity.values()]
        c.veneer_processes = []
        c.temp_directories = []
        c.worker_affinity = affinity
        c.copy_projects = False
        c.dask_client = type('C', (), {'scheduler': type('S', (), {'address': 'tcp://x'})()})()
        return c

    def test_to_json_writes_name_keys_and_declares_the_key_scheme(self):
        # The key scheme is recorded so a config written by an older
        # veneer-py (pid keys) can be told apart on reconnect.
        affinity = build_worker_affinity([0, 1], [_info(9876), _info(9877)])
        config = json.loads(self._cluster(affinity).to_json())
        self.assertEqual(config['affinity_key'], 'worker_name')
        self.assertEqual(sorted(config['worker_affinity']), ['0', '1'])

    def test_load_affinity_reads_a_current_config(self):
        affinity = build_worker_affinity([0, 1], [_info(9876), _info(9877)])
        config = json.loads(self._cluster(affinity).to_json())
        self.assertEqual(cluster_mod.load_worker_affinity(config, [0, 1]), affinity)

    def test_load_affinity_remaps_a_legacy_pid_keyed_config_onto_live_names(self):
        # Old config: keys are dead dask pids. They are meaningless now, so the
        # recorded Veneer instances are re-paired with the live worker names.
        legacy = {
            'affinity_key': None,
            'worker_affinity': {
                '55260': _info(9876).to_dict(),
                '46672': _info(9877).to_dict(),
            },
        }
        remapped = cluster_mod.load_worker_affinity(legacy, [0, 1])
        self.assertEqual(sorted(remapped), ['0', '1'])
        self.assertEqual(sorted(i.port for i in remapped.values()), [9876, 9877])

    def test_load_affinity_accepts_the_legacy_two_element_list_form(self):
        legacy = {
            'worker_affinity': {'55260': [9876, 'C:/tmp/p']},
        }
        remapped = cluster_mod.load_worker_affinity(legacy, [0])
        self.assertEqual(remapped['0'].port, 9876)
        self.assertIsNone(remapped['0'].veneer_pid)


class LiveWorkerNamesTests(unittest.TestCase):
    def test_scheduler_state_is_read_without_the_five_worker_default(self):
        # Client.scheduler_info() defaults to n_workers=5 and truncates
        # SILENTLY, so a 7 worker cluster would report 5 and the affinity map
        # would be short by two.
        seen = {}

        class FakeClient:
            def scheduler_info(self, n_workers=5):
                seen['n_workers'] = n_workers
                return {'workers': {f'tcp://127.0.0.1:{p}': {'name': i}
                                    for i, p in enumerate(range(9000, 9007))}}

        names = cluster_mod.live_worker_names(FakeClient())
        self.assertEqual(seen['n_workers'], -1)
        self.assertEqual(sorted(names), list(range(7)))


@pytest.mark.skipif(os.environ.get('VENEER_CLUSTER_INTEGRATION') != '1',
                    reason='starts a real dask LocalCluster; set '
                           'VENEER_CLUSTER_INTEGRATION=1 to run')
class NannyRestartIntegrationTests(unittest.TestCase):
    """End to end reproduction against a real dask cluster.

    No Source needed: the wrapped function only reads the port off the Veneer
    client, and Veneer.__init__ performs no I/O.
    """

    def test_jobs_still_resolve_after_a_worker_is_killed_and_respawned(self):
        import time

        import distributed
        import psutil
        from dask.distributed import Client, LocalCluster

        dask_cluster = LocalCluster(n_workers=2, threads_per_worker=1,
                                    host='127.0.0.1', dashboard_address=None)
        client = Client(dask_cluster)
        try:
            names = list(dask_cluster.workers)
            fake = cluster_mod.VeneerCluster.__new__(cluster_mod.VeneerCluster)
            fake.name = 'integration'
            fake.copy_projects = False
            fake._veneer_kwargs = {'trust_env': None, 'proxies': None}
            fake.worker_affinity = build_worker_affinity(
                names, [_info(9876), _info(9877)])

            wrapped = cluster_mod.run_on_cluster(fake, _report_port)

            before = client.gather(client.compute(
                [wrapped(i) for i in range(8)], sync=False))
            self.assertTrue(all(r['status'] == 'ok' for r in before))

            target = names[0]
            was = _address_of(client, target)
            psutil.Process(dask_cluster.workers[target].pid).kill()

            # Wait for the SCHEDULER to see the replacement, not just for the
            # nanny's pid to change — the nanny holds the old worker_address
            # until the new process registers, and pinning work to a dead
            # address hangs forever rather than failing.
            now = None
            for _ in range(60):
                time.sleep(1)
                now = _address_of(client, target)
                if now is not None and now != was:
                    break
            self.assertIsNotNone(now, 'replacement worker never registered')
            self.assertNotEqual(now, was)

            # Pin the work to the restarted slot — the one that used to be
            # poisoned. Left to itself the scheduler may route everything to
            # the worker that never died.
            futures = client.compute([wrapped(i) for i in range(4)], sync=False,
                                     workers=[now], allow_other_workers=False)
            distributed.wait(futures, timeout=60)
            after = client.gather(futures)
            self.assertTrue(all(r['status'] == 'ok' for r in after))
            self.assertEqual({r['port'] for r in after},
                             {fake.worker_affinity[str(target)].port})
        finally:
            client.close()
            dask_cluster.close()


def _report_port(_, v=None, **kwargs):
    return v.port


def _address_of(client, worker_name):
    """Current address of the worker with this name, per the scheduler."""
    workers = client.scheduler_info(n_workers=-1)['workers']
    for address, state in workers.items():
        if state.get('name') == worker_name:
            return address
    return None
