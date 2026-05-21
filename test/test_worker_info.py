import json
import pytest

from veneer.cluster import WorkerInfo


def test_to_dict_round_trips():
    w = WorkerInfo(port=9876, directory='C:/tmp/x', veneer_pid=1234, log_path='C:/tmp/x.log')
    d = w.to_dict()
    assert d == {'port': 9876, 'directory': 'C:/tmp/x', 'veneer_pid': 1234, 'log_path': 'C:/tmp/x.log'}
    assert WorkerInfo.from_dict(d) == w


def test_from_dict_optional_log_path():
    w = WorkerInfo.from_dict({'port': 9876, 'directory': '.', 'veneer_pid': 1})
    assert w.log_path is None


def test_from_dict_coerces_ints():
    w = WorkerInfo.from_dict({'port': '9876', 'directory': '.', 'veneer_pid': '1'})
    assert w.port == 9876
    assert w.veneer_pid == 1


def test_from_dict_omitted_veneer_pid_yields_none():
    """When a legacy cluster-config JSON omits veneer_pid, from_dict yields
    None — not a 0 sentinel — so downstream psutil-based liveness checks can
    distinguish 'unknown pid' from a real pid (which would never legitimately
    be 0 on Windows or POSIX for a Veneer worker)."""
    w = WorkerInfo.from_dict({'port': 9876, 'directory': '.'})
    assert w.veneer_pid is None
    assert w.log_path is None


def test_json_round_trip_in_cluster_config():
    w = WorkerInfo(port=9876, directory='.', veneer_pid=1, log_path=None)
    blob = json.dumps({'worker_affinity': {'4444': w.to_dict()}})
    reloaded = json.loads(blob)
    assert WorkerInfo.from_dict(reloaded['worker_affinity']['4444']) == w
