"""GC environment-variable construction for started VeneerCmd instances (manage.py's `gc`
parameter on start()/IsolatedSource/VeneerCluster/the detached-start path).

No real VeneerCmd/Source needed for the env-construction tests below: they mock Popen (the
in-process path) and _start_detached (the detached path) the same way
test_start_return_shape.py already does, and assert on what each would have been called
with. A separate, live check (once GET / reports its own GC mode -- not yet available) will
confirm an actually-started instance reports the mode it was asked for.
"""
import queue
import pytest
import veneer.manage as manage
from veneer.manage import _gc_env_vars


class _EmptyQueue:
    def get_nowait(self):
        raise queue.Empty


def test_default_is_dynamic_adaptation():
    assert _gc_env_vars(None) == {'DOTNET_GCDynamicAdaptationMode': '1'}


def test_dynamic_named_explicitly_matches_the_default():
    assert _gc_env_vars('dynamic') == _gc_env_vars(None)


def test_false_passes_nothing():
    assert _gc_env_vars(False) == {}


def test_server():
    assert _gc_env_vars('server') == {'DOTNET_gcServer': '1'}


def test_workstation():
    assert _gc_env_vars('workstation') == {'DOTNET_gcServer': '0'}


def test_int_sets_heap_count():
    assert _gc_env_vars(4) == {'DOTNET_GCHeapCount': '4'}


def test_dict_passes_through_raw():
    assert _gc_env_vars({'DOTNET_GCgen0size': '1000000'}) == {'DOTNET_GCgen0size': '1000000'}


def test_true_is_rejected():
    # True is a bool, and bool is a subclass of int in Python -- without an explicit guard,
    # this would silently fall through to the int branch and produce DOTNET_GCHeapCount=True.
    with pytest.raises(ValueError, match='gc=True'):
        _gc_env_vars(True)


def test_unrecognised_string_is_rejected():
    with pytest.raises(ValueError, match='Unrecognised gc setting'):
        _gc_env_vars('bogus')


def _fake_detached(start_kwargs, detached_timeout, leave_open):
    n = start_kwargs.get('n_instances', 1)
    return [object()] * n, list(range(9876, 9876 + n)), [None] * n


def test_start_detached_passes_gc_through_in_start_kwargs(monkeypatch):
    captured = {}

    def recording_detached(start_kwargs, detached_timeout, leave_open):
        captured.update(start_kwargs)
        return _fake_detached(start_kwargs, detached_timeout, leave_open)

    monkeypatch.setattr(manage, '_start_detached', recording_detached)
    manage.start(detached=True, n_instances=1, gc=8, return_log_paths=True)
    assert captured['gc'] == 8


def _patch_popen_and_readiness(monkeypatch):
    """Shared setup for the two tests below: fakes Popen (records its kwargs, never spawns
    anything real) and makes the readiness check report "ready" on its very first poll, so
    start() never reaches its startup_timeout/kill_all_now path (which expects a real
    process object -- the fake one here doesn't need to survive that far)."""
    captured_calls = []

    class _FakeProc:
        stdout = None
        stderr = None
        stdin = None
        pid = 99999

        def poll(self):
            return None  # never exited

        def kill(self):
            pass  # start() registers an atexit kill_all_now(); must be a harmless no-op here

        def wait(self, timeout=None):
            pass

    def fake_popen(cmd, **kwargs):
        captured_calls.append(kwargs)
        return _FakeProc()

    def fake_non_blocking_io(processes, *a, **k):
        n = len(processes)
        return [_EmptyQueue() for _ in range(n)], [None] * n

    monkeypatch.setattr(manage, 'Popen', fake_popen)
    monkeypatch.setattr(manage, 'configure_non_blocking_io', fake_non_blocking_io)
    monkeypatch.setattr(manage, '_probe_ready', lambda *a, **k: True)
    return captured_calls


def test_start_in_process_popen_env_includes_the_default_gc_mode(monkeypatch, tmp_path):
    captured_calls = _patch_popen_and_readiness(monkeypatch)

    project = tmp_path / "model.rsproj"
    project.write_text("not a real project")
    manage.start(project_fn=str(project), n_instances=1, veneer_exe="VeneerCmd.exe",
                 exit_on_stdin_eof=False, return_log_paths=True)

    assert len(captured_calls) == 1
    env = captured_calls[0]['env']
    assert env['DOTNET_GCDynamicAdaptationMode'] == '1'
    # exit_on_stdin_eof=False above: confirms gc's env additions don't depend on it.
    assert 'VENEER_EXIT_ON_STDIN_EOF' not in env


def test_start_in_process_popen_env_is_none_when_gc_is_false_and_no_other_overrides(monkeypatch, tmp_path):
    captured_calls = _patch_popen_and_readiness(monkeypatch)

    project = tmp_path / "model.rsproj"
    project.write_text("not a real project")
    manage.start(project_fn=str(project), n_instances=1, veneer_exe="VeneerCmd.exe",
                 exit_on_stdin_eof=False, gc=False, return_log_paths=True)

    assert captured_calls[0]['env'] is None
