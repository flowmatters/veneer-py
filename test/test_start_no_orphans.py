"""start() must never leave a started VeneerCmd running when it raises.

Before this was fixed, start() killed its children only on its own timeout and
"failed to start" branches. Any other exception raised after Popen -- from the
readiness probe, from reading or parsing a child's output, from Popen itself for
a later instance, or a KeyboardInterrupt while waiting -- escaped with every
started VeneerCmd still running, and nothing left to stop them.

Most tests here use a fake Popen (nothing real is spawned): each fake process
records whether it was killed. The tree-kill tests at the end start a tiny
Python script standing in for VeneerCmd, which starts a child of its own, and
check that neither survives. No real VeneerCmd or Source is started.
"""
import io
import logging
import subprocess
import sys
import time

import psutil
import pytest

import veneer.manage as manage
import veneer.detached_handshake as detached_handshake

_REAL_POPEN = subprocess.Popen

ANNOUNCE = b'Now listening on: http://localhost:9990\r\n'


class FakeProc:
    """Stands in for a VeneerCmd Popen. Output comes from in-memory streams, which
    start()'s own reader threads drain exactly as they would a real pipe."""

    _next_pid = 1

    def __init__(self, stdout=b'', stderr=b'', exit_code=None):
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.stdin = None
        # Never a real PID: start() must not try to look these up with psutil.
        self.pid = -FakeProc._next_pid
        FakeProc._next_pid += 1
        self.returncode = exit_code
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        if self.returncode is None:
            self.returncode = 1

    terminate = kill

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture(autouse=True)
def at_exit(monkeypatch):
    """Record what veneer.manage registers to run at interpreter exit, instead of letting it
    run then against this file's fakes and stand-ins. Yields [(fn, args), ...]."""
    registered = []

    class _Recorder:
        @staticmethod
        def register(fn, *args, **kwargs):
            registered.append((fn, args))
            return fn

    monkeypatch.setattr(manage, 'atexit', _Recorder)
    yield registered
    # Still remove the detached launches' temporary directories, as exit would have.
    for fn, args in registered:
        if getattr(fn, '__name__', '') == '_cleanup_work_dir':
            fn(*args)


@pytest.fixture
def fake_popen(monkeypatch):
    """Replace Popen in veneer.manage. Returns (started, outputs): set outputs to a list
    of FakeProc keyword dicts (one per instance); started collects every FakeProc."""
    started = []
    outputs = []

    def popen(cmd, **kwargs):
        spec = outputs[len(started)] if len(started) < len(outputs) else {}
        if isinstance(spec, BaseException):
            raise spec
        proc = FakeProc(**spec)
        started.append(proc)
        return proc

    monkeypatch.setattr(manage, 'Popen', popen)
    monkeypatch.setattr(manage, '_detect_problematic_launcher', lambda: None)
    return started, outputs


def _start(**kwargs):
    defaults = dict(veneer_exe='VeneerCmd.exe', ports=9990, quiet=True,
                    return_log_paths=False, startup_timeout=10.0, gc=False)
    defaults.update(kwargs)
    return manage.start(**defaults)


# --- (a) the readiness wait ---------------------------------------------------------

def test_exception_from_readiness_probe_kills_the_instance_and_is_reraised(
        fake_popen, monkeypatch):
    started, outputs = fake_popen
    outputs.append(dict(stdout=ANNOUNCE))
    boom = RuntimeError('probe exploded')

    def probe(*a, **k):
        raise boom
    monkeypatch.setattr(manage, '_probe_ready', probe)

    with pytest.raises(RuntimeError) as excinfo:
        _start()

    assert excinfo.value is boom
    assert [p.killed for p in started] == [True]


# --- (b) reading and parsing a child's output ----------------------------------------

def test_exception_while_parsing_the_port_kills_the_instance_and_is_reraised(
        fake_popen, monkeypatch):
    started, outputs = fake_popen
    outputs.append(dict(stdout=ANNOUNCE))
    boom = ValueError('cannot parse port')

    class ExplodingRegex:
        def search(self, line):
            raise boom
    monkeypatch.setattr(manage, '_KESTREL_BOUND_RE', ExplodingRegex())
    monkeypatch.setattr(manage, '_probe_ready', lambda *a, **k: False)

    with pytest.raises(ValueError) as excinfo:
        _start()

    assert excinfo.value is boom
    assert [p.killed for p in started] == [True]


def test_undecodable_output_never_orphans_the_instance(fake_popen, monkeypatch):
    """The concrete orphan: a non-UTF-8 console line (an OEM code-page message, or a
    non-ASCII path echoed back) made the strict decode raise UnicodeDecodeError out of
    start() with VeneerCmd still running. Decoding is lenient now (next test), so this
    forces a strict decode to prove the kill does not depend on that."""
    started, outputs = fake_popen
    outputs.append(dict(stdout=b'\xff\xfe not utf-8\r\n' + ANNOUNCE))
    monkeypatch.setattr(manage, '_probe_ready', lambda *a, **k: True)
    # raising=False: before the fix start() decoded inline and strictly, so there was no
    # _decode_line to replace and the raw line raised anyway.
    monkeypatch.setattr(manage, '_decode_line', lambda line: line.decode('utf-8'),
                        raising=False)

    with pytest.raises(UnicodeDecodeError):
        _start()

    assert [p.killed for p in started] == [True]


def test_non_utf8_output_line_does_not_fail_startup(fake_popen, monkeypatch):
    started, outputs = fake_popen
    outputs.append(dict(stdout=b'Loading \xff\xfe project\r\n' + ANNOUNCE,
                        stderr=b'warning \x81\r\n'))
    monkeypatch.setattr(manage, '_probe_ready', lambda *a, **k: True)

    processes, ports = _start()

    assert processes == started
    assert ports == [9990]
    assert [p.killed for p in started] == [False]


def test_non_utf8_output_still_appears_in_a_startup_failure(fake_popen):
    started, outputs = fake_popen
    outputs.append(dict(stdout=b'Cannot find project C:\\\xe9t\xe9\\Example.rsproj\r\n'))

    with pytest.raises(Exception) as excinfo:
        _start()

    assert 'Cannot find project' in str(excinfo.value)
    assert [p.killed for p in started] == [True]


# --- (c) several instances -------------------------------------------------------------

def test_popen_failing_for_a_later_instance_kills_the_earlier_ones(fake_popen):
    started, outputs = fake_popen
    boom = OSError('cannot start instance 2')
    outputs.extend([dict(stdout=ANNOUNCE), boom])

    with pytest.raises(OSError) as excinfo:
        _start(n_instances=2)

    assert excinfo.value is boom
    assert len(started) == 1
    assert [p.killed for p in started] == [True]


def test_second_instance_failing_after_the_first_is_ready_kills_both(
        fake_popen, monkeypatch):
    started, outputs = fake_popen
    outputs.extend([dict(stdout=ANNOUNCE),
                     dict(stdout=b'Now listening on: http://localhost:9991\r\n')])
    boom = RuntimeError('instance 2 probe exploded')
    probed = []

    def probe(port, expected_pid=None, **k):
        probed.append(port)
        if port == 9991:
            raise boom
        return True
    monkeypatch.setattr(manage, '_probe_ready', probe)

    with pytest.raises(RuntimeError) as excinfo:
        _start(n_instances=2)

    assert excinfo.value is boom
    assert 9990 in probed and 9991 in probed
    assert [p.killed for p in started] == [True, True]


# --- (d) KeyboardInterrupt -----------------------------------------------------------

def test_keyboard_interrupt_while_waiting_kills_every_instance(fake_popen, monkeypatch):
    started, outputs = fake_popen
    outputs.extend([dict(), dict()])           # silent: start() keeps polling and sleeping
    monkeypatch.setattr(manage, '_probe_ready', lambda *a, **k: False)

    def interrupted_sleep(seconds):
        raise KeyboardInterrupt()
    monkeypatch.setattr(manage, 'sleep', interrupted_sleep)

    with pytest.raises(KeyboardInterrupt):
        _start(n_instances=2)

    assert [p.killed for p in started] == [True, True]


# --- (e) progress_callback -----------------------------------------------------------

def test_progress_callback_exception_is_still_logged_and_ignored(
        fake_popen, monkeypatch, caplog):
    started, outputs = fake_popen
    outputs.append(dict(stdout=ANNOUNCE))
    monkeypatch.setattr(manage, '_probe_ready', lambda *a, **k: True)
    calls = []

    def callback(stage, current, total, message):
        calls.append((stage, current, total))
        raise RuntimeError('callback bug')

    with caplog.at_level(logging.WARNING, logger='veneer.manage'):
        processes, ports = _start(progress_callback=callback)

    assert calls == [('veneer-start', 0, 1), ('veneer-start', 1, 1)]
    assert processes == started and ports == [9990]
    assert [p.killed for p in started] == [False]
    assert any('progress_callback raised' in r.getMessage() for r in caplog.records)


def test_progress_callback_keyboard_interrupt_kills_the_instances(fake_popen, monkeypatch):
    started, outputs = fake_popen
    outputs.extend([dict(stdout=ANNOUNCE), dict(stdout=ANNOUNCE)])
    monkeypatch.setattr(manage, '_probe_ready', lambda *a, **k: True)

    def callback(stage, current, total, message):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        _start(n_instances=2, progress_callback=callback)

    assert [p.killed for p in started] == [True, True]


# --- existing branches, which must keep killing, and success, which must not ---------

def test_startup_timeout_kills_the_instance(fake_popen, monkeypatch):
    started, outputs = fake_popen
    outputs.append(dict())
    monkeypatch.setattr(manage, '_probe_ready', lambda *a, **k: False)

    with pytest.raises(manage.VeneerStartupTimeout):
        _start(startup_timeout=0.3)

    assert [p.killed for p in started] == [True]


def test_could_not_start_kills_the_instance_and_reports_it_still_running(fake_popen):
    started, outputs = fake_popen
    outputs.append(dict(stdout=b'[10:51:30 AM] COULD NOT START VENEER\r\n'))

    with pytest.raises(Exception) as excinfo:
        _start()

    assert 'COULD NOT START VENEER' in str(excinfo.value)
    # The exit code is read before the kill, so it says what start() actually saw.
    assert 'still running when failure detected' in str(excinfo.value)
    assert [p.killed for p in started] == [True]


def test_silent_exit_puts_the_exit_code_in_the_headline(fake_popen):
    started, outputs = fake_popen
    outputs.append(dict(exit_code=3))

    with pytest.raises(Exception) as excinfo:
        _start()

    headline = str(excinfo.value).splitlines()[0]
    assert headline.startswith('One or more instances of Veneer failed to start')
    assert 'exited with code 3' in headline


def test_successful_start_kills_nothing(fake_popen, monkeypatch):
    started, outputs = fake_popen
    outputs.extend([dict(stdout=ANNOUNCE),
                     dict(stdout=b'Now listening on: http://localhost:9991\r\n')])
    monkeypatch.setattr(manage, '_probe_ready', lambda *a, **k: True)

    processes, ports, log_paths = _start(n_instances=2, return_log_paths=True)

    assert processes == started
    assert ports == [9990, 9991]
    assert log_paths == [None, None]
    assert [p.killed for p in started] == [False, False]


# --- kill_all_now --------------------------------------------------------------------

def test_kill_all_now_waits_a_bounded_time(caplog):
    class NeverExits(FakeProc):
        def wait(self, timeout=None):
            if timeout is None:
                raise AssertionError('kill_all_now waited without a timeout')
            raise subprocess.TimeoutExpired('VeneerCmd.exe', timeout)

    proc = NeverExits()
    with caplog.at_level(logging.WARNING, logger='veneer.manage'):
        manage.kill_all_now([proc], timeout=0.1)
    assert proc.killed
    assert any('did not exit' in r.getMessage() for r in caplog.records)


def test_kill_all_now_still_kills_the_rest_when_one_kill_fails():
    class Unkillable(FakeProc):
        def kill(self):
            raise psutil.AccessDenied(self.pid)

    bad, good = Unkillable(), FakeProc()
    with pytest.raises(psutil.AccessDenied):
        manage.kill_all_now([bad, good])
    assert good.killed


def test_kill_all_now_still_kills_the_rest_after_a_bad_pid():
    good = FakeProc()
    with pytest.raises(ValueError):            # psutil rejects a negative PID
        manage.kill_all_now([-1, good])
    assert good.killed


def test_kill_all_now_sends_every_kill_before_a_keyboard_interrupt_propagates():
    class Interrupted(FakeProc):
        def kill(self):
            raise KeyboardInterrupt()

    first, second = Interrupted(), FakeProc()
    with pytest.raises(KeyboardInterrupt):
        manage.kill_all_now([first, second])
    assert second.killed


def test_kill_all_now_keeps_the_first_error_over_a_later_wait_error():
    class Unkillable(FakeProc):
        def kill(self):
            raise psutil.AccessDenied(self.pid)

    class NoTimeoutWait(FakeProc):
        def wait(self):                       # an old-style wait() with no timeout
            return self.returncode

    with pytest.raises(psutil.AccessDenied):
        manage.kill_all_now([Unkillable(), NoTimeoutWait()])


def test_kill_all_now_with_timeout_none_waits_without_a_limit():
    waited = []

    class Recording(FakeProc):
        def wait(self, timeout='unset'):
            waited.append(timeout)
            return self.returncode

    proc = Recording()
    manage.kill_all_now([proc], timeout=None)
    assert proc.killed and waited == [None]


def test_kill_all_now_skips_a_pid_that_has_already_gone():
    proc = _REAL_POPEN([sys.executable, '-c', 'pass'])
    proc.wait(timeout=30)
    manage.kill_all_now([proc.pid])           # must not raise


def test_keyboard_interrupt_while_killing_keeps_the_original_exception(
        fake_popen, monkeypatch, at_exit):
    """The cleanup must not replace the exception start() is propagating, must still kill
    the instances it can, and leaves an at-exit kill registered for the one it could not."""
    started, outputs = fake_popen
    outputs.extend([dict(), dict()])
    monkeypatch.setattr(manage, '_probe_ready', lambda *a, **k: False)
    orig = RuntimeError('orig')

    def interrupted_kill():
        raise KeyboardInterrupt()

    def failing_sleep(seconds):
        started[0].kill = interrupted_kill
        raise orig
    monkeypatch.setattr(manage, 'sleep', failing_sleep)

    # Not pytest.raises: if the KeyboardInterrupt escaped it would end the whole test run
    # rather than fail this test.
    raised = None
    try:
        _start(n_instances=2)
    except BaseException as e:
        raised = e

    assert raised is orig
    assert started[1].killed
    assert [fn for fn, args in at_exit] == [manage._kill_started]
    assert list(at_exit[0][1][0]) == started


def test_failure_path_at_exit_fallback_kills_past_one_failing_kill(
        fake_popen, monkeypatch, at_exit):
    """The fallback start() registers on its failure path must try every instance (unlike
    kill_all_on_exit's bare loop, which stops at the first failure)."""
    started, outputs = fake_popen
    outputs.extend([dict(), dict()])
    monkeypatch.setattr(manage, '_probe_ready', lambda *a, **k: False)

    def refused_kill():
        raise psutil.AccessDenied(started[0].pid)

    def failing_sleep(seconds):
        started[0].kill = refused_kill
        raise RuntimeError('orig')
    monkeypatch.setattr(manage, 'sleep', failing_sleep)

    with pytest.raises(RuntimeError, match='orig'):
        _start(n_instances=2)

    assert len(at_exit) == 1
    fallback, args = at_exit[0]
    started[1].killed = False
    started[1].returncode = None
    fallback(*args)                              # what interpreter exit would run; never raises
    assert started[1].killed


def test_successful_start_registers_the_at_exit_kill_once(fake_popen, monkeypatch):
    started, outputs = fake_popen
    outputs.append(dict(stdout=ANNOUNCE))
    monkeypatch.setattr(manage, '_probe_ready', lambda *a, **k: True)
    registered = []
    monkeypatch.setattr(manage, 'kill_all_on_exit', lambda procs: registered.append(list(procs)))

    _start()

    assert registered == [started]


# --- real process trees (a stand-in script, not VeneerCmd) -----------------------------

FAKE_VENEER_WITH_CHILD = '''
import pathlib, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
print('Now listening on: http://localhost:9990', flush=True)
time.sleep(120)
'''

windows_only = pytest.mark.skipif(
    sys.platform != 'win32',
    reason='start() launches veneer_exe as one command string (Windows Popen), and '
           'only kills process trees on Windows')


def _wait_gone(pids, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = [p for p in pids if _alive(p)]
        if not alive:
            return []
        time.sleep(0.1)
    return [p for p in pids if _alive(p)]


def _alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.fixture
def reap():
    """Collects psutil.Process objects, or pid files a stand-in writes its child's PID to,
    and kills whatever is still running after the test, so a failing test (e.g. under
    mutation) leaves nothing behind."""
    items = []
    yield items
    procs = []
    for item in items:
        if isinstance(item, psutil.Process):
            procs.append(item)
            continue
        try:
            procs.append(psutil.Process(int(item.read_text())))
        except (OSError, ValueError, psutil.Error):
            pass
    for p in procs:
        try:
            if p.is_running():
                p.kill()
        except psutil.Error:
            pass


def _spawn_tree(tmp_path, name, *extra_args):
    """Start the stand-in script directly; return (parent psutil.Process, child pid).
    extra_args are appended to its command line (it ignores them)."""
    script = tmp_path / f'{name}.py'
    script.write_text(FAKE_VENEER_WITH_CHILD)
    pid_file = tmp_path / f'{name}.pid'
    proc = _REAL_POPEN([sys.executable, str(script), str(pid_file), *map(str, extra_args)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 20
    while not (pid_file.exists() and pid_file.read_text()):
        assert time.monotonic() < deadline, 'stand-in never started its child'
        time.sleep(0.05)
    return psutil.Process(proc.pid), int(pid_file.read_text())


@windows_only
def test_start_kills_the_whole_process_tree_when_startup_raises(
        tmp_path, monkeypatch, reap):
    script = tmp_path / 'fake_veneer_cmd.py'
    script.write_text(FAKE_VENEER_WITH_CHILD)
    pid_file = tmp_path / 'child.pid'
    reap.append(pid_file)
    veneer_exe = subprocess.list2cmdline([sys.executable, str(script), str(pid_file)])

    started = []
    real_manage_popen = manage.Popen

    def recording_popen(*a, **k):
        p = real_manage_popen(*a, **k)
        started.append(p)
        reap.append(psutil.Process(p.pid))
        return p
    monkeypatch.setattr(manage, 'Popen', recording_popen)
    monkeypatch.setattr(manage, '_detect_problematic_launcher', lambda: None)

    def probe(*a, **k):
        raise RuntimeError('probe exploded')
    monkeypatch.setattr(manage, '_probe_ready', probe)

    with pytest.raises(RuntimeError, match='probe exploded'):
        manage.start(veneer_exe=veneer_exe, ports=9990, quiet=True, gc=False,
                     startup_timeout=30.0, return_log_paths=False)

    child_pid = int(pid_file.read_text())
    assert started[0].poll() is not None, 'the stand-in VeneerCmd is still running'
    assert _wait_gone([child_pid]) == [], "the stand-in VeneerCmd's own child is still running"


@windows_only
def test_kill_all_now_kills_the_whole_process_tree(tmp_path, reap):
    parent, child_pid = _spawn_tree(tmp_path, 'tree')
    reap.extend([parent, psutil.Process(child_pid)])

    manage.kill_all_now([parent.pid])

    assert _wait_gone([parent.pid, child_pid]) == []


# --- detached mode -------------------------------------------------------------------

def _detached_setup(monkeypatch, on_wait):
    """Run _start_detached without opening a terminal: the 'start ... cmd /k' Popen is
    recorded, and wait_for_status is replaced by on_wait(handshake_path, config_path)."""
    spawned = []
    monkeypatch.setattr(subprocess, 'Popen', lambda cmd, **k: spawned.append(cmd))
    monkeypatch.setattr(detached_handshake, 'wait_for_status',
                        lambda path, accept, timeout: on_wait(path, path.parent / 'config.json'))
    return spawned


def _starting(path, launcher_pid):
    detached_handshake.write_handshake(path, {
        'status': detached_handshake.HandshakeStatus.STARTING,
        'launcher_pid': launcher_pid, 'veneer_pids': [], 'actual_ports': [], 'error': None,
    })


def _spawn_launcher(tmp_path, reap, config_path, name='launcher'):
    """A stand-in launcher: started after the launch began, naming its config file."""
    launcher, child = _spawn_tree(tmp_path, name, config_path)
    reap.extend([launcher, psutil.Process(child)])
    return launcher, child


@windows_only
@pytest.mark.parametrize('error', [TimeoutError('handshake timed out'), KeyboardInterrupt()])
def test_detached_wait_failing_kills_the_launcher_tree(tmp_path, monkeypatch, reap, error):
    found = {}

    def on_wait(path, config_path):
        found['tree'] = _spawn_launcher(tmp_path, reap, config_path)
        _starting(path, found['tree'][0].pid)
        raise error
    spawned = _detached_setup(monkeypatch, on_wait)

    with pytest.raises(type(error)):
        _start(detached=True, detached_timeout=1.0)

    assert len(spawned) == 1
    assert _wait_gone([found['tree'][0].pid, found['tree'][1]]) == []


@windows_only
def test_detached_launcher_that_has_not_written_its_pid_is_still_killed(
        tmp_path, monkeypatch, reap):
    found = {}

    def on_wait(path, config_path):
        found['tree'] = _spawn_launcher(tmp_path, reap, config_path)
        _starting(path, None)                  # Ctrl-C before the launcher reported in
        raise KeyboardInterrupt()
    _detached_setup(monkeypatch, on_wait)

    with pytest.raises(KeyboardInterrupt):
        _start(detached=True)

    assert _wait_gone([found['tree'][0].pid, found['tree'][1]]) == []


@windows_only
def test_detached_cleanup_never_kills_a_reused_pid(tmp_path, monkeypatch, reap):
    """A launcher PID (or an instance PID) in the handshake can, by the time the parent
    gives up, belong to an unrelated process. Only processes created since the launch
    began, and for the launcher only one naming this launch's config file, are killed."""
    older, older_child = _spawn_tree(tmp_path, 'older')       # predates the launch
    reap.extend([older, psutil.Process(older_child)])
    found = {}

    def on_wait(path, config_path):
        # Started after the launch began, but not this launch's launcher.
        found['newer'] = _spawn_tree(tmp_path, 'newer')
        reap.extend([found['newer'][0], psutil.Process(found['newer'][1])])
        detached_handshake.write_handshake(path, {
            'status': detached_handshake.HandshakeStatus.STARTING,
            'launcher_pid': found['newer'][0].pid, 'veneer_pids': [older.pid],
            # What the launcher recorded for the process that held this PID then.
            'veneer_create_times': [older.create_time() - 100],
            'actual_ports': [], 'error': None,
        })
        raise TimeoutError('handshake timed out')
    _detached_setup(monkeypatch, on_wait)

    with pytest.raises(TimeoutError):
        _start(detached=True, detached_timeout=1.0)

    newer, newer_child = found['newer']
    for pid in (older.pid, older_child, newer.pid, newer_child):
        assert _alive(pid), f'{pid} was killed'


@windows_only
@pytest.mark.parametrize('recorded', ['mismatched', 'missing'])
def test_detached_cleanup_skips_an_instance_pid_it_cannot_verify(
        tmp_path, monkeypatch, reap, recorded):
    """An instance PID reused by a process created since the launch began is not this
    launch's instance unless its create_time matches the one the launcher recorded; a
    handshake from an older launcher records none, so its PIDs cannot be verified."""
    found = {}

    def on_wait(path, config_path):
        found['other'] = _spawn_tree(tmp_path, 'other')
        reap.extend([found['other'][0], psutil.Process(found['other'][1])])
        payload = {
            'status': detached_handshake.HandshakeStatus.STARTING,
            'launcher_pid': None, 'veneer_pids': [found['other'][0].pid],
            'actual_ports': [], 'error': None,
        }
        if recorded == 'mismatched':
            payload['veneer_create_times'] = [found['other'][0].create_time() - 5]
        detached_handshake.write_handshake(path, payload)
        raise TimeoutError('handshake timed out')
    _detached_setup(monkeypatch, on_wait)

    with pytest.raises(TimeoutError):
        _start(detached=True, detached_timeout=1.0)

    assert _alive(found['other'][0].pid)


@windows_only
def test_detached_cleanup_still_kills_the_launcher_when_the_handshake_is_unreadable(
        tmp_path, monkeypatch, reap):
    """Windows can refuse to open the handshake while the launcher is replacing it."""
    found = {}

    def on_wait(path, config_path):
        found['tree'] = _spawn_launcher(tmp_path, reap, config_path)
        _starting(path, found['tree'][0].pid)
        raise TimeoutError('handshake timed out')
    _detached_setup(monkeypatch, on_wait)

    def locked(path):
        raise PermissionError(13, 'The process cannot access the file', str(path))
    monkeypatch.setattr(detached_handshake, 'read_handshake', locked)

    with pytest.raises(TimeoutError):
        _start(detached=True, detached_timeout=1.0)

    assert _wait_gone([found['tree'][0].pid, found['tree'][1]]) == []


@windows_only
def test_detached_interrupt_during_the_search_keeps_the_original_exception(
        tmp_path, monkeypatch, at_exit):
    """A second Ctrl-C while cleanup searches the process table must not replace the
    exception being propagated, and leaves the cleanup registered to run again at exit."""
    orig = RuntimeError('wait failed')     # (a TimeoutError would be re-wrapped)

    def on_wait(path, config_path):
        raise orig
    _detached_setup(monkeypatch, on_wait)

    def interrupted_iter(*a, **k):
        raise KeyboardInterrupt()
    monkeypatch.setattr(manage.psutil, 'process_iter', interrupted_iter)

    raised = None
    try:                                         # not pytest.raises; see above
        _start(detached=True, detached_timeout=1.0)
    except BaseException as e:
        raised = e

    assert raised is orig
    assert [fn for fn, args in at_exit if fn is manage._kill_detached_launch]


@windows_only
def test_detached_cleanup_cancels_a_launcher_that_has_not_started(
        tmp_path, monkeypatch, at_exit):
    """Cleanup renames the config away first, so a launcher that only starts afterwards
    fails reading it before it starts any Veneer."""
    import veneer.detached_start as detached_start
    seen = {}

    def on_wait(path, config_path):
        seen['config'] = config_path
        raise TimeoutError('handshake timed out')
    _detached_setup(monkeypatch, on_wait)
    def no_processes(attrs=None, *a, **k):
        seen['attrs'] = attrs
        return iter([])
    monkeypatch.setattr(manage.psutil, 'process_iter', no_processes)

    with pytest.raises(TimeoutError):
        _start(detached=True, detached_timeout=1.0)

    # create_time is slow to read for every process; only matching ones are asked.
    assert 'create_time' not in seen['attrs']
    config_path = seen['config']
    assert not config_path.exists()
    started = []
    with pytest.raises(FileNotFoundError):
        detached_start.run(config_path, _start_fn=lambda **k: started.append(k),
                           _block_fn=lambda: None)
    assert started == []


@windows_only
def test_detached_failure_after_ready_kills_the_instances_and_launcher(
        tmp_path, monkeypatch, reap):
    found = {}

    def on_wait(path, config_path):
        launcher, launcher_child = _spawn_launcher(tmp_path, reap, config_path)
        instance, instance_child = _spawn_tree(tmp_path, 'instance')
        reap.extend([instance, psutil.Process(instance_child)])
        found['pids'] = [launcher.pid, launcher_child, instance.pid, instance_child]
        payload = {
            'status': detached_handshake.HandshakeStatus.READY,
            'launcher_pid': launcher.pid, 'veneer_pids': [instance.pid],
            'veneer_create_times': [instance.create_time()],
            'actual_ports': [9990], 'error': None,
        }
        detached_handshake.write_handshake(path, payload)
        return payload
    _detached_setup(monkeypatch, on_wait)
    boom = RuntimeError('cannot register exit handler')

    def failing_register(processes):
        raise boom
    monkeypatch.setattr(manage, 'kill_all_on_exit', failing_register)

    with pytest.raises(RuntimeError) as excinfo:
        _start(detached=True)

    assert excinfo.value is boom
    assert _wait_gone(found['pids']) == []


@windows_only
def test_detached_launcher_reporting_failure_is_left_to_exit_by_itself(
        tmp_path, monkeypatch, reap):
    """A FAILED handshake means the launcher's own start() raised (and cleaned up its own
    instances); the launcher then prints the traceback into its window and exits."""
    found = {}

    def on_wait(path, config_path):
        found['tree'] = _spawn_launcher(tmp_path, reap, config_path)
        payload = {
            'status': detached_handshake.HandshakeStatus.FAILED,
            'launcher_pid': found['tree'][0].pid, 'veneer_pids': [], 'actual_ports': [],
            'error': 'Exception: One or more instances of Veneer failed to start.',
        }
        detached_handshake.write_handshake(path, payload)
        return payload
    _detached_setup(monkeypatch, on_wait)

    with pytest.raises(RuntimeError, match='Detached Veneer launcher failed'):
        _start(detached=True)

    assert found['tree'][0].is_running()
