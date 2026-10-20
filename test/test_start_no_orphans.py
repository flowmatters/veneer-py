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
            raise RuntimeError('access denied')

    bad, good = Unkillable(), FakeProc()
    with pytest.raises(RuntimeError, match='access denied'):
        manage.kill_all_now([bad, good])
    assert good.killed


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
    """Collects psutil.Process objects and kills any still running after the test, so a
    failing test (e.g. under mutation) leaves nothing behind."""
    procs = []
    yield procs
    for p in procs:
        try:
            if p.is_running():
                p.kill()
        except psutil.Error:
            pass


def _spawn_tree(tmp_path, name):
    """Start the stand-in script directly; return (parent psutil.Process, child pid)."""
    script = tmp_path / f'{name}.py'
    script.write_text(FAKE_VENEER_WITH_CHILD)
    pid_file = tmp_path / f'{name}.pid'
    proc = _REAL_POPEN([sys.executable, str(script), str(pid_file)],
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
        reap.append(psutil.Process(int(pid_file.read_text())))
        raise RuntimeError('probe exploded')
    monkeypatch.setattr(manage, '_probe_ready', probe)

    with pytest.raises(RuntimeError, match='probe exploded'):
        manage.start(veneer_exe=veneer_exe, ports=9990, quiet=True, gc=False,
                     startup_timeout=30.0, return_log_paths=False)

    child_pid = int(pid_file.read_text())
    assert started[0].poll() is not None, 'the stand-in VeneerCmd is still running'
    assert _wait_gone([child_pid]) == [], 'the stand-in VeneerCmd\'s own child is still running'


@windows_only
def test_kill_all_now_kills_the_whole_process_tree(tmp_path, reap):
    parent, child_pid = _spawn_tree(tmp_path, 'tree')
    reap.extend([parent, psutil.Process(child_pid)])

    manage.kill_all_now([parent.pid])

    assert _wait_gone([parent.pid, child_pid]) == []


# --- detached mode -------------------------------------------------------------------

def _detached_setup(monkeypatch, on_wait):
    """Run _start_detached without opening a terminal: the 'start ... cmd /k' Popen is
    recorded, and wait_for_status is replaced by on_wait(handshake_path)."""
    spawned = []
    monkeypatch.setattr(subprocess, 'Popen', lambda cmd, **k: spawned.append(cmd))
    monkeypatch.setattr(detached_handshake, 'wait_for_status',
                        lambda path, accept, timeout: on_wait(path))
    return spawned


def _starting(path, launcher_pid):
    detached_handshake.write_handshake(path, {
        'status': detached_handshake.HandshakeStatus.STARTING,
        'launcher_pid': launcher_pid, 'veneer_pids': [], 'actual_ports': [], 'error': None,
    })


@windows_only
@pytest.mark.parametrize('error', [TimeoutError('handshake timed out'), KeyboardInterrupt()])
def test_detached_wait_failing_kills_the_launcher_tree(tmp_path, monkeypatch, reap, error):
    launcher, launcher_child = _spawn_tree(tmp_path, 'launcher')
    reap.extend([launcher, psutil.Process(launcher_child)])

    def on_wait(path):
        _starting(path, launcher.pid)
        raise error
    spawned = _detached_setup(monkeypatch, on_wait)

    with pytest.raises(type(error)):
        _start(detached=True, detached_timeout=1.0)

    assert len(spawned) == 1
    assert _wait_gone([launcher.pid, launcher_child]) == []


@windows_only
def test_detached_failure_after_ready_kills_the_instances_and_launcher(
        tmp_path, monkeypatch, reap):
    launcher, launcher_child = _spawn_tree(tmp_path, 'launcher')
    instance, instance_child = _spawn_tree(tmp_path, 'instance')
    reap.extend([launcher, psutil.Process(launcher_child),
                 instance, psutil.Process(instance_child)])

    def on_wait(path):
        payload = {
            'status': detached_handshake.HandshakeStatus.READY,
            'launcher_pid': launcher.pid, 'veneer_pids': [instance.pid],
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
    assert _wait_gone([launcher.pid, launcher_child, instance.pid, instance_child]) == []


@windows_only
def test_detached_launcher_reporting_failure_is_left_to_exit_by_itself(
        tmp_path, monkeypatch, reap):
    """A FAILED handshake means the launcher's own start() raised (and cleaned up its own
    instances); the launcher then prints the traceback into its window and exits."""
    launcher, launcher_child = _spawn_tree(tmp_path, 'launcher')
    reap.extend([launcher, psutil.Process(launcher_child)])

    def on_wait(path):
        payload = {
            'status': detached_handshake.HandshakeStatus.FAILED,
            'launcher_pid': launcher.pid, 'veneer_pids': [], 'actual_ports': [],
            'error': 'Exception: One or more instances of Veneer failed to start.',
        }
        detached_handshake.write_handshake(path, payload)
        return payload
    _detached_setup(monkeypatch, on_wait)

    with pytest.raises(RuntimeError, match='Detached Veneer launcher failed'):
        _start(detached=True)

    assert launcher.is_running()
