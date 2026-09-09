"""Unit tests for the sandbox registry. IsolatedSource is faked throughout,
so no Source is required.
"""
import glob
import json
import os
import shutil
import tempfile
import pytest

import veneer.sandbox as sandbox


SANDBOX_PID = 4242          # the VeneerCmd process
CREATOR_PID = os.getpid()   # the short-lived Python process


class FakeProcess:
    def __init__(self, pid):
        self.pid = pid


class FakeIsolated:
    instances = []

    def __init__(self, project_file, related_files=None, **kwargs):
        self.project_file = project_file
        self.kwargs = kwargs
        self.directory = kwargs.get('tempdir_prefix', 'sbx') + 'dir'
        self.port = 41000            # deliberately NOT the requested port
        self.log_path = os.path.join(self.directory, 'veneer_logs', 'log.txt')
        self.v = object()
        self.process = FakeProcess(SANDBOX_PID)
        self.shutdown_called = False
        FakeIsolated.instances.append(self)

    def shutdown(self, clean=True):
        self.shutdown_called = True


class FailingIsolated:
    """Simulates a Source/VeneerCmd instance that never comes up. Used to
    verify the record left behind by a failed create() - status='failed'
    with its tempdir_prefix intact, so the retained veneer_logs/ can still be
    found even though there is no IsolatedSource object to read .log_path
    from.

    Mirrors the real IsolatedSource.__init__: it creates the temp directory
    (via tempdir_prefix) and writes into veneer_logs/ before the failure that
    raises - so a test can prove the tempdir_prefix in the surviving record
    actually leads to real logs on disk, not just that the kwarg was passed
    through.
    """

    def __init__(self, project_file, related_files=None, **kwargs):
        prefix = kwargs.get('tempdir_prefix', 'sbx')
        directory = tempfile.mkdtemp(prefix=prefix)
        logs_dir = os.path.join(directory, 'veneer_logs')
        os.makedirs(logs_dir)
        with open(os.path.join(logs_dir, 'log.txt'), 'w') as fp:
            fp.write('simulated startup failure log\n')
        raise RuntimeError('simulated Source startup failure')


@pytest.fixture
def cleanup_real_tempdirs():
    """FailingIsolated creates a real temp directory (mirroring the real
    IsolatedSource, which does the same before it raises). Clean up whatever
    it left behind in the system temp root so repeated test runs don't
    accumulate litter there."""
    pattern = os.path.join(tempfile.gettempdir(), 'source-sandbox-*')
    before = set(glob.glob(pattern))
    yield
    after = set(glob.glob(pattern))
    for d in after - before:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def killed(monkeypatch):
    """Record PIDs the registry tries to kill."""
    seen = []
    monkeypatch.setattr(sandbox, '_kill_pid', lambda pid: seen.append(pid))
    return seen


@pytest.fixture
def reg(tmp_path, monkeypatch, killed):
    FakeIsolated.instances = []
    monkeypatch.setattr(sandbox, 'IsolatedSource', FakeIsolated)
    monkeypatch.setattr(sandbox, 'Veneer', lambda port, **kw: ('client', port))
    monkeypatch.setattr(sandbox, '_pid_alive', lambda pid: pid == SANDBOX_PID)
    return sandbox.SandboxRegistry(project_dir=str(tmp_path), session_id='S1')


def test_create_records_actual_bound_port_not_requested(reg):
    sb = reg.create('scratch', 'model.rsproj', veneer_exe='cmd.exe')
    assert sb.port == 41000
    assert reg.list()[0]['port'] == 41000


def test_create_uses_on_clean_and_a_named_tempdir_prefix(reg):
    """Startup failure must leave logs behind, and the directory must be
    locatable when the constructor raised and returned no object."""
    reg.create('scratch', 'model.rsproj', veneer_exe='cmd.exe')
    kwargs = FakeIsolated.instances[0].kwargs
    assert kwargs['cleanup'] == 'on_clean'
    assert 'scratch' in kwargs['tempdir_prefix']


def test_create_does_not_request_the_gui_default_port(reg):
    reg.create('scratch', 'model.rsproj', veneer_exe='cmd.exe')
    assert FakeIsolated.instances[0].kwargs['port'] != 9876


def test_records_persist_and_reload(reg, tmp_path):
    reg.create('scratch', 'model.rsproj', veneer_exe='cmd.exe')
    fresh = sandbox.SandboxRegistry(project_dir=str(tmp_path), session_id='S1')
    assert [r['name'] for r in fresh.list()] == ['scratch']


def test_duplicate_name_rejected(reg):
    reg.create('scratch', 'model.rsproj', veneer_exe='cmd.exe')
    with pytest.raises(Exception):
        reg.create('scratch', 'model.rsproj', veneer_exe='cmd.exe')


def test_close_shuts_down_and_removes_record(reg):
    sb = reg.create('scratch', 'model.rsproj', veneer_exe='cmd.exe')
    reg.close('scratch')
    assert sb.shutdown_called
    assert reg.list() == []


def test_sweep_only_touches_this_session(reg, tmp_path):
    reg.create('mine', 'model.rsproj', veneer_exe='cmd.exe')
    other = sandbox.SandboxRegistry(project_dir=str(tmp_path), session_id='S2')
    other.create('theirs', 'model.rsproj', veneer_exe='cmd.exe')

    reg.sweep()
    names = [r['name'] for r in sandbox.SandboxRegistry(
        project_dir=str(tmp_path), session_id='S2').list()]
    assert names == ['theirs']


def test_sweep_without_session_id_refuses(tmp_path, monkeypatch):
    """A caller that was given no session id must never sweep - it would tear
    down another session's live sandboxes."""
    monkeypatch.setattr(sandbox, 'IsolatedSource', FakeIsolated)
    monkeypatch.delenv('VENEER_SESSION_ID', raising=False)
    anon = sandbox.SandboxRegistry(project_dir=str(tmp_path))
    with pytest.raises(Exception) as err:
        anon.sweep()
    assert 'VENEER_SESSION_ID' in str(err.value)


def test_session_id_is_never_adopted_from_the_file(reg, tmp_path, monkeypatch):
    """Reading an id back from the registry would let session B inherit A's id
    and sweep A's live sandboxes."""
    reg.create('mine', 'model.rsproj', veneer_exe='cmd.exe')
    monkeypatch.delenv('VENEER_SESSION_ID', raising=False)
    anon = sandbox.SandboxRegistry(project_dir=str(tmp_path))
    assert anon.session_id is None


def test_reap_orphans_removes_dead_sandboxes_from_any_session(reg, tmp_path, monkeypatch):
    reg.create('mine', 'model.rsproj', veneer_exe='cmd.exe')
    monkeypatch.setattr(sandbox, '_pid_alive', lambda pid: False)
    other = sandbox.SandboxRegistry(project_dir=str(tmp_path), session_id='S2')
    assert other.reap_orphans() == 1
    assert other.list() == []


def test_reap_orphans_leaves_live_records_alone(reg, tmp_path):
    reg.create('mine', 'model.rsproj', veneer_exe='cmd.exe')
    other = sandbox.SandboxRegistry(project_dir=str(tmp_path), session_id='S2')
    assert other.reap_orphans() == 0
    assert len(other.list()) == 1


def test_reap_keys_on_the_sandbox_process_not_the_creator(reg, tmp_path, monkeypatch):
    """The creating Python process is short-lived - one agent session spans many
    invocations. Reaping on it would destroy records for healthy sandboxes."""
    reg.create('mine', 'model.rsproj', veneer_exe='cmd.exe')
    record = reg.list()[0]
    assert record['veneer_pid'] == SANDBOX_PID
    assert record['owner_pid'] == CREATOR_PID

    # Creator gone, sandbox still running.
    monkeypatch.setattr(sandbox, '_pid_alive',
                        lambda pid: pid == SANDBOX_PID)
    later = sandbox.SandboxRegistry(project_dir=str(tmp_path), session_id='S1')
    assert later.reap_orphans() == 0
    assert len(later.list()) == 1


def test_get_attaches_to_a_sandbox_from_an_earlier_process(reg, tmp_path):
    """A fresh registry has no live object, but the sandbox is still running."""
    reg.create('mine', 'model.rsproj', veneer_exe='cmd.exe')
    later = sandbox.SandboxRegistry(project_dir=str(tmp_path), session_id='S1')
    handle = later.get('mine')
    assert handle is not None
    assert handle.port == 41000
    assert handle.v == ('client', 41000)


def test_get_returns_none_for_a_dead_sandbox(reg, tmp_path, monkeypatch):
    reg.create('mine', 'model.rsproj', veneer_exe='cmd.exe')
    monkeypatch.setattr(sandbox, '_pid_alive', lambda pid: False)
    later = sandbox.SandboxRegistry(project_dir=str(tmp_path), session_id='S1')
    assert later.get('mine') is None


def test_close_from_a_later_process_kills_the_sandbox(reg, tmp_path, killed):
    """Dropping the record without killing the process would orphan VeneerCmd
    AND destroy the record needed to find it again."""
    reg.create('mine', 'model.rsproj', veneer_exe='cmd.exe')
    later = sandbox.SandboxRegistry(project_dir=str(tmp_path), session_id='S1')
    later.close('mine')
    assert killed == [SANDBOX_PID]
    assert later.list() == []


def test_name_reusable_after_the_sandbox_dies(reg, monkeypatch):
    reg.create('mine', 'model.rsproj', veneer_exe='cmd.exe')
    monkeypatch.setattr(sandbox, '_pid_alive', lambda pid: False)
    reg.create('mine', 'model.rsproj', veneer_exe='cmd.exe')   # must not raise
    assert len(reg.list()) == 1


def test_record_carries_a_creation_time(reg):
    reg.create('mine', 'model.rsproj', veneer_exe='cmd.exe')
    assert reg.list()[0]['created'] > 0


def test_create_survives_startup_failure_and_leaves_a_findable_record(
        reg, monkeypatch, cleanup_real_tempdirs):
    """A failed IsolatedSource() raises with no object to read .directory or
    .log_path from. The record written before construction must survive with
    status='failed' and its tempdir_prefix, so the retained veneer_logs/ can
    still be found - and the exception must propagate rather than being
    swallowed."""
    monkeypatch.setattr(sandbox, 'IsolatedSource', FailingIsolated)
    with pytest.raises(RuntimeError):
        reg.create('mine', 'model.rsproj', veneer_exe='cmd.exe')

    records = reg.list()
    assert len(records) == 1
    record = records[0]
    assert record['status'] == sandbox.STATUS_FAILED
    assert 'mine' in record['tempdir_prefix']
    assert record['port'] is None
    assert record['veneer_pid'] is None

    # The name must be reusable afterwards, exactly as when the sandbox died.
    monkeypatch.setattr(sandbox, 'IsolatedSource', FakeIsolated)
    reg.create('mine', 'model.rsproj', veneer_exe='cmd.exe')
    records = reg.list()
    assert len(records) == 1
    assert records[0]['status'] == sandbox.STATUS_RUNNING


def test_write_retries_a_transient_permission_error_then_succeeds(reg, monkeypatch):
    """Simulates the Windows AV/indexer race: os.replace() fails with
    PermissionError a couple of times before succeeding. _write() must
    tolerate that and still land the record."""
    real_replace = sandbox.os.replace
    attempts = {'n': 0}

    def flaky_replace(src, dst):
        attempts['n'] += 1
        if attempts['n'] <= 2:
            raise PermissionError('simulated WinError 5: Access is denied')
        real_replace(src, dst)

    monkeypatch.setattr(sandbox.os, 'replace', flaky_replace)
    reg._write([{'name': 'x'}])

    assert attempts['n'] == 3
    assert reg._read() == [{'name': 'x'}]


def test_write_reraises_and_cleans_up_temp_file_on_persistent_permission_error(reg, monkeypatch):
    """A genuinely persistent lock (not a transient AV/indexer scan) must
    still fail loudly rather than silently losing data, and must not leave
    .tmp litter behind."""
    def always_fails(src, dst):
        raise PermissionError('simulated persistent WinError 5: Access is denied')

    monkeypatch.setattr(sandbox.os, 'replace', always_fails)
    with pytest.raises(PermissionError):
        reg._write([{'name': 'x'}])

    leftover = [f for f in os.listdir(reg.dir) if f.endswith('.tmp')]
    assert leftover == []


def test_startup_failure_leaves_findable_veneer_logs_via_tempdir_prefix(
        reg, monkeypatch, cleanup_real_tempdirs):
    """Proves the actual motivation for cleanup='on_clean' and the
    pre-construction record: after a failed start, the tempdir_prefix in the
    surviving record locates the real directory IsolatedSource created (and
    kept) before raising, and the veneer_logs/ inside it a human would
    inspect for the failure - not just that the kwarg was passed through."""
    monkeypatch.setattr(sandbox, 'IsolatedSource', FailingIsolated)
    with pytest.raises(RuntimeError):
        reg.create('mine', 'model.rsproj', veneer_exe='cmd.exe')

    record = reg.list()[0]
    prefix = record['tempdir_prefix']
    matches = glob.glob(os.path.join(tempfile.gettempdir(), prefix + '*'))
    assert len(matches) == 1
    log_file = os.path.join(matches[0], 'veneer_logs', 'log.txt')
    assert os.path.exists(log_file)


def test_create_does_not_clobber_concurrent_registry_changes_during_startup(reg, monkeypatch):
    """create()'s final write used to reuse the `records` list read before
    IsolatedSource() started, which can take seconds. If another process
    changed the registry during that window (its own create/close/
    reap_orphans), writing back the stale list would silently discard that
    change. Simulates a concurrent write happening mid-startup and asserts
    it survives alongside our own new record."""

    class ConcurrentIsolated(FakeIsolated):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            # Simulate another process creating its own sandbox while this
            # create() call was still waiting on IsolatedSource() to start.
            current = reg._read()
            current.append({
                'name': 'concurrent', 'session_id': 'S2', 'owner_pid': 999,
                'veneer_pid': 999, 'status': sandbox.STATUS_RUNNING,
                'port': 12345, 'directory': 'x', 'log_path': 'y',
                'tempdir_prefix': 'p', 'project_file': 'z', 'created': 0,
            })
            reg._write(current)

    monkeypatch.setattr(sandbox, 'IsolatedSource', ConcurrentIsolated)
    reg.create('mine', 'model.rsproj', veneer_exe='cmd.exe')

    names = sorted(r['name'] for r in reg.list())
    assert names == ['concurrent', 'mine']


def test_close_does_not_drop_record_when_kill_genuinely_fails(reg, tmp_path, monkeypatch):
    """close()'s docstring says dropping a record without actually killing
    its process would orphan a VeneerCmd instance AND destroy the record
    needed to find it again. If _kill_pid fails for a real reason (not
    'already gone'), close() must not reach its final write.

    Note this test replaces _kill_pid wholesale, so it deliberately does NOT
    discriminate _kill_pid's own internals - test_kill_pid_propagates_genuine_
    failures does that. What this guards is close()'s control flow: that
    close() does not re-swallow whatever _kill_pid lets through before its
    final write. Both are needed; neither subsumes the other."""
    reg.create('mine', 'model.rsproj', veneer_exe='cmd.exe')
    later = sandbox.SandboxRegistry(project_dir=str(tmp_path), session_id='S1')

    def failing_kill(pid):
        raise PermissionError('simulated AccessDenied - cannot kill this process')

    monkeypatch.setattr(sandbox, '_kill_pid', failing_kill)
    with pytest.raises(PermissionError):
        later.close('mine')

    assert [r['name'] for r in later.list()] == ['mine']


def test_kill_pid_treats_already_gone_as_success(monkeypatch):
    """psutil.NoSuchProcess means there is nothing left to kill - that is
    success, and must not raise."""
    import psutil

    class FakeProcess:
        def __init__(self, pid):
            raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, 'Process', FakeProcess)
    sandbox._kill_pid(999999)   # must not raise


def test_kill_pid_propagates_genuine_failures(monkeypatch):
    """A real failure to kill (e.g. AccessDenied) is not 'already gone' and
    must not be swallowed - close() depends on this to avoid dropping a
    record for a sandbox it failed to actually terminate."""
    import psutil

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def terminate(self):
            raise psutil.AccessDenied(self.pid)

    monkeypatch.setattr(psutil, 'Process', FakeProcess)
    with pytest.raises(psutil.AccessDenied):
        sandbox._kill_pid(999999)


def test_read_tolerates_a_corrupt_registry_file(tmp_path):
    veneer_dir = tmp_path / sandbox.REGISTRY_DIR
    veneer_dir.mkdir()
    (veneer_dir / sandbox.REGISTRY_FN).write_bytes(b'{not valid json!!!')

    reg = sandbox.SandboxRegistry(project_dir=str(tmp_path), session_id='S1')
    assert reg.list() == []
