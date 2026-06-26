import os
import shutil
import pytest
import veneer.manage as manage


class FakeProc:
    pass


@pytest.fixture
def project(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    proj = src / 'model.rsproj'
    proj.write_text('PROJECT')
    (src / 'climate.csv').write_text('DATA')
    return str(proj)


@pytest.fixture
def patched(monkeypatch):
    state = {'started': {}, 'killed': [], 'clients': []}

    def fake_start(**kwargs):
        state['started'] = kwargs
        return [FakeProc()], [9999], ['C:/tmp/veneer.log']

    def fake_kill(proc):
        state['killed'].append(proc)

    class FakeVeneer:
        def __init__(self, port, **kw):
            self.port = port
            self.kw = kw
            state['clients'].append(self)

    monkeypatch.setattr(manage, 'start', fake_start)
    monkeypatch.setattr(manage, 'kill_all_now', fake_kill)
    monkeypatch.setattr(manage, 'Veneer', FakeVeneer)
    return state


def test_lifecycle_copies_starts_and_exposes_attrs(project, patched):
    with manage.IsolatedSource(project, related_files=['climate.csv'],
                               veneer_exe='X') as s:
        assert os.path.isdir(s.directory)
        assert os.path.exists(os.path.join(s.directory, 'model.rsproj'))
        assert os.path.exists(os.path.join(s.directory, 'climate.csv'))
        assert s.project_file == os.path.join(s.directory, 'model.rsproj')
        assert s.port == 9999
        assert s.log_path == 'C:/tmp/veneer.log'
        assert s.v.port == 9999
        saved_dir = s.directory

    # context-manager exit with default cleanup='always': killed + removed
    assert len(patched['killed']) == 1
    assert not os.path.exists(saved_dir)


def test_start_receives_copied_project_fn(project, patched):
    with manage.IsolatedSource(project, veneer_exe='X'):
        pass
    started = patched['started']
    # passed as project_fn (so overwrite_plugins works), pointing at the copy
    assert os.path.basename(started['project_fn']) == 'model.rsproj'
    assert started['project_fn'] != project
    assert started['n_instances'] == 1
    assert started['return_log_paths'] is True


def test_shutdown_is_idempotent(project, patched):
    s = manage.IsolatedSource(project, veneer_exe='X')
    s.shutdown()
    s.shutdown()  # must not raise or double-kill
    assert len(patched['killed']) == 1


def test_isolated_copy_is_alias():
    assert manage.isolated_copy is manage.IsolatedSource


def test_normalise_cleanup_aliases_and_validation():
    assert manage._normalise_cleanup(True) == 'always'
    assert manage._normalise_cleanup(False) == 'never'
    assert manage._normalise_cleanup('on_clean') == 'on_clean'
    with pytest.raises(ValueError):
        manage._normalise_cleanup('sometimes')


def test_cleanup_never_keeps_dir(project, patched):
    s = manage.IsolatedSource(project, veneer_exe='X', cleanup='never')
    d = s.directory
    s.shutdown()
    assert len(patched['killed']) == 1   # process always killed
    assert os.path.exists(d)             # dir kept
    shutil.rmtree(d)


def test_cleanup_on_clean_removes_on_success(project, patched):
    with manage.IsolatedSource(project, veneer_exe='X', cleanup='on_clean') as s:
        d = s.directory
    assert not os.path.exists(d)


def test_cleanup_on_clean_keeps_on_error(project, patched):
    s = manage.IsolatedSource(project, veneer_exe='X', cleanup='on_clean')
    d = s.directory
    with pytest.raises(RuntimeError):
        with s:
            raise RuntimeError('boom')
    assert os.path.exists(d)             # kept because exit was not clean
    shutil.rmtree(d)


def _spy_mkdtemp(monkeypatch, holder):
    real = manage.tempfile.mkdtemp
    def spy(*a, **k):
        d = real(*a, **k)
        holder['dir'] = d
        return d
    monkeypatch.setattr(manage.tempfile, 'mkdtemp', spy)


def test_startup_failure_removes_dir_under_always(project, monkeypatch):
    def boom(**kwargs):
        raise Exception('start failed')
    monkeypatch.setattr(manage, 'start', boom)
    holder = {}
    _spy_mkdtemp(monkeypatch, holder)
    with pytest.raises(Exception):
        manage.IsolatedSource(project, veneer_exe='X')  # default cleanup='always'
    assert not os.path.exists(holder['dir'])


def test_startup_failure_keeps_dir_under_on_clean(project, monkeypatch):
    def boom(**kwargs):
        raise Exception('start failed')
    monkeypatch.setattr(manage, 'start', boom)
    holder = {}
    _spy_mkdtemp(monkeypatch, holder)
    with pytest.raises(Exception):
        manage.IsolatedSource(project, veneer_exe='X', cleanup='on_clean')
    assert os.path.exists(holder['dir'])
    shutil.rmtree(holder['dir'])
