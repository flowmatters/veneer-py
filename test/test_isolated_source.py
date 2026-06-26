import os
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
