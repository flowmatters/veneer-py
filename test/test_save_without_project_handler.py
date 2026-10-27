"""Unit tests for VeneerIronPython.save() when the server gives scripts no project handler.

Runs the generated script under CPython, with stub RiverSystem and System modules, so the
behaviour is exercised WITHOUT a live Source instance.
"""
import builtins
import types

import pytest

from veneer.server_side import VeneerIronPython


class FakeVeneer:
    pass


class Stub:
    """Accepts any attribute assignment; records calls to SaveProject."""
    def __init__(self, **attrs):
        self.__dict__.update(attrs)
        self.saved = 0

    def SaveProject(self):
        self.saved += 1


def _stub_modules():
    process = Stub(MainModule=Stub(FileName='C:/Somewhere/Host.exe'), Id=1234)
    diagnostics = types.SimpleNamespace(
        Process=types.SimpleNamespace(GetCurrentProcess=lambda: process))
    return {
        'RiverSystem.ApplicationLayer.Consumers': types.SimpleNamespace(DefaultCallback=Stub),
        'RiverSystem.ApplicationLayer.Creation': types.SimpleNamespace(ProjectHandlerFactory=None),
        'RiverSystem': types.SimpleNamespace(RiverSystemProject=None),
        'System': types.SimpleNamespace(Diagnostics=diagnostics),
    }


def run_generated(script, project_handler):
    """Execute a generated save script with the given project_handler in scope."""
    modules = _stub_modules()
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in modules:
            return modules[name]
        return real_import(name, globals, locals, fromlist, level)

    scope = {
        '__builtins__': dict(vars(builtins), __import__=fake_import),
        'project_handler': project_handler,
        'scenario': Stub(Project='the scenario project'),
    }
    exec(compile(script, '<generated>', 'exec'), scope)
    return scope


@pytest.fixture
def ironpy():
    ip = VeneerIronPython(FakeVeneer())
    ip.scripts = []

    def fake_run_script(script):
        ip.scripts.append(ip.clean_script(script))
        return {'Exception': None, 'Response': None}

    ip.run_script = fake_run_script
    return ip


@pytest.mark.parametrize('fn', ['C:/tmp/scratch.rsproj', None])
def test_save_without_project_handler_raises_a_clear_error(ironpy, fn):
    ironpy.save(fn)
    with pytest.raises(Exception) as err:
        run_generated(ironpy.scripts[-1], project_handler=None)
    message = str(err.value)
    assert 'project_handler is None' in message
    assert 'NoneType' not in message


def test_save_without_project_handler_names_the_server_process(ironpy):
    """The process tells the caller which Veneer actually answered, and from which build."""
    ironpy.save('C:/tmp/scratch.rsproj')
    with pytest.raises(Exception) as err:
        run_generated(ironpy.scripts[-1], project_handler=None)
    assert 'C:/Somewhere/Host.exe' in str(err.value)
    assert '1234' in str(err.value)


def test_save_with_a_project_handler_still_saves_and_restores(ironpy):
    ironpy.save('C:/tmp/scratch.rsproj')
    original_cb = Stub()
    handler = Stub(CallBackHandler=original_cb,
                   ProjectMetaStructure=Stub(OutputFile='C:/original.rsproj', Project='original'))
    run_generated(ironpy.scripts[-1], project_handler=handler)
    assert handler.saved == 1
    assert handler.CallBackHandler is original_cb
    assert handler.ProjectMetaStructure.OutputFile == 'C:/original.rsproj'
    assert handler.ProjectMetaStructure.Project == 'original'
