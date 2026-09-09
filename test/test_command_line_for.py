"""Unit tests for command_line_for(), which derives Source/Veneer paths from a
running instance and caches the built command line. No Source required.

Status payloads below use Windows backslash paths deliberately: command_line_for()
is inherently a same-machine, Windows-only operation (it builds a Windows .exe
command line), and _dirname/_basename parse with the client OS's path conventions.
"""
import json
import os
import pytest

import veneer.manage as manage

STATUS = {
    'HostExe': r'C:\Source\RiverSystem.Forms.exe',
    'PluginsLoaded': [
        r'C:\Plugins\Custom Functions.dll',
        r'C:\Plugins\Veneer\FlowMatters.Source.Veneer.dll',
    ],
    'SourceVersion': '6.10.0.14373',
}


class FakeClient:
    def __init__(self, status=None):
        self._status = status or STATUS

    def status(self):
        return self._status


@pytest.fixture
def built(monkeypatch):
    """Record create_command_line calls and fake the resulting exe."""
    calls = []

    def fake_create(veneer_path, source_version=None, source_path=None,
                    dest=None, force=True, init_db=False):
        calls.append({'veneer_path': veneer_path, 'source_version': source_version,
                      'source_path': source_path, 'dest': dest, 'force': force})
        os.makedirs(str(dest), exist_ok=True)  # mirrors create_command_line's dest.mkdir(parents=True)
        exe = os.path.join(str(dest), 'FlowMatters.Source.VeneerCmd.exe')
        with open(exe, 'w') as f:
            f.write('EXE')
        return exe

    monkeypatch.setattr(manage, 'create_command_line', fake_create)
    return calls


def test_derives_source_and_veneer_paths_from_status(built, tmp_path):
    manage.command_line_for(FakeClient(), cache_dir=str(tmp_path))
    assert built[0]['source_path'] == r'C:\Source'
    assert built[0]['veneer_path'] == r'C:\Plugins\Veneer'


def test_passes_no_source_version(built, tmp_path):
    """source_version=None makes create_command_line treat source_path as the
    build directory directly, so no version string is ever guessed."""
    manage.command_line_for(FakeClient(), cache_dir=str(tmp_path))
    assert built[0]['source_version'] is None


def test_second_call_reuses_cache(built, tmp_path):
    manage.command_line_for(FakeClient(), cache_dir=str(tmp_path))
    manage.command_line_for(FakeClient(), cache_dir=str(tmp_path))
    assert len(built) == 1


def test_force_rebuilds(built, tmp_path):
    manage.command_line_for(FakeClient(), cache_dir=str(tmp_path))
    manage.command_line_for(FakeClient(), cache_dir=str(tmp_path), force=True)
    assert len(built) == 2


def test_different_source_version_rebuilds(built, tmp_path):
    manage.command_line_for(FakeClient(), cache_dir=str(tmp_path))
    other = dict(STATUS, SourceVersion='6.11.0.1')
    manage.command_line_for(FakeClient(other), cache_dir=str(tmp_path))
    assert len(built) == 2


def test_missing_veneer_plugin_raises(built, tmp_path):
    bad = dict(STATUS, PluginsLoaded=[r'C:\Plugins\Custom Functions.dll'])
    with pytest.raises(Exception) as err:
        manage.command_line_for(FakeClient(bad), cache_dir=str(tmp_path))
    assert 'Veneer plugin' in str(err.value)


def test_corrupt_stamp_falls_back_to_rebuild(built, tmp_path):
    manage.command_line_for(FakeClient(), cache_dir=str(tmp_path))
    stamp_path = os.path.join(str(tmp_path), manage.CMD_LINE_STAMP_FN)
    with open(stamp_path, 'w') as f:
        f.write('not valid json {')
    manage.command_line_for(FakeClient(), cache_dir=str(tmp_path))
    assert len(built) == 2


def test_unreadable_stamp_falls_back_to_rebuild(built, tmp_path, monkeypatch):
    """Covers the OSError branch specifically: malformed JSON above raises
    ValueError, which was already handled before that branch was added. Here
    reading the stamp raises OSError (e.g. permission denied), which must also
    fall through to a rebuild rather than propagating."""
    manage.command_line_for(FakeClient(), cache_dir=str(tmp_path))
    stamp_path = os.path.join(str(tmp_path), manage.CMD_LINE_STAMP_FN)

    real_open = open

    def flaky_open(file, mode='r', *args, **kwargs):
        if str(file) == stamp_path and 'r' in mode:
            raise PermissionError('simulated permission error reading stamp')
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr('builtins.open', flaky_open)
    manage.command_line_for(FakeClient(), cache_dir=str(tmp_path))
    assert len(built) == 2


def test_missing_exe_triggers_rebuild(built, tmp_path):
    manage.command_line_for(FakeClient(), cache_dir=str(tmp_path))
    exe_path = os.path.join(str(tmp_path), 'FlowMatters.Source.VeneerCmd.exe')
    os.remove(exe_path)
    manage.command_line_for(FakeClient(), cache_dir=str(tmp_path))
    assert len(built) == 2


def test_changed_veneer_dll_busts_cache(built, tmp_path):
    """An in-place rebuild of the Veneer plugin DLL (unchanged path, unchanged
    SourceVersion) must still be detected via mtime/size, since that is exactly
    the workflow veneer-py developers hit when iterating on the plugin."""
    plugin_dir = tmp_path / 'plugin'
    plugin_dir.mkdir()
    dll_path = plugin_dir / 'FlowMatters.Source.Veneer.dll'
    dll_path.write_bytes(b'v1')
    status = dict(STATUS, PluginsLoaded=[r'C:\Plugins\Custom Functions.dll', str(dll_path)])
    cache_dir = tmp_path / 'cache'

    manage.command_line_for(FakeClient(status), cache_dir=str(cache_dir))
    dll_path.write_bytes(b'v2 - rebuilt, different size')
    manage.command_line_for(FakeClient(status), cache_dir=str(cache_dir))
    assert len(built) == 2


def test_unreadable_veneer_dll_stat_does_not_crash(built, tmp_path):
    """The DLL path may not be statable from this machine (e.g. a client on a
    different host); command_line_for should still succeed rather than raising."""
    missing_dll = tmp_path / 'nowhere' / 'FlowMatters.Source.Veneer.dll'
    status = dict(STATUS, PluginsLoaded=[r'C:\Plugins\Custom Functions.dll', str(missing_dll)])
    cache_dir = tmp_path / 'cache'
    manage.command_line_for(FakeClient(status), cache_dir=str(cache_dir))
    assert len(built) == 1


def test_builds_when_cache_dir_does_not_exist(built, tmp_path):
    """The first-ever call gets a cache_dir that isn't there yet. Production
    relies on create_command_line creating it rather than doing so itself."""
    cache_dir = tmp_path / 'not-created-yet'
    assert not cache_dir.exists()
    exe = manage.command_line_for(FakeClient(), cache_dir=str(cache_dir))
    assert len(built) == 1
    assert os.path.exists(exe)


def test_stamp_write_retries_a_transient_permission_error_then_succeeds(built, tmp_path, monkeypatch):
    """Mirrors the SandboxRegistry test for the same underlying helper
    (write_json_atomic / _replace_with_retry in veneer.manage). The stamp
    write follows the expensive create_command_line rebuild, so a transient
    AV/indexer PermissionError on the atomic rename here must not both raise
    AND throw away a build that just succeeded - the cache must still land."""
    real_replace = manage.os.replace
    attempts = {'n': 0}

    def flaky_replace(src, dst):
        attempts['n'] += 1
        if attempts['n'] <= 2:
            raise PermissionError('simulated WinError 5: Access is denied')
        real_replace(src, dst)

    monkeypatch.setattr(manage.os, 'replace', flaky_replace)
    exe = manage.command_line_for(FakeClient(), cache_dir=str(tmp_path))

    assert attempts['n'] == 3
    assert os.path.exists(exe)

    # The build must not have been thrown away: a second call still hits the
    # cache rather than rebuilding.
    manage.command_line_for(FakeClient(), cache_dir=str(tmp_path))
    assert len(built) == 1
