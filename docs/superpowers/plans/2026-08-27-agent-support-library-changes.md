# Library Support for Automated and Sandboxed Workflows — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix a latent `save()` bug and add three helpers — a command-line builder, a sandbox registry, and a build identifier — so automated veneer-py workflows can experiment against throwaway copies of a live model.

**Architecture:** Four self-contained additions to the existing library. `save()` gains capture/restore of project metadata. `command_line_for()` derives Source and Veneer paths from a running instance's `/` endpoint and caches the merged command line. A new `veneer/sandbox.py` wraps `IsolatedSource` with a JSON-backed registry of named sandboxes. A build identifier makes "which veneer-py is this" answerable.

**Tech Stack:** Python 3, pytest, `requests`. No new dependencies. Tests run without Source by stubbing `run_script` / monkeypatching `veneer.manage` symbols.

**Spec:** `docs/superpowers/specs/2026-08-27-agent-support-library-changes.md`

**Branch:** `feature/agent-support-library-changes`, based on `master`.

---

## Before you start — context you will not guess

**1. Generated IronPython is dedented by a fragile rule.** `VeneerIronPython.clean_script` (`veneer/server_side.py:186`) measures the leading whitespace of the *whole script string* and strips `indent - 1` characters from **every** line:

```python
def clean_script(self, script):
    indent = len(script) - len(script.lstrip())
    if indent > 0:
        lines = [l[(indent - 1):] for l in script.splitlines()]
        return '\n'.join(lines)
    return script
```

The `save()` template is a triple-quoted string starting with a newline plus 17 spaces. So **any line you add, and any multi-line text you interpolate with `%s`, must carry exactly the same 17-space indentation** or it will be mangled. Task 1 tests this directly by compiling the cleaned output.

**2. `create_command_line` defaults to `force=True`** (`veneer/manage.py:133`), meaning it re-copies the entire Source distribution (~1144 files) unless you pass `force=False`. Never call it in a loop.

**3. `IsolatedSource` may not bind the port you ask for.** Read `.port` back from the object (`veneer/manage.py:386`). Its default is `port=9876` — the same default the Source GUI uses.

**4. `IsolatedSource.__init__` re-raises after teardown.** On startup failure there is *no object* to read `.log_path` or `.directory` from, and with the default `cleanup='always'` the temp directory is deleted. Construct with `cleanup='on_clean'` and a known `tempdir_prefix`.

**5. Test conventions already exist — follow them.** `test/test_isolated_source.py` (tracked) uses `monkeypatch.setattr(manage, 'start', fake_start)` with `tmp_path`; Tasks 2 and 3 follow it. Tasks 1 and 3 also use a small injected fake class rather than `unittest.mock` — match that style.

**Heads-up:** the other example of the fake-injection style, `test/test_recorder_sets.py`, is **untracked**, so it is NOT in this worktree or any clean checkout. Do not go looking for it; the fixtures you need are written out in full in each task below. (It exists only in the maintainer's main working tree.)

**6. Do not "fix" CLAUDE.md's testing sections.** They describe `veneer/testing/`, which lives on `feature/regression-test-framework` (20 commits ahead of `master`), not on this branch. Only the `objdict` line in Task 5 is safe to change.

**Run all tests with:** `python -m pytest test/ -v`

---

## File Structure

| File | Responsibility |
|---|---|
| `veneer/server_side.py` (modify, ~807) | `save()` gains `preserve_current` |
| `veneer/manage.py` (modify, append) | `command_line_for()` |
| `veneer/sandbox.py` (create) | `SandboxRegistry` over `IsolatedSource` |
| `veneer/__init__.py` (modify) | `__version__` and `build_info()` |
| `pyproject.toml` (modify) | `dynamic = ["version"]` |
| `CLAUDE.md` (modify) | `objdict` correction |
| `test/test_save_preserve_current.py` (create) | Task 1 |
| `test/test_command_line_for.py` (create) | Task 2 |
| `test/test_sandbox_registry.py` (create) | Task 3 |
| `test/test_build_info.py` (create) | Task 4 |

---

## Task 1: `save()` preserves the project's `OutputFile`

**The bug:** `save()` sets `ph.ProjectMetaStructure.OutputFile = ''` and its `finally` restores only `ph.CallBackHandler`. After any `save(fn=...)` the project's output file is blank, so a later `save(fn=None)` — documented as "save using current filename" — writes to `''`.

**Files:**
- Modify: `veneer/server_side.py:807-836`
- Test: `test/test_save_preserve_current.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_save_preserve_current.py`:

```python
"""Unit tests for VeneerIronPython.save() preserving project metadata.

Exercises script generation WITHOUT a live Source instance by stubbing run_script,
following the injection style of test/test_recorder_sets.py.
"""
import pytest

from veneer.server_side import VeneerIronPython


class FakeVeneer:
    """Stands in for the Veneer client; save() never touches it directly."""
    pass


@pytest.fixture
def ironpy():
    ip = VeneerIronPython(FakeVeneer())
    ip.scripts = []

    def fake_run_script(script):
        ip.scripts.append(script)
        return {'Exception': None, 'Response': None}

    ip.run_script = fake_run_script
    return ip


def test_save_captures_and_restores_output_file(ironpy):
    ironpy.save('C:/tmp/scratch.rsproj')
    script = ironpy.scripts[-1]
    assert 'saved_output_file = ph.ProjectMetaStructure.OutputFile' in script
    assert 'ph.ProjectMetaStructure.OutputFile = saved_output_file' in script


def test_save_captures_and_restores_project(ironpy):
    ironpy.save('C:/tmp/scratch.rsproj')
    script = ironpy.scripts[-1]
    assert 'saved_project = ph.ProjectMetaStructure.Project' in script
    assert 'ph.ProjectMetaStructure.Project = saved_project' in script


def test_save_can_opt_out_of_restoring(ironpy):
    ironpy.save('C:/tmp/scratch.rsproj', preserve_current=False)
    script = ironpy.scripts[-1]
    assert 'ph.ProjectMetaStructure.OutputFile = saved_output_file' not in script


def test_save_with_no_filename_still_reads_current_output_file(ironpy):
    """fn=None must resolve the filename BEFORE OutputFile is blanked."""
    ironpy.save()
    script = ironpy.scripts[-1]
    assert 'cb.OutputFileName=ph.ProjectMetaStructure.OutputFile' in script


@pytest.mark.parametrize('preserve', [True, False])
def test_generated_script_survives_clean_script(ironpy, preserve):
    """clean_script strips a fixed indent from every line; a mis-indented
    interpolation would produce syntactically invalid Python."""
    ironpy.save('C:/tmp/scratch.rsproj', preserve_current=preserve)
    cleaned = ironpy.clean_script(ironpy.scripts[-1])
    compile(cleaned, '<generated>', 'exec')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest test/test_save_preserve_current.py -v`

Expected: the three capture/restore tests FAIL (assertion errors — the strings are absent). The `fn=None` and `clean_script` tests should PASS already; that is fine and intentional — they are regression guards for behaviour you must not break.

- [ ] **Step 3: Implement**

Replace `save()` at `veneer/server_side.py:807`. **Give each restore line its own `%s`.** Do NOT build one multi-line restore string: that hard-codes leading spaces matching the template's indent, coupling two literals that must stay in agreement. A later re-indent could land those spaces on another valid indentation level, silently moving the restore statements out of `finally:` - valid Python, wrong behaviour, and every substring test still green. Separate placeholders inherit indentation from the template line they sit on, which is the idiom `cb.OutputFileName=%s` already uses one line above:

```python
    def save(self, fn=None, preserve_current=True):
        '''
        Save the current *project* to disk.

        fn - filename to save to. Should include .rsproj extension. If None, save using current filename

        preserve_current - (default True) restore the project's OutputFile and Project
                           after saving. Without this, saving to a new path leaves
                           OutputFile blank, so a later save(fn=None) writes to ''.
        '''
        if fn:
            fn = "'%s'" % os.path.abspath(fn).replace('\\', '\\\\')
        else:
            fn = 'ph.ProjectMetaStructure.OutputFile'
        if preserve_current:
            restore_output_file = 'ph.ProjectMetaStructure.OutputFile = saved_output_file'
            restore_project = 'ph.ProjectMetaStructure.Project = saved_project'
        else:
            restore_output_file = restore_project = 'pass'
        script = '''
                 from RiverSystem.ApplicationLayer.Consumers import DefaultCallback
                 from RiverSystem.ApplicationLayer.Creation import ProjectHandlerFactory
                 from RiverSystem import RiverSystemProject

                 ph = project_handler
                 cb = DefaultCallback()
                 cb.OutputFileName=%s
                 saved_cb = ph.CallBackHandler
                 saved_output_file = ph.ProjectMetaStructure.OutputFile
                 saved_project = ph.ProjectMetaStructure.Project
                 try:
                     ph.CallBackHandler = cb
                     ph.ProjectMetaStructure.Project = scenario.Project
                     ph.ProjectMetaStructure.OutputFile = ''
                     ph.ProjectMetaStructure.SaveProjectToFile = True
                     ph.SaveProject()
                 finally:
                     ph.CallBackHandler = saved_cb
                     %s
                     %s
                 ''' % (fn, restore_output_file, restore_project)
        return self._safe_run(script)
```

Note the ordering is load-bearing: `cb.OutputFileName=%s` is evaluated before `OutputFile` is blanked, which is what makes `fn=None` work.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest test/test_save_preserve_current.py -v`
Expected: 7 passed.

- [ ] **Step 5: Run the whole suite for regressions**

Run: `python -m pytest test/ -v`
Expected: 49 passed (42 baseline + 7 new), no failures.

- [ ] **Step 6: Commit**

```bash
git add veneer/server_side.py test/test_save_preserve_current.py
git commit -m "fix(server_side): save() no longer blanks the project OutputFile"
```

**Deferred to live verification (do NOT attempt here):** whether `SaveProject()` also repoints the application's current file. That needs a running instance and a throwaway project — never a production model. Record the finding when it is run.

---

## Task 2: `command_line_for()` — derive and cache a command line

> **Superseded in part — trust the code, not the block below.** Code review
> hardened this after the plan was written. Shipped behaviour differs: the cache
> stamp also records the Veneer DLL's mtime and size (so rebuilding the plugin in
> place busts the cache); the stamp read catches `OSError` as well as `ValueError`;
> file handles use `with`; the redundant `os.makedirs(cache_dir)` is gone because
> `create_command_line` creates `dest`. See `e60e223`, `e2e6728`, `f612c1f`.
> Retained as the record of original intent.

**Files:**
- Modify: `veneer/manage.py` (append after `create_command_line`, ~line 220)
- Test: `test/test_command_line_for.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_command_line_for.py`:

```python
"""Unit tests for command_line_for(), which derives Source/Veneer paths from a
running instance and caches the built command line. No Source required.
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
        exe = os.path.join(str(dest), 'FlowMatters.Source.VeneerCmd.exe')
        open(exe, 'w').write('EXE')
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest test/test_command_line_for.py -v`
Expected: all FAIL with `AttributeError: module 'veneer.manage' has no attribute 'command_line_for'`.

- [ ] **Step 3: Implement**

Append to `veneer/manage.py`:

```python
CMD_LINE_STAMP_FN = 'veneer_cmdline_stamp.json'
VENEER_PLUGIN_DLL = 'flowmatters.source.veneer.dll'


def command_line_for(v, cache_dir, force=False):
    '''
    Build (or reuse) a Veneer command line matching the Source instance behind client v.

    Derives the Source build directory and the Veneer plugin directory from the
    instance's own status (the `/` endpoint), so no Source version string is guessed
    and the command line cannot disagree with the Source actually running.

    v: a Veneer client connected to a running instance.
    cache_dir: directory to build into and reuse. Building copies the whole Source
               distribution, so this should be stable across sessions.
    force: rebuild even if a matching cached build exists.

    Returns: full path to FlowMatters.Source.VeneerCmd.exe
    '''
    status = v.status()
    source_path = _dirname(status['HostExe'])

    plugins = status.get('PluginsLoaded') or []
    veneer_dlls = [p for p in plugins
                   if _basename(p).lower() == VENEER_PLUGIN_DLL]
    if not veneer_dlls:
        raise Exception(
            'Veneer plugin not found in PluginsLoaded (%s). Cannot locate the '
            'Veneer files needed to build a command line.' % plugins)
    veneer_path = _dirname(veneer_dlls[0])

    stamp = {'SourceVersion': status.get('SourceVersion'),
             'HostExe': status['HostExe'],
             'veneer_path': veneer_path}

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    stamp_path = os.path.join(cache_dir, CMD_LINE_STAMP_FN)
    exe_path = os.path.join(cache_dir, VENEER_EXE_FN)

    if not force and os.path.exists(stamp_path) and os.path.exists(exe_path):
        try:
            if json.load(open(stamp_path)) == stamp:
                return exe_path
        except ValueError:
            pass  # unreadable stamp: fall through and rebuild

    result = create_command_line(veneer_path, source_version=None,
                                 source_path=source_path, dest=cache_dir,
                                 force=True)
    json.dump(stamp, open(stamp_path, 'w'))
    return str(result)
```

**You must add `import json`** to the top of `veneer/manage.py` — it is not currently imported (the module imports `sys`, `atexit`, `os`, `re`, `tempfile`, `shutil`, `threading`, `logging`). `VENEER_EXE_FN` already exists at `manage.py:23` as `'FlowMatters.Source.VeneerCmd.exe'`; use it rather than a literal.

Note also `MANY_VENEERS` and `VENEER_EXE` (`manage.py:24-25`) are hard-coded to one developer's `D:\src\projects\Veneer\...` paths. That is why the existing lookup is unreliable and why this helper derives everything from the running instance instead. Leave those constants alone — other code still references them.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest test/test_command_line_for.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add veneer/manage.py test/test_command_line_for.py
git commit -m "feat(manage): add command_line_for() deriving paths from a live instance"
```

---

## Task 3: Sandbox registry

> **Superseded in part — trust the code, not the block below.** Code review and a
> reproducible Windows flake changed this substantially after the plan was written.
> Shipped behaviour differs in ways that matter: `os.replace` carries a bounded
> retry (`mkstemp`+`os.replace` fails ~5% of the time here under AV/indexer
> contention); `create()` re-reads the registry before its final write rather than
> reusing a list held across `IsolatedSource` startup; and **`_kill_pid` no longer
> swallows genuine failures** — the version shown below does, which would let
> `close()` drop a record while leaving a live VeneerCmd orphaned. See `913bcc8`,
> `0b1e190`, `0c58477`, `5a02170`, `a393742`. Retained as the record of original intent.

**Files:**
- Create: `veneer/sandbox.py`
- Test: `test/test_sandbox_registry.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_sandbox_registry.py`:

```python
"""Unit tests for the sandbox registry. IsolatedSource is faked throughout,
so no Source is required.
"""
import json
import os
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest test/test_sandbox_registry.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'veneer.sandbox'`.

- [ ] **Step 3: Implement**

Create `veneer/sandbox.py`:

```python
'''
A registry of named, throwaway Source sandboxes.

Wraps veneer.manage.IsolatedSource so that several sandboxes can coexist, be
listed and reused across process invocations, and be torn down without
disturbing sandboxes belonging to other sessions.
'''
import json
import os
import random
import tempfile
import time

from .general import Veneer
from .manage import IsolatedSource

REGISTRY_DIR = '.veneer'
REGISTRY_FN = 'sandboxes.json'

PORT_RANGE = (40000, 60000)
GUI_DEFAULT_PORT = 9876

STATUS_RUNNING = 'running'
STATUS_FAILED = 'failed'


def _pid_alive(pid):
    if pid is None:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        return True     # cannot tell; assume alive and never reap


def _kill_pid(pid):
    try:
        import psutil
        psutil.Process(pid).terminate()
    except Exception:
        pass            # already gone, or we cannot reach it


def _random_port():
    port = random.randint(*PORT_RANGE)
    return port if port != GUI_DEFAULT_PORT else port + 1


class AttachedSandbox(object):
    '''
    A handle to a sandbox started by an EARLIER process.

    Exposes enough to use and to close the sandbox, but not the original
    IsolatedSource object, which died with the process that created it.
    '''

    def __init__(self, record):
        self.name = record['name']
        self.port = record['port']
        self.directory = record['directory']
        self.log_path = record['log_path']
        self.veneer_pid = record['veneer_pid']
        self.v = Veneer(record['port'])


class SandboxRegistry(object):
    '''
    Named sandboxes, recorded in <project_dir>/.veneer/sandboxes.json.

    session_id identifies the owning session. It is supplied by the caller (or
    read from VENEER_SESSION_ID) and is NEVER inferred from the registry file:
    adopting an id found there would let a second session inherit the first's
    identity and sweep its live sandboxes.

    With no session id a caller may create and use sandboxes, but may not sweep.
    '''

    def __init__(self, project_dir='.', session_id=None):
        self.dir = os.path.join(project_dir, REGISTRY_DIR)
        self.path = os.path.join(self.dir, REGISTRY_FN)
        self.session_id = (session_id if session_id is not None
                           else os.environ.get('VENEER_SESSION_ID'))
        self._live = {}

    def _read(self):
        if not os.path.exists(self.path):
            return []
        try:
            return json.load(open(self.path))
        except ValueError:
            return []

    def _write(self, records):
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self.dir, suffix='.tmp')
        with os.fdopen(tmp_fd, 'w') as fp:
            json.dump(records, fp, indent=2)
        os.replace(tmp_path, self.path)     # atomic

    def list(self):
        return self._read()

    def create(self, name, project_file, veneer_exe, related_files=None, **kwargs):
        records = self._read()
        existing = [r for r in records if r['name'] == name]
        for record in existing:
            # A record whose sandbox process is gone (or which never started) is
            # stale - reuse the name. A live one is a genuine clash.
            if record['status'] == STATUS_RUNNING and _pid_alive(record['veneer_pid']):
                raise Exception('A sandbox named %r is already running on port %s. '
                                'Close it first or choose another name.'
                                % (name, record['port']))
        records = [r for r in records if r['name'] != name]

        # Claim the name BEFORE starting, so a concurrent caller cannot take it.
        prefix = 'source-sandbox-%s-' % name
        record = {'name': name, 'session_id': self.session_id,
                  'owner_pid': os.getpid(),      # the Python process that created it
                  'veneer_pid': None,            # the VeneerCmd process itself
                  'status': STATUS_FAILED,       # promoted on success
                  'port': None, 'directory': None, 'log_path': None,
                  'tempdir_prefix': prefix, 'project_file': project_file,
                  'created': time.time()}
        records.append(record)
        self._write(records)

        # A failed start raises from the constructor, leaving no object to read
        # .directory or .log_path from. The record above survives with
        # status='failed' and the tempdir_prefix, so the retained veneer_logs/
        # can still be found. Deliberately not caught here.
        sb = IsolatedSource(project_file, related_files=related_files,
                            veneer_exe=veneer_exe,
                            port=kwargs.pop('port', _random_port()),
                            tempdir_prefix=prefix,
                            cleanup='on_clean',
                            **kwargs)

        record.update({'port': sb.port, 'directory': sb.directory,
                       'log_path': sb.log_path, 'status': STATUS_RUNNING,
                       'veneer_pid': getattr(sb.process, 'pid', None)})
        self._write(records)
        self._live[name] = sb
        return sb

    def get(self, name):
        '''
        The sandbox named `name`, or None.

        Returns the live IsolatedSource when this process created it, otherwise
        an AttachedSandbox reconstructed from the record - so a sandbox created
        by an earlier invocation is still usable.
        '''
        if name in self._live:
            return self._live[name]
        for record in self._read():
            if record['name'] == name and record['status'] == STATUS_RUNNING:
                if _pid_alive(record['veneer_pid']):
                    return AttachedSandbox(record)
        return None

    def close(self, name):
        '''
        Shut a sandbox down and drop its record, whether or not this process
        created it. Closing a record without killing its process would orphan a
        VeneerCmd instance AND destroy the record needed to find it again.
        '''
        sb = self._live.pop(name, None)
        if sb is not None:
            sb.shutdown()
        else:
            for record in self._read():
                if record['name'] == name and record['veneer_pid'] is not None:
                    _kill_pid(record['veneer_pid'])
        self._write([r for r in self._read() if r['name'] != name])

    def sweep(self):
        '''Tear down every sandbox belonging to THIS session.'''
        if self.session_id is None:
            raise Exception(
                'Refusing to sweep: no session id. Set VENEER_SESSION_ID (or pass '
                'session_id=) so this only tears down its own sandboxes.')
        mine = [r for r in self._read() if r['session_id'] == self.session_id]
        for record in mine:
            self.close(record['name'])
        return len(mine)

    def reap_orphans(self):
        '''
        Drop records whose SANDBOX process is gone. Safe from any session,
        because a dead sandbox cannot be reclaimed by anyone.

        Keyed on veneer_pid, NOT owner_pid. The creating Python process is
        short-lived - a single agent session spans many invocations - so
        reaping on the owner would destroy records for perfectly healthy
        sandboxes moments after they were created.
        '''
        records = self._read()
        keep = [r for r in records if _pid_alive(r['veneer_pid'])]
        if len(keep) != len(records):
            self._write(keep)
        return len(records) - len(keep)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest test/test_sandbox_registry.py -v`
Expected: 17 passed.

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest test/ -v`

- [ ] **Step 6: Commit**

```bash
git add veneer/sandbox.py test/test_sandbox_registry.py
git commit -m "feat(sandbox): add a registry of named IsolatedSource sandboxes"
```

---

## Task 4: A real build identifier

> **Superseded in part — trust the code, not the block below.** Code review changed
> this after the plan was written: `build_info()` now returns `__version__` for the
> `version` key in **every** branch, rather than sourcing it from
> `importlib.metadata` in the `distribution` branch — otherwise the same key means
> different things depending on `source`, which undercuts the point of the feature.
> The metadata call is retained purely as an existence probe. A docstring caveat
> notes that `git rev-parse` can report an unrelated enclosing repository's SHA.
> See `6512103`. Retained as the record of original intent.

**The problem:** there is no `veneer.__version__`, and `pyproject.toml` declares a static `version = "0.1"` that has never moved — so every install reports `0.1` and any build comparison is inert.

**Files:**
- Modify: `veneer/__init__.py`
- Modify: `pyproject.toml`
- Test: `test/test_build_info.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_build_info.py`:

```python
"""Unit tests for the build identifier."""
import veneer


def test_version_attribute_exists():
    assert isinstance(veneer.__version__, str)
    assert veneer.__version__


def test_build_info_reports_version_and_provenance():
    info = veneer.build_info()
    assert info['version'] == veneer.__version__
    assert info['source'] in ('git', 'distribution', 'unknown')


def test_build_info_includes_revision_key_even_when_unknown():
    """Consumers must be able to distinguish 'differs' from 'cannot tell',
    so the key is always present."""
    assert 'revision' in veneer.build_info()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest test/test_build_info.py -v`
Expected: FAIL with `AttributeError: module 'veneer' has no attribute '__version__'`.

- [ ] **Step 3: Implement**

Add near the top of `veneer/__init__.py`:

```python
__version__ = '0.2.0'


def build_info():
    '''
    Identify which build of veneer-py is in use.

    Returns a dict with:
      version  - the declared version
      revision - git SHA when running from a source checkout, else None
      source   - 'git', 'distribution', or 'unknown'; says HOW the revision was
                 obtained, so callers can tell "differs" from "cannot tell"
    '''
    import os
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=here,
            stderr=subprocess.DEVNULL).decode().strip()
        return {'version': __version__, 'revision': sha, 'source': 'git'}
    except Exception:
        pass

    try:
        from importlib.metadata import version as _dist_version
        return {'version': _dist_version('veneer-py'),
                'revision': None, 'source': 'distribution'}
    except Exception:
        return {'version': __version__, 'revision': None, 'source': 'unknown'}
```

**`build_info()`'s `distribution` branch deliberately returns no install timestamp**, although the spec mentions one. The version plus the explicit `source` key already lets a caller distinguish "differs" from "cannot tell", which is the actual requirement; a timestamp adds a field nothing reads. If a real need for it appears, add it then.

**Warning before editing `pyproject.toml`: the file ends without a trailing newline** (its last line is `py_modules=['veneer']`). Appending a table blindly produces `py_modules=['veneer'][tool.setuptools.dynamic]` and a TOML parse error. Add the newline first.

Then in `pyproject.toml`, replace the static version so there is only one source of truth:

```toml
dynamic = ["version"]
```

removing the `version = "0.1"` line from `[project]`, and adding:

```toml
[tool.setuptools.dynamic]
version = {attr = "veneer.__version__"}
```

**Do not touch** `[project.optional-dependencies]` or add entry-point tables — `feature/regression-test-framework` modifies that region and this change must not collide with it.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest test/test_build_info.py -v`
Expected: 3 passed.

- [ ] **Step 5: Verify the packaging change is valid**

Run: `python -c "import tomllib,sys; tomllib.load(open('pyproject.toml','rb')); print('pyproject parses')"`
Expected: `pyproject parses`

Run: `pip install -e . && python -c "import importlib.metadata as m; print(m.version('veneer-py'))"`
Expected: `0.2.0` — confirming the dynamic version resolves rather than reporting the old static `0.1`.

- [ ] **Step 6: Commit**

```bash
git add veneer/__init__.py pyproject.toml test/test_build_info.py
git commit -m "feat: add veneer.__version__ and build_info()"
```

---

## Task 5: Correct the `objdict` description in CLAUDE.md

`veneer/utils.py:71` is `def objdict(orig): return UserDict(orig)`, with the attribute-access class commented out beneath it. CLAUDE.md still calls it "dict with attribute access", so anyone following it writes `result.name` and gets an `AttributeError`.

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Confirm the current behaviour before changing the docs**

Run:
```bash
python -c "from veneer.utils import objdict; d = objdict({'a': 1}); print(d['a']); print(getattr(d, 'a', 'NO ATTRIBUTE ACCESS'))"
```
Expected: `1` then `NO ATTRIBUTE ACCESS`.

- [ ] **Step 2: Edit the description**

In `CLAUDE.md`, under "Supporting modules", change:

> `veneer/utils.py` — `SearchableList` (lists with `find_by_*` methods), `objdict` (dict with attribute access), CSV/date utilities

to:

> `veneer/utils.py` — `SearchableList` (lists with `find_by_*` methods), `objdict` (currently a thin `UserDict` wrapper — **not** attribute access, despite the name), CSV/date utilities

- [ ] **Step 3: Verify you changed only that line**

**`CLAUDE.md` is untracked** — it has never been committed on any branch
(`git log --all -- CLAUDE.md` is empty). So `git diff` shows **nothing** for it, and
that is expected, not a failed check. Verify with `grep` instead:

```bash
grep -c "not\*\* attribute access" CLAUDE.md      # expect 1
grep -c "dict with attribute access" CLAUDE.md    # expect 0
```

Do **not** touch the Testing section or the `veneer/testing/` architecture entry — they describe committed work on `feature/regression-test-framework`, not errors.

- [ ] **Step 4: Do NOT commit `CLAUDE.md`**

Leave the edit local and uncommitted, matching its current state.

`git add CLAUDE.md` would newly track the whole 103-line file, committing a full
description of the regression-test framework onto a branch where `veneer/testing/`
does not exist — exactly the merge-base hazard in note #6. Whether `CLAUDE.md` should
be tracked at all is a separate decision for the maintainer, and not this branch's to
make.

There is no commit for this task.

---

## Final verification

- [ ] Run the full suite: `python -m pytest test/ -v`

Expected: the **42** tests present at the branch point, plus 33 new (7 + 6 + 17 + 3) = **75 passed**.

(Measured in a clean worktree of this branch after rebasing onto `master` at `1437286`. A count of 50 means you are in the maintainer's main working tree, which additionally has the untracked `test/test_recorder_sets.py` — see note #5.)

- [ ] Confirm no unintended files changed: `git diff --stat master...HEAD`
- [ ] Expected **committed** files: `veneer/server_side.py`, `veneer/manage.py`, `veneer/sandbox.py`, `veneer/__init__.py`, `pyproject.toml`, plus four new test files.
- [ ] `CLAUDE.md` should appear as **untracked and modified**, not committed (Task 5). Confirm with `git status --short CLAUDE.md` → `?? CLAUDE.md`.

## Deferred — needs a live Source instance

These cannot be done here and should be scheduled separately, against a **throwaway project, never a production model**:

- Whether `SaveProject()` repoints the application's current file (Task 1).
- That `command_line_for()` produces a command line that actually starts (Task 2).
- That the registry starts real sandboxes on non-colliding ports and preserves `veneer_logs/` when startup fails (Task 3).
