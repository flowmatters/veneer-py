# Isolated single-instance Source sandbox — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a helper for running a single Veneer instance against an isolated, throwaway copy of a Source project, sharing the cluster's temp-copy logic.

**Architecture:** Extract the pure file-copy body out of the dask-decorated `cluster.copy_project()` into a plain `copy_project_files()` helper in `manage.py`; the cluster wrapper delegates to it. Add an `IsolatedSource` class in `manage.py` that copies the project + related files to a temp dir, starts one VeneerCmd instance (via `manage.start`), exposes a ready `Veneer` client plus the sandbox path, and tears down per a three-way cleanup policy. It is usable as a context manager and inline.

**Tech Stack:** Python, pytest (with `monkeypatch`/`tmp_path` — no live Source needed for tests, following the existing `test/test_start_return_shape.py` pattern), `shutil`/`tempfile`, existing `veneer.manage.start` / `veneer.manage.kill_all_now`.

**Spec:** `docs/superpowers/specs/2026-06-26-isolated-source-sandbox-design.md`

---

## File Structure

- **Modify** `veneer/manage.py`:
  - Add `copy_project_files(project_file, extras, dest)` — pure copy helper.
  - Add `_normalise_cleanup(cleanup)` — validate/alias the cleanup policy.
  - Add `IsolatedSource` class + `isolated_copy = IsolatedSource` alias.
- **Modify** `veneer/cluster.py`:
  - Import `copy_project_files` from `.manage`; rewrite `copy_project` to delegate.
- **Create** `test/test_copy_project_files.py` — pure tests for the helper + a cluster-parity test.
- **Create** `test/test_isolated_source.py` — monkeypatched lifecycle + cleanup-policy tests.

**Conventions to follow** (from existing code):
- Tests live flat in `test/`, named `test_*.py`, use `monkeypatch`/`tmp_path`, and never spawn real VeneerCmd (see `test/test_start_return_shape.py`).
- `manage.py` already imports `os`, `shutil`, `tempfile`, `logger`, and `from .general import Veneer`. No new imports are needed there.
- `kill_all_now` already accepts a single `Popen`/`Process` or a list.

---

## Task 1: Extract `copy_project_files` pure helper

**Files:**
- Test: `test/test_copy_project_files.py`
- Modify: `veneer/manage.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_copy_project_files.py`:

```python
import os
import pytest
import veneer.manage as manage


def test_copies_project_and_extras(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    proj = src / 'model.rsproj'
    proj.write_text('PROJECT')
    (src / 'climate.csv').write_text('DATA')
    inputs = src / 'inputs'
    inputs.mkdir()
    (inputs / 'a.txt').write_text('A')

    dest = tmp_path / 'dest'
    dest.mkdir()

    result = manage.copy_project_files(str(proj), ['climate.csv', 'inputs'], str(dest))

    assert result == os.path.join(str(dest), 'model.rsproj')
    assert (dest / 'model.rsproj').read_text() == 'PROJECT'
    assert (dest / 'climate.csv').read_text() == 'DATA'
    assert (dest / 'inputs' / 'a.txt').read_text() == 'A'


def test_missing_extra_raises(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    proj = src / 'model.rsproj'
    proj.write_text('P')
    dest = tmp_path / 'dest'
    dest.mkdir()
    with pytest.raises(AssertionError):
        manage.copy_project_files(str(proj), ['nope.csv'], str(dest))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_copy_project_files.py -v`
Expected: FAIL with `AttributeError: module 'veneer.manage' has no attribute 'copy_project_files'`.

- [ ] **Step 3: Write minimal implementation**

In `veneer/manage.py`, add (near the other module-level helpers, e.g. just above `def start(`):

```python
def copy_project_files(project_file, extras, dest):
    """Copy project_file and each extra into dest (an existing directory).

    project_file: path to the project file (e.g. .rsproj). Its basename is
                  written directly into dest.
    extras: iterable of paths, each resolved relative to the project file's
            directory and copied into dest preserving that relative path.
            Directories are copied recursively.
    dest: an existing destination directory.

    Returns the full path to the copied project file inside dest.
    """
    assert os.path.exists(project_file)
    dest_fn = os.path.join(dest, os.path.basename(project_file))
    shutil.copyfile(project_file, dest_fn)
    src_dir = os.path.dirname(project_file)
    for e in extras:
        source = os.path.abspath(os.path.join(src_dir, e))
        assert os.path.exists(source)
        target = os.path.abspath(os.path.join(dest, e))
        if os.path.isdir(source):
            shutil.copytree(source, target)
        else:
            shutil.copyfile(source, target)
    return dest_fn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/test_copy_project_files.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add veneer/manage.py test/test_copy_project_files.py
git commit -m "Add copy_project_files helper for isolated project copies"
```

---

## Task 2: Refactor `cluster.copy_project` to delegate (DRY)

**Files:**
- Modify: `veneer/cluster.py:80-98` (the `copy_project` function) and its import line (`veneer/cluster.py:2`).
- Test: `test/test_copy_project_files.py` (add a parity test).

- [ ] **Step 1: Write the failing/parity test**

Append to `test/test_copy_project_files.py`:

```python
def test_cluster_copy_project_parity(tmp_path):
    """cluster.copy_project still produces the expected on-disk layout after
    delegating to copy_project_files."""
    import shutil
    from veneer.cluster import copy_project

    src = tmp_path / 'src'
    src.mkdir()
    proj = src / 'm.rsproj'
    proj.write_text('P')
    (src / 'data.csv').write_text('D')

    # copy_project is dask.delayed; .compute() runs it with the default scheduler.
    tmp = copy_project('veneer-parity-test-', str(proj), ['data.csv']).compute()
    try:
        assert os.path.exists(os.path.join(tmp, 'm.rsproj'))
        assert os.path.exists(os.path.join(tmp, 'data.csv'))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
```

- [ ] **Step 2: Run test to verify current state**

Run: `pytest test/test_copy_project_files.py::test_cluster_copy_project_parity -v`
Expected: PASS already (the current `copy_project` produces this layout). This test is a regression guard for the refactor — it must keep passing after Step 3.

- [ ] **Step 3: Refactor `copy_project` to delegate**

In `veneer/cluster.py`, update the import on line 2 from:

```python
from .manage import start, kill_all_now
```
to:
```python
from .manage import start, kill_all_now, copy_project_files
```

Replace the `copy_project` function (`veneer/cluster.py:80-98`) with:

```python
@dask.delayed
def copy_project(prefix,project_file,extras):
    assert os.path.exists(project_file)
    tmp = tempfile.mkdtemp(prefix=prefix)
    assert os.path.exists(tmp)
    copy_project_files(project_file, extras, tmp)
    return tmp
```

(Behaviour identical: same asserts, same temp dir, same copy semantics; only the copy body is now shared. The wrapper still returns `tmp` to preserve the cluster's contract.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/test_copy_project_files.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add veneer/cluster.py test/test_copy_project_files.py
git commit -m "Refactor cluster.copy_project to share copy_project_files helper"
```

---

## Task 3: `IsolatedSource` core lifecycle + context manager

**Files:**
- Modify: `veneer/manage.py`
- Test: `test/test_isolated_source.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_isolated_source.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_isolated_source.py -v`
Expected: FAIL with `AttributeError: module 'veneer.manage' has no attribute 'IsolatedSource'`.

- [ ] **Step 3: Write minimal implementation**

In `veneer/manage.py`, add (after `copy_project_files`, before `def start(` or after `start`):

```python
_CLEANUP_POLICIES = ('always', 'never', 'on_clean')


def _normalise_cleanup(cleanup):
    if cleanup is True:
        return 'always'
    if cleanup is False:
        return 'never'
    if cleanup in _CLEANUP_POLICIES:
        return cleanup
    raise ValueError(
        "cleanup must be one of 'always', 'never', 'on_clean' (or True/False); "
        "got %r" % (cleanup,)
    )


class IsolatedSource(object):
    """Run a single Veneer instance against an isolated, throwaway copy of a
    Source project.

    Copies project_file (and related_files) into a fresh temporary directory,
    starts one VeneerCmd instance against the copy, and exposes a ready Veneer
    client (``.v``) plus the sandbox path (``.directory``). Intended for
    workloads that must modify on-disk input files without disturbing the
    original model.

    Usable as a context manager (recommended) or inline with an explicit
    ``shutdown()``.

    Attributes set once construction succeeds:
      v            - ready Veneer client
      directory    - the temp copy directory (modify/read files here)
      project_file - the copied project-file path inside ``directory``
      port         - the actual bound port (may differ from requested)
      process      - the VeneerCmd process
      log_path     - captured stdout/stderr log path, or None
    """

    def __init__(self, project_file, related_files=None, *, veneer_exe=None,
                 port=9876, tempdir_prefix='source-isolated-',
                 remote=False, script=True, overwrite_plugins=None,
                 additional_plugins=None, custom_endpoints=None, model=None,
                 debug=False, cleanup='always',
                 trust_env=None, proxies=None):
        self._cleanup = _normalise_cleanup(cleanup)
        self._shutdown = False
        self.directory = None
        self.project_file = None
        self.process = None
        self.port = None
        self.log_path = None
        self.v = None

        self.directory = tempfile.mkdtemp(prefix=tempdir_prefix)
        try:
            self.project_file = copy_project_files(
                project_file, related_files or [], self.directory)
            # Pass the copy as project_fn (not projects=[...]) so that
            # overwrite_plugins actually takes effect; start() then defaults
            # projects to [project_fn] for the single instance.
            processes, ports, log_paths = start(
                project_fn=self.project_file,
                n_instances=1,
                ports=port,
                veneer_exe=veneer_exe,
                remote=remote,
                script=script,
                overwrite_plugins=overwrite_plugins,
                additional_plugins=additional_plugins or [],
                custom_endpoints=custom_endpoints or [],
                model=model,
                debug=debug,
                return_log_paths=True,
            )
            self.process = processes[0]
            self.port = ports[0]
            self.log_path = log_paths[0]
            self.v = Veneer(self.port, trust_env=trust_env, proxies=proxies)
        except BaseException:
            # Startup failure is "not clean": only the 'always' policy removes
            # the temp dir. The process (if any) is always killed.
            self._teardown(remove=(self._cleanup == 'always'))
            self._shutdown = True
            raise

    def _teardown(self, remove):
        if self.process is not None:
            try:
                kill_all_now(self.process)
            except Exception:
                logger.exception('Error killing Veneer process during teardown')
            self.process = None
        if remove and self.directory is not None and os.path.exists(self.directory):
            try:
                shutil.rmtree(self.directory)
            except Exception:
                logger.exception('Error removing temp directory %s during teardown',
                                 self.directory)

    def shutdown(self, clean=True):
        """Kill the Veneer instance and (per cleanup policy) remove the temp
        copy. Idempotent. ``clean`` indicates whether the surrounding work
        finished without error; it only affects the 'on_clean' policy. Inline
        callers using 'on_clean' must pass clean=False on their error path to
        keep the sandbox (the context-manager form does this automatically)."""
        if self._shutdown:
            return
        self._shutdown = True
        if self._cleanup == 'always':
            remove = True
        elif self._cleanup == 'never':
            remove = False
        else:  # 'on_clean'
            remove = clean
        self._teardown(remove=remove)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown(clean=(exc_type is None))
        return False


isolated_copy = IsolatedSource
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/test_isolated_source.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add veneer/manage.py test/test_isolated_source.py
git commit -m "Add IsolatedSource for single-instance isolated Source sandbox"
```

---

## Task 4: Cleanup policy behaviour + startup-failure handling

**Files:**
- Test: `test/test_isolated_source.py` (extend)
- (No new implementation — exercises the policy logic from Task 3.)

- [ ] **Step 1: Write the failing tests**

Append to `test/test_isolated_source.py`:

```python
import shutil


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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest test/test_isolated_source.py -v`
Expected: PASS (all tests). These exercise behaviour already implemented in Task 3; if any fail, fix the policy logic in `IsolatedSource` / `_normalise_cleanup` rather than the tests.

- [ ] **Step 3: Run the full test module + the copy helper module**

Run: `pytest test/test_isolated_source.py test/test_copy_project_files.py -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add test/test_isolated_source.py
git commit -m "Test IsolatedSource cleanup policy and startup-failure handling"
```

---

## Task 5: Regression check + docs touch

**Files:**
- (Verification only; optionally update `CLAUDE.md` architecture notes.)

- [ ] **Step 1: Run the existing related tests to confirm no regressions**

Run: `pytest test/test_start_return_shape.py test/test_worker_info.py test/test_copy_project_files.py test/test_isolated_source.py -v`
Expected: all PASS (the cluster import path and `start()` behaviour are unchanged).

- [ ] **Step 2 (optional): Note the new helper in `CLAUDE.md`**

Under "### Instance management (`veneer/manage.py`)", add a sentence: "Also provides `IsolatedSource` (alias `isolated_copy`) for running a single Veneer instance against an isolated temp copy of a project — the single-instance counterpart to the cluster's per-worker isolation." Only do this if the user wants the architecture doc updated.

- [ ] **Step 3: Commit (if CLAUDE.md changed)**

```bash
git add CLAUDE.md
git commit -m "Document IsolatedSource in CLAUDE.md"
```

---

## Notes for the implementer

- **Do not spawn real VeneerCmd in tests.** Follow `test/test_start_return_shape.py`: monkeypatch `manage.start`, `manage.kill_all_now`, and `manage.Veneer`. The copy helper runs for real against `tmp_path` files (cheap, no Source needed).
- `manage.py` already imports everything needed (`os`, `shutil`, `tempfile`, `logger`, `Veneer`). Do not add imports for the new code.
- `IsolatedSource` does its work in `__init__` (consistent with `VeneerCluster`); the context manager only guarantees teardown.
- The process is **always** killed on teardown/failure; only directory removal follows the cleanup policy.
- Keep `copy_project`'s return value as the temp dir — the cluster relies on it.
