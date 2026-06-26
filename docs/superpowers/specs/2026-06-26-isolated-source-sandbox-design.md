# Isolated single-instance Source sandbox — design

Date: 2026-06-26
Status: Approved (pending spec review)

## Problem

`veneer.cluster.VeneerCluster` can run many command-line copies of a Source
project in parallel, and when a workload needs to modify on-disk timeseries
input files it can be configured to create temporary, isolated copies of the
model and its related files/directories (one per worker). That isolation is
only available through the full dask-backed cluster machinery.

Many workloads need the *same isolation* but only a *single* instance: copy the
model and its inputs to a throwaway location, start one Veneer instance against
the copy, do work (including mutating the on-disk inputs), then tear everything
down. Today there is no first-class helper for this; users must hand-assemble
`copy_project`-style logic plus `start()` plus cleanup.

## Goals

- A first-class helper to run an isolated, single-instance Source sandbox.
- The temp copy **directory must be reachable** by the caller — the whole point
  of isolation is that the workload modifies/reads on-disk files there.
- Usable both as a context manager and inline (with an explicit shutdown).
- DRY with respect to the cluster's isolation management: share the temp-copy
  logic rather than duplicating it.

## Non-goals

- No change to the dask-backed cluster execution model. The cluster is *not*
  re-expressed in terms of the new single-instance primitive (explicitly chosen
  low-risk DRY depth: share the copy helper only).
- No multi-instance support in the new helper.
- No `copy=False` mode — isolation (always copy) is the helper's identity.

## Design

### 1. Shared copy helper (the DRY piece)

The pure file-copy logic currently lives *inside* the `@dask.delayed`
`copy_project()` function in `veneer/cluster.py`:

```python
@dask.delayed
def copy_project(prefix, project_file, extras):
    assert os.path.exists(project_file)
    tmp = tempfile.mkdtemp(prefix=prefix)
    assert os.path.exists(tmp)
    dest_fn = os.path.join(tmp, os.path.basename(project_file))
    shutil.copyfile(project_file, dest_fn)
    src_dir = os.path.dirname(project_file)
    for e in extras:
        source = os.path.abspath(os.path.join(src_dir, e))
        assert os.path.exists(source)
        dest = os.path.abspath(os.path.join(tmp, e))
        if os.path.isdir(source):
            shutil.copytree(source, dest)
        else:
            shutil.copyfile(source, dest)
    return tmp
```

Extract the pure body into a plain function in `veneer/manage.py`:

```python
def copy_project_files(project_file, extras, dest):
    """Copy project_file and each extra (file or directory) into dest.

    project_file: path to the .rsproj (or similar) to copy. Its basename is
                  written directly into dest.
    extras: iterable of paths, each interpreted relative to the project file's
            directory; copied into dest preserving that relative path.
    dest: an existing destination directory.

    Returns: full path to the copied project file inside dest.
    """
```

Behaviour is identical to the current inline logic: project basename copied to
`dest`; extras resolved relative to the project's source directory and copied
into `dest` (dirs via `copytree`, files via `copyfile`); existence asserted for
the project file and each extra.

`cluster.copy_project` is rewritten to delegate:

```python
@dask.delayed
def copy_project(prefix, project_file, extras):
    tmp = tempfile.mkdtemp(prefix=prefix)
    copy_project_files(project_file, extras, tmp)
    return tmp
```

Placement: `manage.py` is the instance-management module and is already
imported by `cluster.py` (not vice-versa), so there is no circular-import risk.

Difference to note: the original `copy_project` returns the temp directory,
while `copy_project_files` returns the copied project-file path (which the
single-instance helper needs to pass to `start()`). The cluster wrapper keeps
returning `tmp` to preserve its existing contract.

### 2. `IsolatedSource` class (in `veneer/manage.py`)

Mirrors `VeneerCluster`'s lifecycle convention: it does the work in `__init__`
and exposes `.shutdown()`, plus context-manager support.

```python
class IsolatedSource:
    def __init__(self, project_file, related_files=None, *, veneer_exe=None,
                 port=9876, tempdir_prefix='source-isolated-',
                 remote=False, script=True, overwrite_plugins=None,
                 additional_plugins=None, custom_endpoints=None, model=None,
                 debug=False, cleanup='always',
                 trust_env=None, proxies=None):
        ...
```

`__init__` sequence:

1. Validate `cleanup` (see policy below).
2. `tmp = tempfile.mkdtemp(prefix=tempdir_prefix)`.
3. `copied_project = copy_project_files(project_file, related_files or [], tmp)`.
4. `start(project_fn=copied_project, n_instances=1, ports=port,
   veneer_exe=veneer_exe, remote=remote, script=script,
   overwrite_plugins=overwrite_plugins, additional_plugins=additional_plugins
   or [], custom_endpoints=custom_endpoints or [], model=model, debug=debug,
   return_log_paths=True)` → reads back the **actual** bound port and log path.

   Note: the copied project is passed as `project_fn` (not `projects=[...]`).
   `start()` only applies `overwrite_plugins` when given `project_fn`
   (`manage.py:475-478`); with a single instance it then defaults `projects` to
   `[project_fn]`. Passing `project_fn` therefore makes `overwrite_plugins`
   actually take effect, whereas `projects=[...]` would silently no-op it (the
   same latent quirk that exists in the cluster path).
5. `self.v = Veneer(actual_port, trust_env=trust_env, proxies=proxies)`.
6. On any exception in steps 2–5: always kill any started process, then remove
   the temp dir iff policy is `'always'`; re-raise.

**State attributes:**

- `.v` — the `Veneer` client, ready to use.
- `.directory` — the temp copy directory (the path the caller needs).
- `.project_file` — the copied project-file path inside `.directory`.
- `.port` — the actual bound port (may differ from requested on collision).
- `.process` — the `Popen`/`Process` for the VeneerCmd instance.
- `.log_path` — captured stdout/stderr log path, or `None`.

**Methods:**

- `shutdown(clean=True)` — kill the process (idempotent), then remove the temp
  dir per policy + `clean`. Safe to call more than once.
- `__enter__` → returns `self`.
- `__exit__(exc_type, exc, tb)` → `self.shutdown(clean=(exc_type is None))`;
  returns `False` (does not suppress exceptions).

**Convenience alias:** `isolated_copy = IsolatedSource`, so the originally
imagined `with isolated_copy(model_file, related_files, ...) as s:` reads
literally.

### 3. Cleanup policy

A single three-way `cleanup` parameter governs **directory removal** only. The
process is *always* killed on shutdown/failure regardless of policy — a Veneer
process is never leaked.

| Value | Clean exit (success) | Body raised exception | `__init__` startup failure |
|---|---|---|---|
| `'always'` (default) | remove | remove | remove |
| `'never'` | keep | keep | keep |
| `'on_clean'` | remove | keep | keep |

- `__exit__` computes `clean = exc_type is None` and calls
  `shutdown(clean=clean)`.
- `shutdown(clean=True)` — manual/inline calls treat the exit as clean by
  default.
- Removal decision: `'always'` → remove; `'never'` → keep; `'on_clean'` →
  remove iff `clean`.
- `__init__` startup failure is "not clean", so only `'always'` removes the
  temp dir there.
- `True`/`False` are accepted as friendly aliases for `'always'`/`'never'`.
- Unknown values raise `ValueError` in `__init__`.

### 4. Usage

```python
from veneer.manage import IsolatedSource  # or: isolated_copy

with IsolatedSource(model_file,
                    related_files=['inputs', 'climate.csv'],
                    veneer_exe=exe) as s:
    # mutate on-disk inputs inside the sandbox, then run
    edit_input_file(os.path.join(s.directory, 'climate.csv'))
    s.v.run_model()
    df = s.v.retrieve_multiple_time_series(...)
# instance killed; temp copy removed (policy 'always')

# inline / manual lifecycle:
s = IsolatedSource(model_file, veneer_exe=exe, cleanup='on_clean')
try:
    s.v.run_model()
    s.shutdown()              # clean=True -> removed under 'on_clean'
except Exception:
    s.shutdown(clean=False)   # keep sandbox for post-mortem under 'on_clean'
    raise
```

Note: `shutdown()` defaults to `clean=True`. Inline callers using `'on_clean'`
must pass `clean=False` on their error path to keep the sandbox; the
context-manager form does this automatically via `exc_type`.

## Error handling

- Copy failures (missing project file or extra) surface via the existing
  `assert`/exception behaviour of the shared helper; the temp dir is removed iff
  policy is `'always'`.
- Start failures propagate `start()`'s descriptive exception (which already
  includes the captured stdout/stderr tail naming the underlying cause); temp
  dir removed iff policy is `'always'`.
- `shutdown()` is tolerant and idempotent: each step is guarded, calling it
  twice (or `__exit__` after a manual `shutdown()`) does not error.
- Port collisions need no special handling: a requested port is passed to
  `start()` and the actual bound port is read back from its return value.

## Testing

Tests follow the integration framework in `test/` (see `test/TESTING_GUIDE.md`).

1. **Pure unit test of `copy_project_files`** — no Veneer required. Given a
   fixture project file plus a related file and a related directory, copy into a
   `tmp_path` dest; assert the project basename, the file, and the directory
   tree are present, and that the returned path is the copied project file.
   Assert that a missing extra raises.
2. **Cluster parity** — a focused test (or reuse of existing cluster copy
   coverage) confirming `cluster.copy_project` still produces the same on-disk
   layout after delegating to `copy_project_files`.
3. **`IsolatedSource` lifecycle (integration)** — against a fixture project:
   - `.directory` exists and contains the copied project + related files.
   - `.v` answers a basic request; `.port` is populated.
   - mutate an input file inside `.directory`, run the model.
   - after `with`-exit (or `.shutdown()`), the process is gone and (policy
     `'always'`) the temp dir is removed.
4. **Cleanup policy** — `'never'` keeps the dir after shutdown; `'on_clean'`
   keeps it when the body raises and removes it on clean exit. (Process always
   killed.)

## Scope choices (YAGNI)

- Copy is always on — no `copy=False` toggle.
- No dask — single in-process instance.
- Cluster internals (dask affinity, per-port logs, multi-instance) are
  untouched beyond extracting the shared copy helper.
