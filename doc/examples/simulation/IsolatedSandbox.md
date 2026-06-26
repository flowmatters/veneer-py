# Isolated sandbox copies of a Source model

`veneer.manage.IsolatedSource` runs a single Veneer command-line instance
against a **throwaway copy** of a Source project. It is the single-instance
counterpart to the isolation that [`VeneerCluster`](VeneerBatchRuns.ipynb)
provides per worker.

Use it when a workload needs to **modify on-disk inputs** (e.g. rewrite
time-series CSVs, swap an input set, edit a data file referenced by the model)
and you don't want those changes to touch your original project. The model and
the files you nominate are copied into a fresh temporary directory, an instance
is started against the copy, and everything is torn down when you're done.

```python
import os
from veneer.manage import IsolatedSource   # alias: isolated_copy

with IsolatedSource('C:/models/catchment.rsproj',
                    related_files=['inputs', 'climate.csv'],
                    veneer_exe='C:/Veneer/FlowMatters.Source.VeneerCmd.exe') as s:
    # s.directory is the temp copy — edit inputs there, not in the original
    rewrite_csv(os.path.join(s.directory, 'climate.csv'))

    s.v.run_model()                                  # s.v is a ready Veneer client
    results = s.v.retrieve_multiple_time_series(...)

# instance killed, temp copy removed (default cleanup policy)
```

## What you get

While the `with` block is active (or after a successful constructor call), the
object exposes:

| Attribute | Description |
|-----------|-------------|
| `s.v` | A ready [`Veneer`](../../source/veneer.rst) client connected to the instance. |
| `s.directory` | The temporary copy directory. **Edit/read your input and output files here.** |
| `s.project_file` | The copied project file inside `s.directory`. |
| `s.port` | The actual bound port (may differ from the requested one if it was taken). |
| `s.process` | The `VeneerCmd` process. |
| `s.log_path` | Path to the captured stdout/stderr log (under `s.directory/veneer_logs/`), or `None` if `capture_output=False`. |

## What gets copied

- The project file itself (its basename is written directly into the temp dir).
- Every entry in `related_files`, resolved **relative to the project file's
  directory** and copied into the temp dir preserving that relative path. Both
  files and directories are supported (directories are copied recursively).

```python
# project at C:/models/catchment.rsproj
related_files=['inputs',            # copies C:/models/inputs/  ->  <tmp>/inputs/
               'data/climate.csv']  # copies C:/models/data/climate.csv -> <tmp>/data/climate.csv
```

Only nominate the files the workload actually reads or writes — there's no need
to copy the whole project tree.

## Cleanup policy

The `cleanup` argument controls whether the temporary directory is removed.
The **instance process is always killed** on teardown regardless of policy;
only directory removal is governed by the policy.

| `cleanup` | Clean exit (success) | Body raised an exception | Startup failure |
|-----------|----------------------|--------------------------|-----------------|
| `'always'` *(default)* | remove | remove | remove |
| `'never'` | keep | keep | keep |
| `'on_clean'` | remove | **keep** | **keep** |

`'on_clean'` is handy for debugging: the sandbox (and its captured logs)
survives exactly when something went wrong.

```python
# Keep the sandbox for inspection if the run fails
with IsolatedSource(project, veneer_exe=exe, cleanup='on_clean') as s:
    s.v.run_model()
    check_results(s.v)        # if this raises, s.directory is left on disk
```

`True` / `False` are accepted as aliases for `'always'` / `'never'`.

### Inline use (no `with` block)

`IsolatedSource` does its work in the constructor, so you can also manage the
lifecycle yourself. When using `'on_clean'` inline, you must tell `shutdown()`
whether the work finished cleanly — the context-manager form does this
automatically from the exception state.

```python
s = IsolatedSource(project, veneer_exe=exe, cleanup='on_clean')
try:
    s.v.run_model()
    s.shutdown()                # clean=True (default) -> sandbox removed
except Exception:
    s.shutdown(clean=False)     # keep the sandbox for post-mortem
    raise
```

`shutdown()` is idempotent — calling it again (or letting `__exit__` run after a
manual `shutdown()`) is a no-op.

## Captured logs

By default (`capture_output=True`) the instance's combined stdout/stderr are
teed to a per-port log file under `<directory>/veneer_logs/`, and `s.log_path`
points at it. Combined with `cleanup='never'`/`'on_clean'`, this means a failed
run leaves both the model copy and the Veneer log on disk for inspection. Pass
`capture_output=False` to disable capture (`s.log_path` is then `None`).

## Selected parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `project_file` | *(required)* | Path to the `.rsproj` (or similar) to copy. |
| `related_files` | `None` | Files/directories to copy alongside the project (see above). |
| `veneer_exe` | `None` | Path to `FlowMatters.Source.VeneerCmd.exe`. If omitted, veneer-py attempts to locate it (see `find_veneer_cmd_line_exe`). |
| `port` | `9876` | Requested port; the actual bound port is read back into `s.port`. |
| `cleanup` | `'always'` | See the cleanup policy table. |
| `capture_output` | `True` | Tee stdout/stderr into the sandbox. |
| `overwrite_plugins` | `None` | When truthy, apply a `Plugins.xml` (resolved from the project directory) before starting. |
| `additional_plugins`, `custom_endpoints`, `model`, `remote`, `script`, `debug` | — | Forwarded to `veneer.manage.start`; see its docstring. |
| `trust_env`, `proxies` | `None` | Forwarded to the `Veneer` client; see `Veneer.__init__`. |

## Relationship to `VeneerCluster`

`IsolatedSource` and the cluster share the same underlying copy logic
(`veneer.manage.copy_project_files`). Reach for the cluster when you need many
instances running in parallel; reach for `IsolatedSource` when a single
isolated instance is enough.

## Requirements & gotchas

- You need a working Source installation and a valid (or discoverable)
  `veneer_exe` — `IsolatedSource` launches a real `VeneerCmd` process.
- Copying large input directories has a cost (disk + time); copy only what the
  workload touches.
- Read the outputs/inputs you care about **before** the `with` block exits
  under the default `'always'` policy — the temp copy is removed on exit. Use
  `'never'`/`'on_clean'` if you need the sandbox afterwards.
