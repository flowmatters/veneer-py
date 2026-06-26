# Running a Veneer cluster

`veneer.cluster.VeneerCluster` manages a pool of Source/Veneer command-line
instances and runs work across them in parallel, backed by a local
[Dask](https://www.dask.org/) cluster (one Dask worker per Source instance).
Use it to spread many simulations — parameter sweeps, scenario ensembles,
Monte-Carlo runs — over several CPU cores.

> **Not the same as the older batch runner.** The `BulkVeneer` +
> `BatchRunner` approach (see `VeneerBatchRuns.ipynb`) connects to instances
> *you* started by hand and broadcasts to them. `VeneerCluster` instead
> **starts and manages** the instances for you and adds Dask-based job
> scheduling and optional per-worker model isolation. For a single isolated
> instance, see [`IsolatedSource`](IsolatedSandbox.md).

## Starting a cluster

```python
from veneer.cluster import VeneerCluster

cluster = VeneerCluster(
    project_file='C:/models/catchment.rsproj',
    veneer_exe='C:/Veneer/FlowMatters.Source.VeneerCmd.exe',
    n_workers=4,           # default: number of CPUs (capped at 64)
)
print(cluster.veneer_ports)   # e.g. [9876, 9877, 9878, 9879]
```

The constructor spins up the Dask cluster, starts `n_workers` Veneer instances,
and maps each Dask worker to a Source instance. If startup fails it tears down
any partially-started resources (pass `cleanup_on_failure=False` to leave them
running for investigation).

Always shut the cluster down when finished — this kills the Veneer processes,
removes any temporary copies, and closes the Dask cluster:

```python
cluster.shutdown()
```

## Talking to the instances

**Broadcast to every instance** with `cluster.v`. It mirrors the `Veneer`
client API and returns one result per instance:

```python
infos = cluster.v.scenario_info()      # list, one entry per worker
cluster.v.configure_recording(enable=[{'RecordingVariable': 'Downstream Flow Volume'}])
```

**Address a single instance** by index via `cluster.workers` (a list of plain
`Veneer` clients):

```python
v0 = cluster.workers[0]
network = v0.network()
```

## Isolated per-worker copies

By default every worker loads the *same* project file. When a workload needs to
**modify on-disk inputs** (e.g. rewrite time-series files), give each worker its
own copy so they don't collide:

```python
cluster = VeneerCluster(
    project_file='C:/models/catchment.rsproj',
    veneer_exe=veneer_exe,
    n_workers=4,
    copy=True,                          # one temp copy of the project per worker
    copy_extras=['inputs', 'climate.csv'],  # also copy these (relative to the project dir)
)
```

`copy_extras` entries are resolved relative to the project file's directory and
copied into each worker's temp directory (files and directories alike). Setting
`copy_extras` implies `copy=True`. The temp directories are removed on
`shutdown()`. This is the same isolation mechanism that
[`IsolatedSource`](IsolatedSandbox.md) provides for a single instance — both
share `veneer.manage.copy_project_files` under the hood.

When projects are copied, each worker also knows its own directory, which your
jobs can receive (see below).

## Running custom parallel jobs

For arbitrary work, write a function that accepts a `v` keyword (the worker's
`Veneer` client), wrap it with `cluster.wrap`, build one job per unit of work,
and run them with `cluster.run_jobs`:

```python
import pandas as pd

def run_one(params, v):
    # apply parameters, run, return a scalar/series/frame
    for name, value in params.items():
        v.model.catchment.runoff.set_param_values(name, value, fus=['Grazing'])
    v.run_model()
    return v.retrieve_multiple_time_series(
        criteria={'RecordingVariable': 'Downstream Flow Volume'}).sum()[0]

wrapped = cluster.wrap(run_one)
jobs = [wrapped(p) for p in list_of_param_dicts]
results = cluster.run_jobs(jobs)        # blocks; returns [((port, directory), value), ...]
```

If a worker process dies mid-run, `run_jobs` raises `WorkerDied` carrying the
structured failure details (port, exit code, log tail). Pass
`partial_results=True` to instead get back the raw list of per-job result dicts
(including any failures) without raising.

If your cluster was created with `copy=True`, your job function may also accept a
`directory` keyword — the path of that worker's project copy — so it can read or
write files in its own isolated sandbox:

```python
def run_one(params, v, directory):
    rewrite_inputs(directory)           # mutate this worker's copy
    v.run_model()
    return v.retrieve_multiple_time_series(...)
```

## Reconnecting to a running cluster

A cluster can be serialised so another process (or a later session) can attach
to the already-running instances instead of starting new ones:

```python
cluster.to_json('cluster.json')         # save config

# elsewhere / later:
from veneer.cluster import VeneerCluster
cluster = VeneerCluster(existing_cluster='cluster.json')
```

## Command line

A cluster can also be started from a terminal, which keeps it running until you
press Ctrl+C:

```bash
python -m veneer.cluster C:/models/catchment.rsproj \
    -e C:/Veneer/FlowMatters.Source.VeneerCmd.exe \
    -n 4 --copy --copy-extras inputs climate.csv \
    --save cluster.json
```

Useful flags: `-n/--n-workers`, `-p/--port` (starting port), `--copy`,
`--copy-extras`, `--additional-plugins`, `--custom-endpoints`, `--remote`,
`--existing {raise,remove,ignore}` (what to do about leftover temp directories),
and `--save` (write a config JSON for reconnection). Run with `-h` for the full
list.

## Requirements & gotchas

- You need a working Source installation and a valid (or discoverable)
  `veneer_exe` — the cluster launches real `VeneerCmd` processes.
- Each worker is a full Source instance, so memory and startup time scale with
  `n_workers`. The default worker count is `min(cpu_count, 64)`.
- Per-worker copies (`copy=True`) cost disk and copy time; copy only the inputs
  your jobs touch via `copy_extras`.
- Always call `shutdown()` (or use the command-line form and Ctrl+C) so
  instances and temp directories are cleaned up.
- For a *single* isolated instance rather than a pool, prefer
  [`IsolatedSource`](IsolatedSandbox.md).
```

