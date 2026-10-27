# Setting up Veneer and veneer-py

This guide covers getting from a fresh machine to scripting eWater Source from Python:

1. [Install Veneer (the Source plugin)](#1-install-veneer-the-source-plugin)
2. [Install veneer-py](#2-install-veneer-py)
3. [Connect to Source running in the GUI](#3-connect-to-source-running-in-the-gui)
4. [Create a Veneer command line](#4-create-a-veneer-command-line)
5. [Start command-line instances from Python with `veneer.manage.start`](#5-start-command-line-instances-from-python)
6. [Troubleshooting](#6-troubleshooting)

## What the pieces are

| Piece | What it is |
|---|---|
| **eWater Source** | The modelling application, installed under something like `C:\Program Files\eWater\Source 5.42.0.12345`. |
| **Veneer** | A Source plugin (`FlowMatters.Source.Veneer.dll`) that runs an HTTP server inside Source, exposing the model through a REST/JSON API. |
| **Veneer command line** | `FlowMatters.Source.VeneerCmd.exe`, shipped with Veneer. It runs the Source engine plus the Veneer server **without** the Source user interface. You can run several at once, which makes it the tool for batch and parallel work. |
| **veneer-py** | This Python package. It talks HTTP to either the GUI or the command line, and can also launch and manage command-line instances. |

Everything veneer-py does works the same way against the GUI and against the command line. The only difference is how the server gets started.

---

## 1. Install Veneer (the Source plugin)

Recent Source releases bundle Veneer as a community plugin, under `Plugins\CommunityPlugins\` in the Source install directory. If that version is recent enough for you, you can skip the download.

Otherwise:

1. Download the [Veneer release](https://github.com/flowmatters/Veneer/releases) that matches your Source version, and unzip it somewhere convenient, for example `C:\Veneer\Source 5.42`. **Veneer builds are version-specific**, so use the build for your exact Source version.
2. In Source, open **Tools → Plugin Manager**, add `FlowMatters.Source.Veneer.dll` from the unzipped folder, then restart Source.

Keep the path to the unzipped Veneer folder. Section 4 needs it.

> **If your Source was installed by MSI and already bundles Veneer, step 2 won't work.** The Source GUI always loads the bundled copy from its own `Plugins\CommunityPlugins\`, even if you select a different `FlowMatters.Source.Veneer.dll` in the Plugin Manager. To use a newer release or your own build with that install, close Source and copy **all** the new Veneer files over the existing copy. Depending on the Source version, that copy is directly in `CommunityPlugins`, in a subfolder such as `CommunityPlugins\Veneer`, or in both. Copying into `C:\Program Files` needs administrator rights. Back up the old files first if you might want them back. The [Veneer README](https://github.com/flowmatters/Veneer#updating-a-bundled-veneer) has the steps.
>
> This affects the GUI only. A command line built in section 4 uses whichever Veneer folder you give it. If you use `command_line_for`, which finds Veneer through the running GUI, replace the bundled copy first so the GUI and the command line run the same Veneer.

## 2. Install veneer-py

You need Python 3. Any distribution works: Anaconda, Miniforge, python.org Python in a virtual environment, and so on. veneer-py depends on `requests`, `pandas`, `numpy`, `geopandas` and `psutil`. On Windows, `geopandas` is usually easiest to get through conda (`conda install -c conda-forge geopandas`).

Install the latest version from GitHub:

```
pip install https://github.com/flowmatters/veneer-py/archive/master.zip
```

To upgrade later, run the same command with `--upgrade --force-reinstall`.

If you plan to modify veneer-py itself, clone the repository and do an editable install instead:

```
git clone https://github.com/flowmatters/veneer-py.git
cd veneer-py
pip install -e .
```

## 3. Connect to Source running in the GUI

This is the simplest setup and a good way to check that everything works.

1. Start Source and load a project (`.rsproj`).
2. Open **Tools → Web Server Monitoring** to start the Veneer server. The window shows the port, which defaults to **9876**. If that port is taken, for example by another Source, Veneer tries 9877, 9878 and so on.
   - Tick **Allow Scripts** if you want to use the server-side (IronPython) features of veneer-py (`v.model.*`). It is off by default.
3. Check the connection in a browser at <http://localhost:9876/network>. It should return a large block of GeoJSON.
4. Connect from Python:

```python
import veneer
v = veneer.Veneer()            # default port 9876
# v = veneer.Veneer(port=9877) # if Veneer reported a different port

network = v.network()
nodes = network['features'].find_by_feature_type('node')
print(nodes._unique_values('name'))
```

> **Remote connections.** By default Veneer only accepts connections from the same machine. Connecting from another machine needs **Allow Remote Connections** (or `-r` on the command line), and the port may first need registering with Windows, as administrator:
> `netsh http add urlacl url=http://+:9876/ user=%USERNAME%`
> The [Veneer README](https://github.com/flowmatters/Veneer#allow-remote-connections) has the details. Firewall rules may still block the connection.

## 4. Create a Veneer command line

`FlowMatters.Source.VeneerCmd.exe` is in the Veneer folder, but it needs Source's own DLLs to run. The most reliable approach, which works for every Source version, is to **copy Veneer and Source into one directory** and run the exe from there. veneer-py has two helpers that do this for you.

> The copy is a full Source distribution, typically hundreds of MB. Put it somewhere stable, such as `C:\veneer_cmd\Source 5.42`, and reuse it. Don't rebuild it every session.

### Option A: `create_command_line` (from an installed Source)

```python
from veneer.manage import create_command_line

veneer_cmd = create_command_line(
    veneer_path=r'C:\Veneer\Source 5.42',      # the unzipped Veneer release (or the bundled plugin folder)
    source_version='5.42',                     # matched as a prefix against "Source <version>" folders...
    source_path=r'C:\Program Files\eWater',    # ...under this directory
    dest=r'C:\veneer_cmd\Source 5.42',         # where to build the combined copy
    force=False,                               # reuse an existing build at dest
)
veneer_cmd   # full path to FlowMatters.Source.VeneerCmd.exe
```

If your Source is not under `C:\Program Files\eWater\Source <version>`, pass `source_version=None` and set `source_path` to the Source directory itself, i.e. the folder that contains `RiverSystem.Forms.exe`:

```python
veneer_cmd = create_command_line(r'C:\Veneer\Source 5.42',
                                 source_version=None,
                                 source_path=r'D:\Apps\Source 5.42.0.12345',
                                 dest=r'C:\veneer_cmd\Source 5.42')
```

Veneer is copied first and Source second, so Source's files win when both ship the same DLL. Source 6 (.NET 8) builds need this ordering.

### Option B: `command_line_for` (from a running GUI instance)

If Source is already open with Veneer running (section 3), veneer-py can work out both directories from the live instance, so you don't need to know any paths or version strings:

```python
import veneer
from veneer.manage import command_line_for

v = veneer.Veneer()   # connected to the GUI
veneer_cmd = command_line_for(v, cache_dir=r'C:\veneer_cmd\cache')
```

The result is cached in `cache_dir`. Later calls reuse it unless the Source version, Source location or Veneer DLL has changed. Pass `force=True` to rebuild anyway.

### Running it by hand (optional)

You can also run the command line directly from a Windows command prompt:

```
"C:\veneer_cmd\Source 5.42\FlowMatters.Source.VeneerCmd.exe" -p 9876 -s "C:\models\MyModel.rsproj"
```

Wait for the server to report it is listening (e.g. `Server started. Ctrl-C to exit...` or `Now listening on: http://localhost:9876`). Then connect with `veneer.Veneer(port=9876)` as in section 3. The main options are:

| Flag | Meaning |
|---|---|
| `-p <port>` | Port to listen on (default 9876) |
| `-s` | Allow IronPython scripts (needed for `v.model.*`) |
| `-r` | Allow remote connections |
| `-m <name or index>` | Model (scenario) to load. Default is the first. Index is 1-based. |
| `-a` | List available models (scenarios) in the project, then exit |
| `-l <dll,dll>` | Load extra plugins, in addition to the registered ones |
| `-x` | Skip loading the already-registered plugins |
| `-c <file,file>` | Enable custom endpoint files |
| `-d <dir>` | Source directory to load DLLs from (recent Veneer builds only; see below) |
| `-v <level>` | Log level: `Debug`, `Info`, `Warning`, `Error` |

Run it with `--help` for the full list.

> **Recent Veneer builds (Source 6 era)** can find Source without a combined copy. They accept `-d <Source directory>`, and the newest builds also search the directories above the exe for `RiverSystem.dll`, so the exe can run straight from `Plugins\CommunityPlugins\Veneer_for_...`. That is convenient for running it by hand. `veneer.manage.start` does not pass `-d`, though, and older Veneer builds have neither feature. The combined copy from Option A or B is therefore the setup to use with veneer-py.

## 5. Start command-line instances from Python

`veneer.manage.start` launches one or more copies of the command line, waits until each is ready, and returns their processes and **actual** ports:

```python
import veneer
from veneer.manage import start, kill_all_now

project = r'C:\models\MyModel.rsproj'

processes, ports = start(project,
                         n_instances=4,        # e.g. one per CPU core you want to use
                         ports=9950,           # first port; the rest count up from here
                         veneer_exe=veneer_cmd,
                         return_log_paths=False)

clients = [veneer.Veneer(port=p) for p in ports]
clients[0].run_model()
```

Always use the `ports` that `start` returns. An instance may bind a different port than you asked for if the requested one was in use.

The instances are killed automatically when your Python process exits. To stop them sooner:

```python
kill_all_now(processes)
```

### Useful `start` options

| Argument | Purpose |
|---|---|
| `project_fn` | The `.rsproj` to load into every instance. |
| `n_instances` | Number of copies to start. |
| `ports` | First port (int) or an explicit list, one per instance. |
| `veneer_exe` | Path from `create_command_line` / `command_line_for`. Supply it explicitly: the fallback default is a developer path that won't exist on your machine. |
| `projects` | A list of different project files, one per instance. It sets `n_instances` for you. |
| `model` | Scenario to load, by name or 1-based index. Pass a list to load a different scenario in each instance. |
| `script` | Allow IronPython scripts (default `True`, i.e. `-s`). |
| `remote` | Allow remote connections (default `False`). |
| `additional_plugins` | List of extra plugin DLLs to load. |
| `overwrite_plugins` | If truthy, copy a `Plugins.xml` found beside the project (or in a parent folder) into Source's per-user plugin config first. |
| `debug` | Echo the command lines and all startup output. Use this when an instance won't start. |
| `quiet` | Suppress all server output. |
| `capture_output_dir` | Tee each instance's output to `<dir>/veneer_port_<port>.log`. |
| `detached` | Host the instances from a separate Python process in a new console window (Windows only). Use it if you launch from a VS Code Jupyter notebook; see [Troubleshooting](#6-troubleshooting). |
| `return_log_paths` | Pass `True` to get `(processes, ports, log_paths)`, or `False` for the plain `(processes, ports)`. Set it explicitly either way to silence a `DeprecationWarning`; the default will change to `True`. |
| `readiness` | `'http'` (default): poll `GET /ready`, falling back to `GET /` on servers built before it existed. `'stdout'`: the legacy log-parsing-only behaviour. |
| `startup_timeout` | Seconds to wait for every instance to become ready before raising `VeneerStartupTimeout` and killing them all (default `600.0`). `None` waits forever. |
| `exit_on_stdin_eof` | Default `True`: instances die with this process (a held stdin pipe closes when it exits), via `VENEER_EXIT_ON_STDIN_EOF`. Older builds ignore the unknown variable. |
| `gc` | See [Memory: GC mode and results retention](#memory-gc-mode-and-results-retention) below. |

Run `help(veneer.manage.start)` for everything else (`return_io`, `progress_callback`, `custom_endpoints`, ...).

### Memory: GC mode and results retention

Two separate things grow an instance's memory footprint, and only one of them is `start`'s to control:

**GC mode** (`gc` parameter, default `None`). Server GC gives each `VeneerCmd` instance its own large set of heaps/threads (one heap per logical CPU by default): fast, but memory-hungry (measured: ~8.2 GB RSS for one instance) and the wrong call once several instances have to share a host's RAM *or* its CPUs. Workstation GC is far leaner but slower. Today's data: workstation is the leanest choice overall; server is faster for a single instance when RAM is plentiful. A 112-core Fedora host running 2+ instances under server GC — well over 112 heaps competing for 112 cores — **livelocked**: Server GC's own threads burned all their CPU contending with each other rather than running the model. Workstation GC had no hangs at the same N, about 2x the throughput and about half the memory. The default (`gc=None`) now requires BOTH of two checks to pick server GC, rather than RAM alone: physical RAM / (`n_instances` + 1) at least 16 GB, **and** `n_instances * heaps` at or below the host's logical CPU count (`heaps` = one per logical CPU) — `DOTNET_gcServer=1` (server) only when both hold, `DOTNET_gcServer=0` (workstation) otherwise. In practice the CPU check means `n_instances >= 2` already fails on its own, so the default only ever picks server GC for a single instance (and only when RAM is plentiful too). `DOTNET_GCDynamicAdaptationMode=1` ("DATAS") is server GC's own lower-memory mode (~1.6-2.0 GB, some speed cost) — it needs server GC underneath it to do anything, so pass `gc='dynamic'` to get both set unconditionally (bypassing both checks — an explicit choice, not the default policy). `gc=False` passes no `DOTNET_GC*` variables at all (the build's own default applies). `gc='server'`/`'workstation'` set `DOTNET_gcServer` explicitly without DATAS. An int sets `DOTNET_gcServer=1` plus an explicit `DOTNET_GCHeapCount` — the escape hatch for capped server GC with several instances, also bypassing the CPU check. A dict passes raw `DOTNET_*` variables. See `veneer.manage._gc_env_vars` for the exact mapping. Every `start()` call logs its GC decision as one line via the module logger (`gc=None, default): 4 instances × 112 heaps > 112 CPUs, 31.7 GB RAM → 6.3 GB/instance < 16 GB → DOTNET_gcServer=0`, for example) — see `veneer.manage._format_gc_decision_line`.

**Results retention** is unrelated to GC mode: Source keeps every run's results in memory until told to drop them, so a long-lived instance that keeps running models grows regardless of GC mode (one instance measured 2 GB → 10 GB over 22 runs). Call `v.drop_all_runs()` (or `v.drop_run(run)` for a single one) between runs on any instance you intend to keep running rather than recycle.

### Running in parallel

Each client is independent. Use `run_async=True` to start several runs at once, then wait on each one:

```python
responses = [v.run_model(run_async=True) for v in clients]
codes = [r.getresponse().getcode() for r in responses]   # blocks until each run finishes
```

To send the same call to every instance, use `BulkVeneer`:

```python
from veneer.manage import BulkVeneer
bulk = BulkVeneer(ports)
bulk.configure_recording(enable=[{'RecordingVariable': 'Downstream Flow Volume'}])
```

For larger jobs, `veneer.cluster.VeneerCluster` manages a pool of instances behind dask workers.

### Working on a throwaway copy of a project

If your script modifies input files on disk, run against an isolated copy so the original model is never touched:

```python
from veneer.manage import IsolatedSource

with IsolatedSource(r'C:\models\MyModel.rsproj',
                    related_files=['inputs'],   # extra files/folders, RELATIVE to the project's folder
                    veneer_exe=veneer_cmd) as iso:
    iso.v.run_model()          # iso.v is a ready Veneer client
    print(iso.directory)       # the temp copy; it is deleted on exit by default
```

## 6. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `start()` raises "failed to start", with output mentioning an assembly or type it could not load | The command line can't find Source's DLLs. Rebuild it with `create_command_line` / `command_line_for`, and check that the Veneer build matches the Source version. |
| "Cannot find project" | Wrong `.rsproj` path. `start` makes the path absolute relative to the current working directory. |
| Instances start but hang or don't respond, when launched from **VS Code Jupyter** | A known problem with child processes launched under VS Code. Use `start(..., detached=True)`, or run from Jupyter Lab or a plain `python` script. veneer-py logs a warning when it detects VS Code. |
| `v.model.*` calls fail with a scripting error | Scripts aren't enabled. Tick **Allow Scripts** in the GUI, or use `-s` / `script=True` (the default for `start`). |
| Project loads but plugin-provided models are missing | The plugins aren't registered for this Source version on this user account. Register them in the GUI's Plugin Manager, pass them via `additional_plugins`, or use `overwrite_plugins=True` with a `Plugins.xml` next to the project. |
| Leftover `FlowMatters.Source.VeneerCmd.exe` processes after a crash | `veneer.manage.find_veneers()` lists them. `veneer.manage.kill_every_running_instance_of_veneer_cmd_line()` kills **all** of them, including ones from other sessions. |
| Can't connect from another machine | See the remote connections note in section 3: `-r` / `remote=True`, URL ACL registration and the firewall. |

## Further reading

- [README](README.md): overview and a quick example
- [doc/training](doc/training): tutorial notebooks. Session 7 covers parallel processing with the command line.
- [doc/examples](doc/examples): short examples by topic
- [Veneer repository](https://github.com/flowmatters/Veneer): plugin details, GUI options, releases
- `help(veneer.Veneer)` and tab completion in Jupyter, for exploring the client API
