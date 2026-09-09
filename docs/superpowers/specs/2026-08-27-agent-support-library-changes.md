# Library support for automated and sandboxed workflows — design

Date: 2026-08-27
Status: Draft

## Problem

Scripted and automated veneer-py workflows increasingly need to do three things that
the library makes harder than it should:

1. **Experiment against a throwaway copy of a live model** without disturbing the
   modeller's open project. `IsolatedSource` provides the primitive, but obtaining a
   runnable command line for it, and managing more than one copy at a time, is
   hand-rolled by every caller.
2. **Seed such a copy from the running instance**, so it reflects what the modeller
   is actually looking at rather than the last-saved file. `v.model.save()` does this
   — but it currently leaves the project in a damaged state.
3. **Tell which build of veneer-py is in use**, which is currently impossible.

Each item below stands on its own. One is a straightforward bug fix that affects
anyone calling `save()` today.

## Non-goals

- No change to `VeneerCluster` or the dask execution model.
- No change to `IsolatedSource`'s own semantics; the registry wraps it.
- No new dependencies.

---

## 1. `v.model.save()` leaves `OutputFile` blank — bug fix

**Current behaviour** (`veneer/server_side.py:807`): `save()` builds an IronPython
script that sets `ph.ProjectMetaStructure.OutputFile = ''` and reassigns
`ph.ProjectMetaStructure.Project`, then calls `ph.SaveProject()`. Its `finally`
restores only `ph.CallBackHandler`.

So after any `save(fn=...)`, the project's `OutputFile` is left empty. A subsequent
`save(fn=None)` — documented as "save using current filename" — resolves its filename
from `ph.ProjectMetaStructure.OutputFile` and therefore saves to `''`.

**Change:** add `preserve_current=True` (default), capturing `OutputFile` and
`Project` before the save and restoring both in the `finally` alongside
`CallBackHandler`.

**Also to establish:** whether `SaveProject()` with a callback `OutputFileName`
repoints the application's notion of its current file. If it does, a save-to-elsewhere
would leave a GUI user's later "Save" writing to the new location — worth confirming
and, if so, restoring under the same flag.

**Verification requires a live instance.** Probe on a throwaway project, never a
production model.

---

## 2. `command_line_for(v, cache_dir=...)` — derive the command line from a
   running instance

**Problem with the current path:** `find_veneer_cmd_line_exe()`
(`veneer/manage.py:111`) depends on a `source_version.txt` beside the project and
otherwise falls back to a module-level constant. Its `MANY_VENEERS` branch calls
`len()` on a `Path.glob()` generator, which raises `TypeError` — that branch cannot
succeed.

**Better source of truth: the running instance.** `v.status()`
(`veneer/general.py:274`) already returns the `/` payload, which carries `HostExe`,
`PluginsLoaded` and `SourceVersion`. So:

- `source_path = dirname(HostExe)` — the Source build directory.
- `veneer_path = dirname(<the Veneer plugin dll in PluginsLoaded>)`.
- `source_version=None`, meaning `source_path` *is* the build directory.

This cannot disagree with the Source actually running, and needs no version string.

**The command line must still be built.** `FlowMatters.Source.VeneerCmd.exe` ships in
the plugin directory, but without the Source assemblies beside it (observed: 18 files
there, no `RiverSystem.*` or `TIME*`), so it cannot run in place.
`create_command_line()` performs the merge — a heavyweight copy of the whole Source
distribution (observed: 1144 files).

**Therefore cache.** `command_line_for()` builds on first use into `cache_dir`,
stamps the build it came from, and reuses thereafter. Note `create_command_line()`
defaults to `force=True`, so reuse requires passing `force=False` explicitly.

Do not bake in a default `source_version`: callers span many Source versions, and the
direct-directory path above makes a version string unnecessary.

---

## 3. Sandbox registry

`IsolatedSource` handles one sandbox. Callers running several, or reusing one across
invocations, currently track them by hand.

**Add a registry** supporting named sandboxes: create, list, reuse, tear down
individually, and sweep at session end.

- **Store:** `.veneer/sandboxes.json`, project-local. One record per sandbox: name,
  session id, PID, bound port, temp directory, log path, creation time.
- **Atomicity:** the file is rewritten atomically, and a record is claimed before its
  sandbox starts, so two concurrent callers cannot select the same name.
- **Ports:** request a base well away from 9876, randomised within a high range so
  concurrent sessions do not collide. Note `IsolatedSource` defaults to `port=9876`
  (`manage.py:386`) — the same default the GUI uses — and that `start()` may bind
  elsewhere, so callers must read the actual port back from `.port`.
- **Session identity is supplied by the caller** via `VENEER_SESSION_ID`, never
  invented and never read back from the registry file. If it were adopted from the
  file, a second concurrent session would inherit the first's id and its sweep would
  tear down the first's live sandboxes. **With no `VENEER_SESSION_ID` set, a caller
  may create and use sandboxes but must not sweep.**
- **Orphan reaping:** records whose PID is no longer alive may be reaped by any
  caller, since the owner cannot return for them. This is the only sanctioned
  cross-session cleanup.
- **Never call `kill_every_running_instance_of_veneer_cmd_line()`**
  (`manage.py:76`) as part of cleanup. It is the obvious reach for tidying up, and
  with concurrent sessions it kills other callers' sandboxes and any running cluster.

**Construction settings the registry must use**, so failures are diagnosable:

- `cleanup='on_clean'`, not the default `'always'`. On startup failure
  `IsolatedSource.__init__` calls `_teardown(remove=(self._cleanup == 'always'))` and
  re-raises — under `'always'` the temp directory and its `veneer_logs/` are deleted,
  destroying the evidence.
- A per-sandbox `tempdir_prefix`, **recorded before construction**. Because the
  constructor raises, there is no object to read `.log_path` or `.directory` from, so
  the directory must be locatable independently.

---

## 4. A real build identifier

There is currently no way to tell which veneer-py is installed. There is no
`veneer.__version__` anywhere in the package, and `pyproject.toml` declares a static
`version = "0.1"` that has never moved — so `importlib.metadata.version('veneer-py')`
returns `0.1` for every install, including the
`pip install .../archive/master.zip` route documented in `CLAUDE.md`.

Any tooling that wants to record or compare "which veneer-py produced this" is
therefore inert today.

**Change:** add `veneer.__version__`, and expose a helper that reports the build —
the git SHA when running from a source checkout, falling back to distribution version
plus install timestamp. The report should say *how* the value was obtained, so a
consumer can distinguish "differs" from "cannot tell".

**Avoid two sources of truth.** If `__version__` is added, `pyproject.toml` should
declare `dynamic = ["version"]` and read from the module rather than keeping a
hand-written literal beside it.

---

## 5. `objdict` documentation correction

`CLAUDE.md` describes `objdict` as "dict with attribute access". `veneer/utils.py:71`
is now:

```python
def objdict(orig):
    return UserDict(orig)
```

with the attribute-access class commented out beneath it. Anyone following the
documentation writes `result.name` and gets an `AttributeError`. Correct the
description.

---

## Merge-base hazard — read before planning

`CLAUDE.md` documents a regression-test framework that is **not on `master`**. The
`feature/regression-test-framework` branch is **20 commits ahead** of `master`, and
its tree contains `veneer/testing/` (9 files) plus a `pyproject.toml` declaring
`[project.entry-points.pytest11]` and a `testing = ["pytest>=7.0"]` extra.

Two consequences:

- **Do not "correct" CLAUDE.md's testing sections.** They are not wrong — they are
  ahead of `master`. Only the `objdict` correction in §5 is unconditionally safe.
- **§4 modifies `pyproject.toml`, and so does that branch.** Establish which of
  `master` or `feature/regression-test-framework` is the merge base before planning,
  and sequence accordingly.

Separately, `pyproject.toml` carries a `[tools.setuptools]` table — a typo for
`[tool.setuptools]` — which setuptools silently ignores, so the
`include-package-data`, `zip-safe`, `packages` and `py_modules` keys inside it have
no effect. Package discovery runs off the empty `[tool.setuptools.packages.find]`
instead. Nothing in this spec depends on it, but it is a latent packaging bug: note
that *renaming* the table would activate a `packages` list conflicting with
`packages.find`, and a `py_modules=['veneer']` entry that is wrong for a package.
Removing it and re-declaring what is actually wanted is the only safe variant.

---

## Testing

**Without Source:**

- `save()` script generation includes the capture/restore of `OutputFile` and
  `Project`.
- `command_line_for()` derives the expected `source_path` / `veneer_path` from a
  representative `v.status()` payload, and reuses a stamped cache rather than
  rebuilding.
- Registry bookkeeping — claim-before-start, atomic rewrite, session-scoped sweep,
  refusal to sweep without `VENEER_SESSION_ID`, PID-based orphan reaping — with the
  launcher mocked.
- The build identifier reports its provenance correctly in each fallback case.

**Requiring Source:**

- `save(preserve_current=True)` genuinely leaves `OutputFile` intact, and a
  subsequent `save(fn=None)` writes to the original path.
- `command_line_for()` produces a command line that starts.
- The registry starts real sandboxes on non-colliding ports, tears down only its own
  records, and preserves `veneer_logs/` when startup fails.

---

## Evidence

Observed against this working tree and a live Source 6.10.0.14373 instance:

- `v.status()` wraps `GET /` (`general.py:274`); the payload carries `HostExe`,
  `PluginsLoaded`, `ProjectFullFilename`, `SourceVersion`.
- The Veneer plugin directory holds `FlowMatters.Source.VeneerCmd.exe` among only 18
  files, with no `RiverSystem.*` or `TIME*` assemblies; the Source build directory
  holds 1144.
- `save()`'s `finally` restores only `ph.CallBackHandler` (`server_side.py:807`).
- `find_veneer_cmd_line_exe` calls `len()` on a `Path.glob()` generator
  (`manage.py:111`).
- `IsolatedSource.__init__` defaults `port=9876` and re-raises after `_teardown()`
  on startup failure (`manage.py:386`).
- No `veneer.__version__`; `importlib.metadata.version('veneer-py')` returns `0.1`.
- `objdict` is `return UserDict(orig)` (`utils.py:71`).
- `feature/regression-test-framework` is 20 commits ahead of `master` with
  `veneer/testing/` present.
