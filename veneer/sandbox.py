'''
A registry of named, throwaway Source sandboxes.

Wraps veneer.manage.IsolatedSource so that several sandboxes can coexist, be
listed and reused across process invocations, and be torn down without
disturbing sandboxes belonging to other sessions.

Typical workflow, composing with veneer.manage.command_line_for() and
VeneerIronPython.save() (v.model.save()):

    v.model.save('scratch_model.rsproj')     # seed a project file from live GUI state
    veneer_exe = command_line_for(v, cache_dir='./.veneer-cache')
    registry = SandboxRegistry(project_dir='.', session_id=os.environ['VENEER_SESSION_ID'])
    sandbox = registry.create('scratch', 'scratch_model.rsproj', veneer_exe)

Each step supplies exactly what the next needs: save() gives a project file
reflecting the live model, command_line_for() gives the veneer_exe to launch
it standalone, and create() launches an isolated, named, throwaway copy.
'''
import json
import os
import random
import time

from .general import Veneer
from .manage import IsolatedSource, write_json_atomic

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
        # Unreachable via veneer.sandbox's own import chain: veneer.manage
        # does `from psutil import Process` unconditionally at module scope,
        # and this module imports from veneer.manage, so psutil is already
        # loaded before any of this code can run. Kept as defence in depth
        # in case that import chain ever changes.
        return True     # cannot tell; assume alive and never reap


def _kill_pid(pid):
    '''
    Terminate the process at pid.

    "Already gone" (psutil.NoSuchProcess) is treated as success - there is
    nothing left to kill. Any other failure (e.g. AccessDenied) is a genuine
    problem and is deliberately NOT swallowed: close() relies on this to
    avoid dropping a sandbox's record when it did not actually manage to
    kill the process (see close()'s docstring).
    '''
    try:
        import psutil
    except ImportError:
        # Unreachable - see the comment in _pid_alive.
        return
    try:
        psutil.Process(pid).terminate()
    except psutil.NoSuchProcess:
        pass            # already gone - that is success, not failure


def _random_port():
    port = random.randint(*PORT_RANGE)
    return port if port != GUI_DEFAULT_PORT else port + 1


class AttachedSandbox(object):
    '''
    A handle to a sandbox started by an EARLIER process.

    Exposes enough to use and to close the sandbox, but not the original
    IsolatedSource object, which died with the process that created it.

    Note: reconstructs a bare Veneer(port). Any trust_env/proxies the
    original sandbox was created with are not persisted in the record and so
    are not restored here. That is a deliberate simplification for the
    common case, not an oversight.
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
            with open(self.path) as fp:
                return json.load(fp)
        except (ValueError, OSError):
            # Corrupt JSON, or a transient/permission problem reading the
            # file (e.g. another process mid-write, or the path is not a
            # plain file). Either way, treat the registry as empty rather
            # than propagate - a stale/corrupt file must not wedge every
            # subsequent call.
            return []

    def _write(self, records):
        # Atomic (temp file + rename) and tolerant of the transient Windows
        # AV/indexer PermissionError on that rename - see write_json_atomic's
        # docstring in veneer.manage for why. Shared with command_line_for()'s
        # build-stamp cache, which hits the exact same failure mode.
        write_json_atomic(self.path, records, tmp_dir=self.dir)

    def list(self):
        return self._read()

    def create(self, name, project_file, veneer_exe, related_files=None, **kwargs):
        '''
        Start a new sandbox named `name` and record it.

        veneer_exe: path to FlowMatters.Source.VeneerCmd.exe - typically the
                    return value of veneer.manage.command_line_for().

        Sequence, and why:
          1. Claim the name by writing a placeholder record (status='failed')
             before starting anything, so a concurrent create() for the same
             name fails fast rather than racing.
          2. Start IsolatedSource(), which can take seconds and copies the
             project - a genuinely long-running step with real failure modes.
          3. On success, re-read the registry fresh (not the list read in
             step 1) and write back the promoted record (status='running',
             actual port/directory/log_path/veneer_pid). Re-reading matters
             because another process may have changed the registry - its own
             create/close/reap_orphans - while step 2 was running; writing
             back a stale list would silently discard that change. Mirrors
             close()'s existing re-read-before-final-write.

        A name whose existing record is 'failed', or whose sandbox process is
        no longer alive, is reusable. A name with a live 'running' record
        raises. If IsolatedSource() itself raises, the placeholder record from
        step 1 survives with status='failed' and its tempdir_prefix intact
        (see cleanup='on_clean' below) and the exception propagates.
        '''
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

        # Only compute a random port when the caller didn't supply one -
        # calling _random_port() unconditionally would burn random state
        # even when its result is discarded.
        port = kwargs.pop('port', None)
        if port is None:
            port = _random_port()

        # A failed start raises from the constructor, leaving no object to read
        # .directory or .log_path from. The record above survives with
        # status='failed' and the tempdir_prefix, so the retained veneer_logs/
        # can still be found. Deliberately not caught here.
        sb = IsolatedSource(project_file, related_files=related_files,
                            veneer_exe=veneer_exe,
                            port=port,
                            tempdir_prefix=prefix,
                            cleanup='on_clean',
                            **kwargs)

        record.update({'port': sb.port, 'directory': sb.directory,
                       'log_path': sb.log_path, 'status': STATUS_RUNNING,
                       'veneer_pid': getattr(sb.process, 'pid', None)})

        # IsolatedSource() can take seconds to start. Re-read immediately
        # before this final write rather than reusing the `records` list read
        # before startup began - another process may have changed the
        # registry (its own create/close/reap_orphans) during that window,
        # and writing back the stale list would silently discard that change.
        # Mirrors close()'s re-read-before-final-write.
        fresh = [r for r in self._read() if r['name'] != name]
        fresh.append(record)
        self._write(fresh)
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
