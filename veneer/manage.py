from subprocess import Popen, PIPE
from psutil import Process
from queue import Queue, Empty  # python 3.x
from threading  import Thread
import sys
from time import sleep, time as _now
from glob import glob
import atexit
import os
import re
import tempfile
import shutil
import threading
from .general import Veneer
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

# Non blocking IO solution from http://stackoverflow.com/a/4896288
ON_POSIX = 'posix' in sys.builtin_module_names

VENEER_EXE_FN='FlowMatters.Source.VeneerCmd.exe'
MANY_VENEERS='D:\\src\\projects\\Veneer\\Compiled'
VENEER_EXE='D:\\src\\projects\\Veneer\\Output\\FlowMatters.Source.VeneerCmd.exe'
IGNORE_LOGS=[
    'log4net:ERROR Failed to find configuration section'
]

# Kestrel's authoritative "bound" message (Source 6+/.NET 8). Reflects the actual port after any
# increment from port-in-use collisions.
_KESTREL_BOUND_RE = re.compile(r'Now listening on:\s*http://[^:/\s]+:(\d+)')
# Veneer's own port announcement. Legacy format used a space separator (" port N"); Source 6 uses
# "http port:N" (colon). The latter can reflect the REQUESTED port for the first instance on a
# taken port and is then superseded by a second emission with the actual port. Treat as a hint
# only; the Kestrel message is authoritative when present.
_VENEER_BOUND_RE = re.compile(r'Started Source RESTful Service on (?:http )?port[ :]\s*(\d+)')

def _dirname(path):
    return os.path.dirname(str(path))

def _basename(path):
    return os.path.basename(str(path))

def _get_version_number (filename):
    from win32api import GetFileVersionInfo, LOWORD, HIWORD
    try:
        info = GetFileVersionInfo (str(filename), "\\")
        ms = info['FileVersionMS']
        ls = info['FileVersionLS']
        return HIWORD (ms), LOWORD (ms), HIWORD (ls), LOWORD (ls)
    except:
        return 0,0,0,0

def ignore_log(line:str):
    for ignore in IGNORE_LOGS:
        if ignore in line:
            return True
    return False

def kill_all_on_exit(processes):
    def end_processes():
        for p in processes:
            p.kill()
    atexit.register(end_processes)

def kill_all_now(processes):
    processes = [p if isinstance(p,Popen) else Process(p) for p in processes]
    for p in processes:
        p.kill()
    for p in processes:
        p.wait()

def kill_every_running_instance_of_veneer_cmd_line():
    import psutil
    for pid in psutil.pids():
        proc = psutil.Process(pid)
        try:
            cmd = proc.cmdline()
        except:
            continue
        if len(cmd) and 'FlowMatters.Source.VeneerCmd.exe' in proc.cmdline()[0]:
            print(f'Killing {pid}')
            proc.kill()

def _enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

def find_veneer_cmd_line_exe(project_fn=None,source_version=None):
    if project_fn:
        project_dir = _dirname(project_fn)
        print('Looking for source_version.txt in %s'%project_dir)
        ver_file = Path(project_dir)/'source_version.txt'
        if ver_file.exists():
            version = open(ver_file).read().split('\n')[0]
            print('Trying to interpret source version=%s'%version)
            if Path(version).exists(): # Actual path to Source...
                direct_path = Path(version)/VENEER_EXE_FN
                print('Trying to go directly to %s'%direct_path)
                if direct_path.exists():
                    return direct_path

            version_in_many = Path(MANY_VENEERS).glob('*'+version+'*')
            if len(version_in_many):
                path_in_many = Path(version_in_many[0])/VENEER_EXE_FN
                print('Trying for %s'%path_in_many)
                if path_in_many.exists():
                    return path_in_many
    return VENEER_EXE

def create_command_line(veneer_path,source_version=None,
                        source_path='C:\\Program Files\\eWater',
                        dest=None,
                        force=True,
                        init_db=False):
    '''
    Copy all Veneer related files and all files from the relevant Source distribution to a third directory,
    for use as the veneer command line.

    veneer_path: Directory containing the Veneer files

    source_version: Version of Source to locate within source_path (e.g. "4.1.1"). If None (default),
                    source_path is treated as the Source build directory directly.

    source_path: Base installation directory for eWater Source, or (when source_version is None) the
                 Source build directory itself.

    dest: Destination to copy Source and Veneer to. If not provided, a temporary directory will be created.

    force: (default True) copy all files even if the directory already exists

    init_db: (default False) if True AND if the command line was created, start an instance in order to
             initialise the database, then terminate.

    Returns: Full path to FlowMatters.Source.VeneerCmd.exe for use with start()

    Note: It is your responsibility to delete the copy of Source and Veneer from the destination directory
    when you are finished with it! (Even if a temporary directory is used!)

    Copy order: Veneer is copied first, then Source. Where files collide (commonly Microsoft/framework
    DLLs), Source's versions win — this is required for Source 6 / .NET 8 builds where Veneer and Source
    each ship their own copies of overlapping dependencies and only Source's versions are compatible.
    '''

    if dest: dest = Path(dest)
    if dest:
        exe_path = dest/'FlowMatters.Source.VeneerCmd.exe'

    if dest and exe_path.exists() and not force:
        return exe_path

    if dest is None:
        dest = Path(tempfile.mkdtemp(suffix='_veneer'))

    if not dest.exists():
        dest.mkdir(parents=True)

    if source_version:
        available = [str(p) for p in Path(source_path).glob('Source *')]
        versions = [_basename(ver).split(' ')[1] for ver in available]
        chosen_one = [ver for ver in versions if ver.startswith(source_version)][-1]
        chosen_one_full = [product_ver for product_ver in available if chosen_one in product_ver][0]
        source_dir = Path(source_path)/chosen_one_full
    else:
        source_dir = Path(source_path)

    source_files = list(source_dir.glob('*.*'))
    if not len(source_files):
        raise Exception('Source files not found at %s'%source_dir)

    veneer_path = Path(veneer_path)
    veneer_files = list(veneer_path.glob('*.*'))
    if not len(veneer_files):
        raise Exception('Veneer files not found at %s'%veneer_path)

    def _copy_tree(src_root, dst_root):
        for child in src_root.iterdir():
            if child.is_file():
                shutil.copyfile(str(child), str(dst_root/child.name))
            elif child.is_dir():
                shutil.copytree(str(child), str(dst_root/child.name), dirs_exist_ok=True)

    # Copy Veneer first, then Source — Source wins on overlapping Microsoft/framework DLLs.
    if veneer_path != dest:
        _copy_tree(veneer_path, dest)
    _copy_tree(source_dir, dest)

    exe_path = Path(dest)/'FlowMatters.Source.VeneerCmd.exe'
    assert exe_path.exists()

    if init_db:
        _proc,_ = start(project_fn=None,n_instances=1,debug=False,veneer_exe=exe_path,ports=9878)
        kill_all_now(_proc)

    return exe_path

def clean_up_cmd_line_exe(path=None):
    if path is None:
        pass

    shutil.rmtree(path)

def configure_non_blocking_io(processes,stream):
    queues = [Queue() for p in range(len(processes))]
    threads = [Thread(target=_enqueue_output,args=(getattr(p,stream),q)) for (p,q) in zip(processes,queues)]
    for t in threads:
        t.daemon = True
        t.start()
    return queues,threads

def _find_plugins_file(project_path):
    project_path = Path(project_path)
    if not project_path.is_dir():
        project_path = _dirname(project_path)

    plugin_fn = Path(project_path)/'Plugins.xml'
    if plugin_fn.exists():
        return plugin_fn

    parent = (Path(project_path).parent).absolute()
    if parent==Path(project_path):
        raise Exception('Could not find Plugins.xml in current or ancestor directories')

    return _find_plugins_file(parent)

def overwrite_plugin_configuration(source_binaries,project_fn=None,plugin_fn=None):
    source_binaries = Path(source_binaries)
    if plugin_fn is None:
        project_fn = Path(project_fn)
        plugin_fn = _find_plugins_file(project_fn)
    if not plugin_fn:
        # logger.warn('Unable to overwrite plugins. No Plugin.xml found')
        return
    print(plugin_fn)

    if source_binaries.is_file():
        source_binaries = Path(_dirname(source_binaries))
    source_binaries = source_binaries/'RiverSystem.Forms.exe'

    source_version = '.'.join([str(v) for v in _get_version_number(source_binaries)[00:-1]])
    print(source_version)

    plugin_dir = Path('C:')/'Users'/os.environ['USERNAME']/'AppData'/'Roaming'/'Source'/source_version
    print(plugin_dir)
    if not plugin_dir.exists():
        plugin_dir.mkdir(parents=True)
    plugin_dest_file = Path(plugin_dir)/'Plugins.xml'
    shutil.copyfile(str(plugin_fn),str(plugin_dest_file))
    assert plugin_dest_file.exists()

def _format_startup_failure(n_instances, failed, ports, cmd_lines, processes,
                            captured_stdout, captured_stderr, tail_lines=40):
    """Build a descriptive exception message for one or more failed Veneer
    instances, including the captured stdout/stderr tail so that callers see
    the actual underlying error (e.g. missing rsproj, port already in use,
    missing plugin) rather than a generic 'failed to start'."""
    msg_lines = ['One or more instances of Veneer failed to start.']
    for i in range(n_instances):
        if not failed[i]:
            continue
        msg_lines.append('')
        msg_lines.append('  Instance %d (requested port %d):' % (i, ports[i]))
        msg_lines.append('    command: %s' % cmd_lines[i])
        exit_code = processes[i].poll()
        if exit_code is not None:
            msg_lines.append('    exit code: %s' % exit_code)
        else:
            msg_lines.append('    exit code: (still running when failure detected)')

        stderr_tail = [ln.rstrip('\r\n') for ln in captured_stderr[i][-tail_lines:]]
        stdout_tail = [ln.rstrip('\r\n') for ln in captured_stdout[i][-tail_lines:]]
        if stderr_tail:
            msg_lines.append('    stderr (last %d lines):' % len(stderr_tail))
            msg_lines.extend('      ' + ln for ln in stderr_tail)
        if stdout_tail:
            msg_lines.append('    stdout (last %d lines):' % len(stdout_tail))
            msg_lines.extend('      ' + ln for ln in stdout_tail)
        if not stderr_tail and not stdout_tail:
            msg_lines.append('    (no output captured before exit)')
    return '\n'.join(msg_lines)


def start(project_fn=None,n_instances=1,ports=9876,debug=False,remote=False,
          script=True, veneer_exe=None,overwrite_plugins=None,return_io=False,
          model=None,start_new_session=False,additional_plugins=[],
          custom_endpoints=[],projects=None,quiet=False,
          detached=False,detached_timeout=120.0,leave_open=False):
    """
    Start one or more copies of the Veneer command line progeram with a given project file

    Parameters:

    - project_fn - Path to a Source project file (.rsproj) to be loaded into all instances

    - n_instances - Number of copies of the Veneer command line to start (default: 1)
                  - Alternatively, specify a list of models (scenarios) to use, one for each instance

    - ports - A single port number, indicating the port number of the first copy of the Veneer command line,
              OR a list of ports, in which case len(ports)==n_instances  (default: 9876)

    - debug - Set to True to echo all output from Veneer Command Line during startup

    - remote - Allow remote connections (requires registration)

    - script - Allow IronPython scripts

    - veneer_exe - Optional (but often required) path to the Veneer Command Line. If not provided,
                   veneer-py will attempt to identify the version of Veneer Command Line to invoke.
                   If there is a source_version.txt file in the same directory as the project file,
                   this text file will be consulted to identify the version of Source.

    - overwrite_plugins - Falsy (default) or True, If truthy, attempt to override the Source plugins using a Plugins.xml file,
                          in the same directory as the project file (or a parent directory).

    - return_io - Return Queues and Threads for asynchronous IO - for reading output of servers (boolean, default: False)

    - model - Specify the model (scenario) to use. Default None (use first scenario). Can specify by scenario name or 1-based index.
            - If a list of models is provided, the length of the list will be used as n_instances

    - additional_plugins - List of plugin files (DLLs) to load

    - projects - List of project files to load. Should be the same length as n_instances

    - quiet - Suppress all output from the servers (default: False)

    - detached - If True, spawn a sibling Python process in a new terminal window to host the Veneer
                 instances, rather than creating them as children of the current process. Useful when
                 the current interpreter is hosted by something that interferes with child-process IO
                 (e.g. the VS Code Jupyter extension). Windows only; raises NotImplementedError
                 elsewhere. Note: when detached=True, return_io is ignored.

    - detached_timeout - Maximum seconds to wait for the detached launcher to become ready
                         (default: 120.0). Only used when detached=True.

    - leave_open - If True (and detached=True), the spawned terminal window is left open when Veneer
                   children are killed, so the user can read final log output. Default False, which
                   lets the terminal close once its children die.

        returns processes, ports
       processes - list of process objects that can be used to terminate the servers
       ports - the port numbers used for each copy of the server
       ((stdout_queues,stdout_threads),(stderr_queues,stderr_threads)) if return_io
    """
    if detached:
        return _start_detached(
            start_kwargs=dict(
                project_fn=str(project_fn) if project_fn else None,
                n_instances=n_instances,
                ports=ports if isinstance(ports, int) else list(ports),
                debug=debug,
                remote=remote,
                script=script,
                veneer_exe=str(veneer_exe) if veneer_exe else None,
                overwrite_plugins=overwrite_plugins,
                return_io=False,
                model=model,
                start_new_session=False,
                additional_plugins=list(additional_plugins) if additional_plugins else [],
                custom_endpoints=list(custom_endpoints) if custom_endpoints else [],
                projects=[str(p) for p in projects] if projects else None,
                quiet=quiet,
            ),
            detached_timeout=detached_timeout,
            leave_open=leave_open,
        )

    if problem_launcher_warning := _detect_problematic_launcher():
        logger.warning(problem_launcher_warning)

    if not hasattr(ports,'__len__'):
        ports = list(range(ports,ports+n_instances))

    if not veneer_exe:
        veneer_exe = find_veneer_cmd_line_exe(project_fn)

    if projects and len(projects):
        n_instances = len(projects)
        ports = ports[:n_instances]
        projects = [Path(p).absolute() for p in projects]

    if project_fn:
        project_fn = Path(project_fn).absolute()
        if overwrite_plugins:
            overwrite_plugin_configuration(veneer_exe,project_fn)
        if not projects or not len(projects):
            projects = [project_fn] * n_instances

    extras = ''
    if remote: extras += '-r '
    if script: extras += '-s '
    if model is not None:
        if isinstance(model,int) and model <= 0:
            raise Exception('Model index must be 1-based')

        if isinstance(model,list):
            n_instances = len(model)
        else:
            model = [model] * n_instances

        model_args = []
        for m in model:
            m = str(m)
            if ' ' in m:
                m = '"%s"'%m

            model_args.append(' -m %s '%m)
    else:
        model_args = [' '] * n_instances

    if not hasattr(ports,'__len__'):
        ports = list(range(ports,ports+n_instances))

    if len(additional_plugins):
        extras += ' -l '+ quote_if_space(','.join(additional_plugins))

    if len(custom_endpoints):
        extras += ' -c '+ quote_if_space(','.join(custom_endpoints))

    cmd_lines = ['%s -p %d %s %s'%(veneer_exe,port,model,extras) for (port,model) in zip(ports,model_args)]

    if projects and len(projects):
        cmd_lines = ['%s "%s"'%(cmd_line,str(project_fn)) for cmd_line,project_fn in zip(cmd_lines,projects)]
        # extras += ' "%s"'%str(project_fn)

    # cmd_lines = [cmd_line%port for port in ports]
    if debug:
        for cmd in cmd_lines:
            print('Starting %s'%cmd)
    kwargs={}
    if start_new_session:
        if ON_POSIX:
            kwargs['start_new_session']=start_new_session
        else:
            cmd_line = 'start "Veneer server for %s" %s'%(os.path.basename(project_fn),cmd_line)
            kwargs['shell']=True
    processes = [Popen(cmd,stdout=PIPE,stderr=PIPE,bufsize=1, close_fds=ON_POSIX, **kwargs) for cmd in cmd_lines]
    std_out_queues,std_out_threads = configure_non_blocking_io(processes,'stdout')
    std_err_queues,std_err_threads = configure_non_blocking_io(processes,'stderr')

    ready = [False for p in range(n_instances)]
    failed = [False for p in range(n_instances)]
    # Buffer all stdout/stderr per instance during startup so that, if any instance fails, we
    # can include the actual error output in the exception we raise. Without this, callers
    # (including non-interactive ones whose stdout is captured/redirected) only see the
    # generic "failed to start" message and lose the specific reason — e.g. that the rsproj
    # file does not exist.
    captured_stdout = [[] for _ in range(n_instances)]
    captured_stderr = [[] for _ in range(n_instances)]
    # True once Kestrel has reported the actual bound port for instance i. Prefer this over any
    # earlier hint from Veneer's own "Started Source RESTful Service" line (which can reflect the
    # requested port before a port-in-use increment on Source 6+).
    port_authoritative = [False for p in range(n_instances)]
    # Timestamp of "Server started. Ctrl-C to exit" for instance i. Used to fall back to the
    # Veneer-announced port if Kestrel's authoritative message never arrives (legacy Source).
    server_started_at = [None for p in range(n_instances)]
    LEGACY_GRACE_SECS = 5.0

    all_ready = False
    any_failed = False
    actual_ports = ports[:]
    while not (all_ready or any_failed):
        for i in range(n_instances):
            process_exited = processes[i].poll() is not None
            if process_exited:
                failed[i]=True

            end=False
            while not end:
                try:
                    line = std_err_queues[i].get_nowait().decode('utf-8')
                except Empty:
                    end = True
                    pass
                else:
                    if not ignore_log(line):
                        captured_stderr[i].append(line)
                        if not quiet:
                            print('ERROR[%d] %s'%(i,line))

            if ready[i]:
                continue

            end = False
            while not end:
                try:
                    line = std_out_queues[i].get_nowait().decode('utf-8')
                except Empty:
                    end = True
                    sleep(0.05)
                else:
                    captured_stdout[i].append(line)
                    kestrel_match = _KESTREL_BOUND_RE.search(line)
                    if kestrel_match:
                        actual_ports[i] = int(kestrel_match.group(1))
                        port_authoritative[i] = True
                    else:
                        veneer_match = _VENEER_BOUND_RE.search(line)
                        if veneer_match and not port_authoritative[i]:
                            actual_ports[i] = int(veneer_match.group(1))

                    if line.startswith('Server started. Ctrl-C to exit'):
                        if server_started_at[i] is None:
                            server_started_at[i] = _now()
                    elif line.startswith('Cannot find project') or line.startswith('Unhandled exception'):
                        end = True
                        failed[i] = True
                        if not quiet:
                            print('Server %d on port %d failed to start: %s'%(i,ports[i],line.rstrip('\r\n')))
                        sys.stdout.flush()
                    if debug and not line.startswith('Unable to delete'):
                        if not quiet:
                            print("[%d] %s"%(i,line))
                        sys.stdout.flush()

            # Decide readiness per instance after each drain pass.
            # - Kestrel message seen: authoritative port, ready immediately.
            # - Legacy Source (no Kestrel output): after "Server started" plus a short grace,
            #   accept the Veneer-announced port. Without any port signal, fall back to the
            #   requested port — the legacy behaviour prior to this change.
            if not ready[i] and not failed[i]:
                legacy_timeout = (server_started_at[i] is not None
                                  and (_now() - server_started_at[i]) >= LEGACY_GRACE_SECS)
                if port_authoritative[i] or legacy_timeout:
                    ready[i] = True
                    if not quiet:
                        print('Server %d on port %d is ready'%(i,actual_ports[i]))
                    sys.stdout.flush()
        all_ready = len([r for r in ready if not r])==0
        any_failed = len([f for f in failed if f])>0

    if any_failed:
        # Give the reader threads a moment to drain any remaining buffered output from
        # processes that have just exited, then pull whatever is left in the queues. This
        # ensures the actual error message (e.g. the line naming the missing rsproj) makes
        # it into the exception even if it arrived after we first noticed the process had
        # exited.
        sleep(0.5)
        for i in range(n_instances):
            while True:
                try:
                    line = std_err_queues[i].get_nowait().decode('utf-8')
                except Empty:
                    break
                if not ignore_log(line):
                    captured_stderr[i].append(line)
            while True:
                try:
                    line = std_out_queues[i].get_nowait().decode('utf-8')
                except Empty:
                    break
                captured_stdout[i].append(line)

        for p in processes:
            if p.poll() is None:
                p.kill()

        raise Exception(_format_startup_failure(
            n_instances, failed, ports, cmd_lines, processes,
            captured_stdout, captured_stderr,
        ))

    kill_all_on_exit(processes)

    if return_io:
        return processes,actual_ports,((std_out_queues,std_out_threads),(std_err_queues,std_err_threads))
    return processes,actual_ports

def _start_detached(start_kwargs, detached_timeout, leave_open):
    """Spawn ``veneer.detached_start`` in a new terminal window, wait for it
    to publish actual ports, and return ``(processes, actual_ports)`` in the
    same shape as the in-process ``start()`` path.

    Windows-only for now. On POSIX, raises NotImplementedError.
    """
    import json
    import subprocess
    from psutil import Process as _PsProcess

    from .detached_handshake import (
        HandshakeStatus,
        wait_for_status,
        write_handshake,
    )

    if not sys.platform.startswith("win"):
        raise NotImplementedError(
            "detached=True is currently Windows-only. Open an issue with your "
            "preferred terminal emulator if you need POSIX support."
        )

    work_dir = Path(tempfile.mkdtemp(prefix="veneer_detached_"))
    config_path = work_dir / "config.json"
    handshake_path = work_dir / "handshake.json"

    config_path.write_text(json.dumps({
        "handshake_path": str(handshake_path),
        "start_kwargs": start_kwargs,
    }))

    write_handshake(handshake_path, {
        "status": HandshakeStatus.STARTING,
        "launcher_pid": None,
        "veneer_pids": [],
        "actual_ports": [],
        "error": None,
    })

    # Fix 1: Register work_dir cleanup before the wait so KeyboardInterrupt also triggers it.
    def _cleanup_work_dir():
        shutil.rmtree(work_dir, ignore_errors=True)
    atexit.register(_cleanup_work_dir)

    # Fix 4: Use list2cmdline so paths with spaces are quoted correctly on Windows.
    title = f"Veneer launcher ({start_kwargs.get('n_instances') or 1})"
    inner = subprocess.list2cmdline(
        [sys.executable, "-m", "veneer.detached_start", str(config_path)]
    )
    cmd = f'start "{title}" cmd /k {inner}'
    subprocess.Popen(cmd, shell=True)

    try:
        payload = wait_for_status(
            handshake_path,
            {HandshakeStatus.READY, HandshakeStatus.FAILED},
            timeout=detached_timeout,
        )
    except TimeoutError:
        raise TimeoutError(
            f"Detached Veneer launcher did not become ready within "
            f"{detached_timeout}s. Config at {config_path}, handshake at "
            f"{handshake_path}."
        )

    if payload["status"] == HandshakeStatus.FAILED:
        raise RuntimeError(
            f"Detached Veneer launcher failed: {payload.get('error')}"
        )

    processes = [_PsProcess(pid) for pid in payload["veneer_pids"]]
    actual_ports = payload["actual_ports"]

    # Fix 3: Reconstruct a Process for the launcher PID so we can kill it when leave_open=False.
    launcher_proc = _PsProcess(payload["launcher_pid"]) if payload.get("launcher_pid") else None

    # Fix 2: Always register Veneer children for atexit teardown regardless of leave_open.
    kill_all_on_exit(processes)

    # Fix 3: Kill the launcher terminal only when leave_open=False (registered separately so
    # callers that call kill_all_now(processes) directly don't accidentally hit the launcher).
    if not leave_open and launcher_proc is not None:
        kill_all_on_exit([launcher_proc])

    return processes, actual_ports

def print_from_all(queues,prefix=''):
    for i in range(len(queues)):
        end=False
        while not end:
            try:
                line = queues[i].get_nowait().decode('utf-8')
            except Empty:
                end = True
                pass
            else:
                print('%s[%d] %s'%(prefix,i,line))

def find_processes(search,ext='.exe'):
    import psutil
    processes = psutil.process_iter()
    def try_name(p):
        try: return p.name()
        except:return None

    if not search.endswith(ext):
        search += ext

    return [p for p in processes if try_name(p)==search]

def find_veneers(search=None):
    veneers = find_processes('FlowMatters.Source.VeneerCmd')

    if search:
        veneers = [v for v in veneers if search in ' '.join(v.cmdline())]
    return veneers

def cleanup_project_temp_dir(dir,background=True):
    dir = Path(dir)
    if not dir.exists():
        return
    
    holding_dir = (Path(dir)/'..'/'veneer-holding-directory-for-removal').absolute()
    
    project_temp_dirs = dir.glob('*')

    for project_temp_dir in project_temp_dirs:
        if not project_temp_dir.is_dir():
            continue
        try:
            potential_pid = int(project_temp_dir.name)
        except:
            continue
        try:
            _process = Process(potential_pid)
            if _process.is_running():
                    print('Process %d is running. Ignoring'%potential_pid)
                    continue
        except:
            pass
        print('Cleaning up %s'%project_temp_dir)
        shutil.move(str(project_temp_dir),str(holding_dir/str(potential_pid)))
    if holding_dir.exists():
        if background:
            print('Cleaning up %s in background.'%holding_dir)
            threading.Thread(target = lambda : shutil.rmtree(str(holding_dir))).start()
        else:
            print('Cleaning up %s.'%holding_dir)
            shutil.rmtree(str(holding_dir))
class BulkVeneerApplication(object):
    def __init__(self,clients,name):
        self.clients = clients
        self.names = [name] 

    def __getattr__(self,attrname):
        self.names.append(attrname)
        return self

    def __call__(self,*pargs,**kwargs):
        return self.clients.call_on_all(self.names,*pargs,**kwargs)

class BulkVeneer(object):
    def __init__(self,ports=[],clients=[],verbose=False):
        self.veneers = [Veneer(port) for port in ports]
        self.veneers += clients
        self.verbose = verbose

    def call_path(self,client,path,*pargs,**kwargs):
        if self.verbose:
            print('Calling %s on port %d'%('.'.join(path),client.port))
        target = client
        for p in path:
            target = getattr(target,p)
        return target(*pargs,**kwargs)

    def call_on_all(self,path,*pargs,**kwargs):
        result = [self.call_path(v,path,*pargs,**kwargs) for v in self.veneers]
        result = [r for r in result if not r is None]

        if 'run_async' in kwargs and kwargs['run_async']:
            return [r.getresponse().getcode() for r in result]

        if len(result):
            return result
        return None

    def __getattr__(self,attrname):
        return BulkVeneerApplication(self,attrname)

# +++ Need something to read latest messages from processes...

def quote_if_space(fn:str)->str:
    if (' ' in fn) and (not fn.startswith('"')) and (not fn.endswith('"')):
        return '"%s"'%fn
    return fn

def _detect_problematic_launcher():
    """Return a warning string if the current process chain looks like it may
    spawn child processes without proper interactive window station
    attachment, otherwise None."""
    try:
        import psutil, sys
        if not sys.platform.startswith('win'):
            return None
        names = {p.name().lower() for p in psutil.Process().parents()}
        if 'code.exe' in names:
            return ('Detected VS Code in parent process chain. Launching Veneer '
                    'instances from a VS Code Jupyter notebook is known to cause '
                    'intermittent startup hangs. If you see Veneer instances '
                    'unable to respond to requests, run the same code from Jupyter Lab '
                    'or a plain python file. See veneer-py docs.')
    except Exception:
        pass
    return None
