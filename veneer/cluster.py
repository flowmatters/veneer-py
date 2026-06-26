import json
from .manage import start, kill_all_now, copy_project_files
from ._proxy import AttributeChainProxy
from dask.distributed import LocalCluster, Client
from psutil import Process
import dask
from .general import Veneer
import os
import tempfile
import shutil
from functools import partial
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerInfo:
    """Per-worker information held in VeneerCluster.worker_affinity.

    port: the Veneer HTTP port for this worker.
    directory: directory containing the Source project file used by this worker.
    veneer_pid: PID of the FlowMatters.Source.VeneerCmd.exe process. None when
                the PID is not known (e.g., legacy cluster-config reconnect
                from a JSON file written before the migration); downstream
                liveness checks must guard for None and treat it as "unknown,
                cannot detect death" rather than passing it to psutil.
    log_path: filesystem path to the captured stdout+stderr log for this
              VeneerCmd process. None when capture is not configured (e.g.,
              start() called without capture_output_dir).
    """
    port: int
    directory: str
    veneer_pid: int | None = None
    log_path: str | None = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        raw_pid = d.get('veneer_pid')
        return cls(
            port=int(d['port']),
            directory=d['directory'],
            veneer_pid=int(raw_pid) if raw_pid is not None else None,
            log_path=d.get('log_path'),
        )


class WorkerDied(RuntimeError):
    """Raised by run_jobs(partial_results=False) when one or more cluster
    jobs failed because their VeneerCmd worker process exited.

    failures: list of structured failure dicts as produced by the wrapper.
    """
    def __init__(self, failures):
        self.failures = failures
        dead_ports = sorted({f['port'] for f in failures if f.get('status') == 'worker_dead'})
        super().__init__(
            f'{len(failures)} cluster job(s) failed; '
            f'workers down on ports: {dead_ports}'
        )


MAX_DEFAULT_CLUSTER_SIZE=64

def _make_emitter(callback):
    """Return a (stage, current, total, message) emitter that swallows callback exceptions."""
    def _emit(stage, current, total, message):
        if callback is None:
            return
        try:
            callback(stage, current, total, message)
        except Exception:
            logger.warning('progress_callback raised', exc_info=True)
    return _emit

@dask.delayed
def copy_project(prefix,project_file,extras):
    assert os.path.exists(project_file)
    tmp = tempfile.mkdtemp(prefix=prefix)
    assert os.path.exists(tmp)
    copy_project_files(project_file, extras, tmp)
    return tmp

@dask.delayed
def remove_copy(directory):
    shutil.rmtree(directory)

@dask.delayed
def scenario_info(port, veneer_kwargs=None):
    v = Veneer(port, **(veneer_kwargs or {}))
    info = v.scenario_info()
    info['port'] = port
    return info

def run_on_cluster(cluster,fn):
    '''
    Wrap a function to run on a specific veneer cluster

    fn should accept kwargs v (veneer client)
    if the cluster.copy_projects is True, then fn should also accept directory:str (the temp directory where the project file is located)
    '''
    logger.info('Wrapping %s to run on cluster %s'%(fn.__name__,cluster.name))
    worker_map = cluster.worker_affinity
    copy_projects = cluster.copy_projects
    veneer_kwargs = cluster._veneer_kwargs
    @dask.delayed
    def inner_wrapped(*args, **kwargs):
        import os
        import time
        import requests
        import psutil
        import veneer
        from veneer._failure import safe_exit_code, tail

        worker_pid = os.getpid()
        info = worker_map[worker_pid]
        pid_known = info.veneer_pid is not None
        alive_before = psutil.pid_exists(info.veneer_pid) if pid_known else None
        inner_args = {'v': veneer.Veneer(info.port, **veneer_kwargs)}
        if copy_projects:
            inner_args['directory'] = info.directory
        try:
            started_at = time.time()
            value = fn(*args, **kwargs, **inner_args)
            return {
                'status': 'ok',
                'port': info.port,
                'pid': info.veneer_pid,
                'value': value,
                'directory': info.directory,
            }
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as exc:
            # Without a known PID we cannot tell whether the worker process died,
            # so we cannot promise a 'worker_dead' classification — propagate the
            # original exception unchanged.
            if not pid_known:
                raise
            if not psutil.pid_exists(info.veneer_pid):
                return {
                    'status': 'worker_dead',
                    'port': info.port,
                    'pid': info.veneer_pid,
                    'directory': info.directory,
                    'log_path': info.log_path,
                    'started_at': started_at,
                    'died_at': time.time(),
                    'alive_before': alive_before,
                    'exit_code': safe_exit_code(info.veneer_pid),
                    'log_tail': tail(info.log_path, 200),
                    'exception': repr(exc),
                }
            raise  # Process alive — propagate as today (transient or Source-side).

    return inner_wrapped

def check_existing_cluster_temp_directory(prefix,behaviour_on_existing):
    if behaviour_on_existing == 'raise':
        if len([d for d in os.listdir(tempfile.gettempdir()) if d.startswith(prefix)]):
            raise Exception('Found existing temporary directories with prefix %s. Use existing="remove" to automatically remove old directories.'%prefix)
    elif behaviour_on_existing == 'remove':
        for d in os.listdir(tempfile.gettempdir()):
            if d.startswith(prefix):
                logger.info('Removing existing temporary directory %s with prefix %s',d,prefix)
                shutil.rmtree(os.path.join(tempfile.gettempdir(),d))
    elif behaviour_on_existing == 'ignore':
        for d in os.listdir(tempfile.gettempdir()):
            if d.startswith(prefix):
                logger.info('Found existing temporary directory %s with prefix %s',d,prefix)
    else:
        raise Exception('Unknown behaviour on existing temporary directories: %s'%behaviour_on_existing)

class ClusterVeneerClient(object):
    def __init__(self, cluster):
        self._cluster = cluster
        self._dummy_v = Veneer(0, **cluster._veneer_kwargs)

    def __getattr__(self, attrname):
        if attrname.startswith('_'):
            raise AttributeError(attrname)
        return AttributeChainProxy(self._make_resolver(), (attrname,))

    def _make_resolver(self):
        cluster = self._cluster
        veneer_kwargs = cluster._veneer_kwargs
        def resolve(path, args, kwargs):
            @dask.delayed
            def fn(p):
                target = Veneer(p, **veneer_kwargs)
                for n in path:
                    target = getattr(target, n)
                return target(*args, **kwargs)
            return cluster.run_on_each(fn)
        return resolve

    def __dir__(self):
        return self._dummy_v.__dir__()

class VeneerCluster(object):
    def __init__(self,project_file=None,veneer_exe=None,n_workers=None,debug=False,remote=False,
          script=True,overwrite_plugins=None,custom_endpoints=None,additional_plugins=None,
          copy=False,tempdir_prefix='source-cluster-',copy_extras=None,existing='raise',
          existing_cluster=None,cleanup_on_failure=True,progress_callback=None,
          trust_env=None,proxies=None):
        '''
        Create a cluster of Veneer instances, each running in a separate process

        project_file: str
            The path to the source project file to be copied for each instance
        veneer_exe: str
            The path to the Veneer executable
        n_workers: int
            The number of workers to create. If None, will use the number of CPUs on the machine, up to MAX_DEFAULT_CLUSTER_SIZE
        debug: bool
            Whether to log debugging information from the Veneer instances during initialisation
        remote: bool
            Whether to configure the Veneer instances to accept remote connections
        script: bool
            Whether to run the Veneer instances in script mode
        overwrite_plugins: list
            A list of plugin names to overwrite in the Veneer instances
        custom_endpoints: list
            A list of custom endpoints to add to the Veneer instances
        additional_plugins: list
            A list of additional plugins to load in the Veneer instances
        copy: bool
            Whether to copy the project file for each instance. Note automatically set to True if copy_extras contains any files
        tempdir_prefix: str
            The prefix to use for the temporary directories if copy is True
        copy_extras: list
            A list of additional files or directories to copy for each instance
        existing: str
            What to do if there are existing temporary directories with the same prefix
            - 'raise': raise an exception
            - 'remove': remove the existing directories
            - 'ignore': ignore the existing directories
        existing_cluster: str
            The path to an existing cluster to use. If None, a new cluster will be created
        cleanup_on_failure: bool
            If True (default), tear down any partially-started resources (Veneer
            instances, temp directories, dask cluster) when startup fails. Set to
            False to leave them running for investigation.
        progress_callback: Optional callable invoked at startup/shutdown milestones with
            (stage: str, current: int, total: int, message: str).
            Stages: 'dask-init', 'project-copy', 'veneer-start' (forwarded from
            manage.start), 'affinity-mapping', 'ready', 'connect-existing',
            'shutdown-veneer', 'shutdown-temp', 'shutdown-dask'. The callback runs
            on the calling thread; exceptions are caught and logged.
        trust_env, proxies: Forwarded to every Veneer client constructed by this
            cluster (workers, ClusterVeneerClient, and the per-job clients used
            inside dask tasks). Default behaviour is unchanged: loopback hosts
            automatically bypass any inherited HTTP_PROXY/HTTPS_PROXY environment
            variables. Set trust_env=True or pass an explicit proxies dict only
            when you genuinely need the cluster's HTTP calls routed through a
            proxy. See Veneer.__init__ for full semantics.
        '''

        self._progress_callback = progress_callback
        emit = _make_emitter(progress_callback)
        self._veneer_kwargs = {'trust_env': trust_env, 'proxies': proxies}

        self.wrap = partial(run_on_cluster,self)
        self.v = ClusterVeneerClient(self)
        self._workers = None
        self.veneer_processes = []
        self.temp_directories = []
        self.dask_cluster = None
        self.dask_client = None

        if existing_cluster is not None:
            if os.path.exists(existing_cluster):
                logger.info('Using existing cluster at %s',existing_cluster)
                with open(existing_cluster,'r') as f:
                    existing_cluster = json.load(f)
            elif isinstance(existing_cluster,dict):
                logger.info('Using existing cluster with config')
            else:
                raise Exception('Invalid existing cluster: %s'%existing_cluster)

            emit('connect-existing', 0, 1, f"Connecting to existing cluster at {existing_cluster.get('dask_scheduler', '')}")
            self.dask_client = Client(existing_cluster['dask_scheduler'])
            self.dask_cluster = self.dask_client.cluster

            self.original_project_file = existing_cluster['project_file']
            self.name = existing_cluster['name']
            self.n_workers = existing_cluster['n_workers']
            self.veneer_ports = existing_cluster['veneer_ports']
            self.veneer_processes = [Process(p) for p in existing_cluster['veneer_processes']]
            self.temp_directories = existing_cluster['temp_directories']
            self.worker_affinity = {
                int(k): WorkerInfo.from_dict(v) if isinstance(v, dict) else WorkerInfo(
                    port=int(v[0]), directory=v[1], veneer_pid=None, log_path=None,
                )
                for k, v in existing_cluster['worker_affinity'].items()
            }
            self.copy_projects = existing_cluster['copy_projects']
            emit('connect-existing', 1, 1, f'Connected to existing cluster ({self.n_workers} workers)')
            return

        check_existing_cluster_temp_directory(tempdir_prefix,existing)
        self.original_project_file = project_file
        self.name = f'{n_workers} node cluster for {project_file}'

        if n_workers is None:
            n_workers = min(os.cpu_count(),MAX_DEFAULT_CLUSTER_SIZE)
        self.n_workers = n_workers or os.cpu_count()

        try:
            logger.info('Initialising DASK cluster with %d workers',self.n_workers)
            emit('dask-init', 0, 1, 'Initialising Dask cluster')
            self.dask_cluster = LocalCluster(threads_per_worker=1,n_workers=self.n_workers)
            self.dask_client = Client(self.dask_cluster)
            emit('dask-init', 1, 1, 'Dask cluster ready')

            if not copy and (copy_extras is not None) and len(copy_extras):
                logger.info('Copying project file for each instance to copy extras')
                copy = True
            self.copy_projects = copy

            if copy:
                logger.info('Creating %d copies of project file %s', n_workers, project_file)
                emit('project-copy', 0, n_workers, f'Copying project files (0/{n_workers} complete)')
                from dask.distributed import as_completed
                creation = [copy_project(tempdir_prefix, project_file, copy_extras or []) for _ in range(n_workers)]
                futures = self.dask_client.compute(creation, sync=False)
                # Track original index alongside each future so the result list is reassembled in order.
                fut_to_idx = {f: i for i, f in enumerate(futures)}
                results = [None] * n_workers
                completed = 0
                for fut in as_completed(futures):
                    results[fut_to_idx[fut]] = fut.result()
                    completed += 1
                    emit('project-copy', completed, n_workers, f'Copying project files ({completed}/{n_workers} complete)')
                self.temp_directories = results
                self.project_files = [os.path.join(t, os.path.basename(self.original_project_file)) for t in self.temp_directories]
            else:
                self.project_files = [self.original_project_file] * self.n_workers

            logger.info('Starting %d Veneer instances',self.n_workers)
            # Capture VeneerCmd stdout/stderr next to the (first) project temp dir so the
            # incident reporter can read it post-mortem. When project files aren't being
            # copied per-worker, fall back to the system temp dir.
            capture_dir = (
                os.path.join(self.temp_directories[0], 'veneer_logs')
                if self.temp_directories else
                os.path.join(tempfile.gettempdir(), f'{tempdir_prefix}veneer_logs')
            )
            veneer_processes, veneer_ports, veneer_log_paths = start(
                n_instances=self.n_workers,debug=debug,remote=remote,
                script=script, veneer_exe=veneer_exe,overwrite_plugins=overwrite_plugins,
                additional_plugins=additional_plugins or [],custom_endpoints=custom_endpoints or [],
                projects=self.project_files,
                progress_callback=progress_callback,
                capture_output_dir=capture_dir,
                return_log_paths=True,
            )
            self.veneer_ports = veneer_ports
            self.veneer_processes = veneer_processes
            self.veneer_log_paths = veneer_log_paths

            logger.info('Assigning Veneer instances to DASK workers')
            emit('affinity-mapping', 0, 1, 'Mapping Veneer ports to Dask workers')
            self.worker_affinity = {}
            worker_info = self.dask_cluster.workers
            scenario_jobs = [scenario_info(p, self._veneer_kwargs) for p in self.veneer_ports]
            veneer_info_map = self.dask_client.compute(scenario_jobs, sync=True)

            # veneer_processes already aligned to ports by index (see start() return shape).
            veneer_pid_by_port = {port: proc.pid for port, proc in zip(self.veneer_ports, self.veneer_processes)}
            log_path_by_port = dict(zip(self.veneer_ports, self.veneer_log_paths))
            self.worker_affinity = {
                w.pid: WorkerInfo(
                    port=info['port'],
                    directory=os.path.dirname(info['ProjectFullFilename']),
                    veneer_pid=veneer_pid_by_port[info['port']],
                    log_path=log_path_by_port.get(info['port']),
                )
                for w, info in zip(worker_info.values(), veneer_info_map)
            }
            emit('affinity-mapping', 1, 1, 'Mapping complete')
            emit('ready', 1, 1, f'Cluster ready ({self.n_workers} workers)')
        except BaseException:
            if cleanup_on_failure:
                logger.exception('Cluster startup failed; cleaning up partial resources')
                self._cleanup_partial()
            else:
                logger.exception('Cluster startup failed; leaving partial resources in place for investigation')
            raise

    def _cleanup_partial(self):
        '''
        Tear down any resources that may have been acquired during a failed startup.
        Tolerant of partial state — each step is guarded independently.
        '''
        if self.veneer_processes:
            try:
                kill_all_now(self.veneer_processes)
            except Exception:
                logger.exception('Error killing Veneer processes during cleanup')
            self.veneer_processes = []

        if self.temp_directories:
            for d in self.temp_directories:
                try:
                    shutil.rmtree(d)
                except Exception:
                    logger.exception('Error removing temp directory %s during cleanup',d)
            self.temp_directories = []

        if self.dask_client is not None:
            try:
                self.dask_client.close()
            except Exception:
                logger.exception('Error closing dask client during cleanup')
            self.dask_client = None

        if self.dask_cluster is not None:
            try:
                self.dask_cluster.close()
            except Exception:
                logger.exception('Error closing dask cluster during cleanup')
            self.dask_cluster = None

    @property
    def workers(self):
        '''
        Access individual Veneer clients by index.

        Example: cluster.workers[0].retrieve_json('/runs')
        '''
        if self._workers is None or len(self._workers) != len(self.veneer_ports):
            self._workers = [Veneer(p, **self._veneer_kwargs) for p in self.veneer_ports]
        return self._workers

    def worker_alive(self, port: int) -> bool:
        '''psutil-based liveness check for the VeneerCmd process on this port.

        Does not hit HTTP. Returns False for unknown ports, and for ports whose
        WorkerInfo.veneer_pid is None (legacy reconnect path where the PID was
        not preserved in the cluster config JSON).
        '''
        import psutil
        for info in self.worker_affinity.values():
            if info.port == port:
                if info.veneer_pid is None:
                    return False
                return psutil.pid_exists(info.veneer_pid)
        return False

    def to_json(self,fn=None):
        '''
        Save the cluster configuration to a JSON file

        fn: str
            The path to the file to save the configuration to. If None, will return JSON string
        '''
        config = {
            'project_file': self.original_project_file,
            'n_workers': self.n_workers,
            'name': self.name,
            'veneer_ports': self.veneer_ports,
            'veneer_processes': [p.pid for p in self.veneer_processes],
            'temp_directories': self.temp_directories,
            'worker_affinity': {pid: info.to_dict() for pid, info in self.worker_affinity.items()},
            'dask_scheduler': self.dask_client.scheduler.address,
            'copy_projects': self.copy_projects,
        }
        txt = json.dumps(config,indent=2)

        if fn is not None:
            with open(fn,'w') as f:
                f.write(txt)
            return None

        return txt

    def run_on_each(self,operation,arg='port'):
        '''
        Run an operation on each Veneer instance in the cluster

        operation: function
            A function to run for each worker that accepts a port number
        arg: str
            UNUSED
        '''
        operation_ = dask.delayed(operation)
        jobs = [operation_(p) for p in self.veneer_ports]
        return self.dask_client.compute(jobs,sync=True)

    def run_jobs(self, jobs, sync=True, partial_results=False):
        '''Run a set of jobs on the cluster.

        jobs: list of dask.delayed jobs (wrapped via cluster.wrap or otherwise).
        sync: True (default) → block and return results; False → return Future.
        partial_results: False (default) → preserve historical behaviour. If
                 every result is 'ok', unwrap to the legacy
                 [(worker_tuple, value), ...] shape. If any result has
                 status != 'ok', raise WorkerDied carrying the structured
                 failure list. True → return the structured list of dicts as
                 produced by the wrapper (empty input returns an empty list).
        '''
        if sync:
            results = self.dask_client.compute(jobs, sync=True)
            if partial_results:
                return list(results)
            return _unwrap_or_raise(results)

        # sync=False: dask_client.compute returns list[Future] when jobs is a
        # list, not a single Future. Both branches below wrap it into a single
        # Future via dask_client.submit, which deep-resolves Futures inside
        # the args before calling — so the closure receives the resolved list.
        future_results = self.dask_client.compute(jobs, sync=False)

        if partial_results:
            return self.dask_client.submit(list, future_results)

        return self.dask_client.submit(_unwrap_or_raise, future_results)

    def shutdown(self, progress_callback=None):
        '''
        Shutdown the cluster. Remove any temporary directories created for the project files.

        progress_callback: optional override for this call only. If None, falls back to the
                           callback that was passed to the constructor (if any).
        '''
        callback = progress_callback if progress_callback is not None else self._progress_callback
        emit = _make_emitter(callback)

        n_proc = len(self.veneer_processes)
        emit('shutdown-veneer', 0, n_proc, f'Stopping Veneer instances (0/{n_proc} stopped)')
        for k, p in enumerate(self.veneer_processes, start=1):
            try:
                p.kill()
                p.wait()
            except Exception:
                logger.exception('Error killing Veneer process %s during shutdown', p)
            emit('shutdown-veneer', k, n_proc, f'Stopping Veneer instances ({k}/{n_proc} stopped)')

        n_temp = len(self.temp_directories)
        if n_temp:
            emit('shutdown-temp', 0, n_temp, f'Removing temporary project files (0/{n_temp})')
            for k, d in enumerate(self.temp_directories, start=1):
                try:
                    shutil.rmtree(d)
                except Exception:
                    logger.exception('Error removing temp directory %s during shutdown', d)
                emit('shutdown-temp', k, n_temp, f'Removing temporary project files ({k}/{n_temp})')

        emit('shutdown-dask', 0, 1, 'Closing Dask cluster')
        try:
            self.dask_cluster.close()
        except Exception:
            logger.exception('Error closing Dask cluster during shutdown')
        emit('shutdown-dask', 1, 1, 'Cluster shut down')


def _unwrap_or_raise(results):
    '''Convert a list of wrapper-produced dicts to the legacy shape, or raise.

    Empty input returns []. Non-dict elements will raise AttributeError on
    .get(); the caller is responsible for only passing wrapper output.
    '''
    failures = [r for r in results if r.get('status') != 'ok']
    if failures:
        raise WorkerDied(failures)
    # Legacy shape: [( (port, directory), value ), ...]
    return [((r['port'], r['directory']), r['value']) for r in results]


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Start a Veneer cluster for parallel Source model execution'
    )
    parser.add_argument('project', help='Path to the Source project file (.rsproj)')
    parser.add_argument('-e', '--veneer-exe', required=True,
                        help='Path to FlowMatters.Source.VeneerCmd.exe')
    parser.add_argument('-n', '--n-workers', type=int, default=None,
                        help='Number of worker instances (default: number of CPUs)')
    parser.add_argument('-p', '--port', type=int, default=9876,
                        help='Starting port number (default: 9876)')
    parser.add_argument('--copy', action='store_true',
                        help='Copy the project file for each instance')
    parser.add_argument('--copy-extras', nargs='*', default=[],
                        help='Additional files/directories to copy with the project')
    parser.add_argument('--additional-plugins', nargs='*', default=[],
                        help='Additional plugin DLLs to load')
    parser.add_argument('--custom-endpoints', nargs='*', default=[],
                        help='Custom endpoints to add')
    parser.add_argument('--remote', action='store_true',
                        help='Allow remote connections')
    parser.add_argument('--no-script', action='store_true',
                        help='Disable IronPython scripting')
    parser.add_argument('--overwrite-plugins', action='store_true', default=None,
                        help='Overwrite Source plugin configuration')
    parser.add_argument('--existing', choices=['raise', 'remove', 'ignore'], default='raise',
                        help='Behaviour when existing temp directories are found (default: raise)')
    parser.add_argument('--save', default=None,
                        help='Save cluster config to a JSON file for later reconnection')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug output from Veneer instances during startup')
    parser.add_argument('--no-cleanup-on-failure', dest='cleanup_on_failure',
                        action='store_false', default=True,
                        help='Leave partially-started resources running if startup fails (for investigation)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cluster = VeneerCluster(
        project_file=args.project,
        veneer_exe=args.veneer_exe,
        n_workers=args.n_workers,
        debug=args.debug,
        remote=args.remote,
        script=not args.no_script,
        overwrite_plugins=args.overwrite_plugins,
        additional_plugins=args.additional_plugins,
        custom_endpoints=args.custom_endpoints,
        copy=args.copy,
        copy_extras=args.copy_extras,
        existing=args.existing,
        cleanup_on_failure=args.cleanup_on_failure,
    )

    if args.save:
        cluster.to_json(args.save)
        print(f'Cluster config saved to {args.save}')

    print(f'Cluster started with {cluster.n_workers} workers on ports {cluster.veneer_ports}')
    print('Press Ctrl+C to shut down the cluster')
    try:
        import signal
        signal.signal(signal.SIGINT, signal.default_int_handler)
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nShutting down cluster...')
        cluster.shutdown()
        print('Cluster shut down.')

