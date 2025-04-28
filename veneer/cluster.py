import json
from .manage import start, kill_all_now
from dask.distributed import LocalCluster, Client
from psutil import Process
import dask
from .general import Veneer
import os
import tempfile
import shutil
from functools import partial
import logging
logger = logging.getLogger(__name__)

MAX_DEFAULT_CLUSTER_SIZE=64

@dask.delayed
def copy_project(prefix,project_file,extras):
    assert os.path.exists(project_file)

    tmp = tempfile.mkdtemp(prefix=prefix)
    assert os.path.exists(tmp)

    dest_fn = os.path.join(tmp,os.path.basename(project_file))
    shutil.copyfile(project_file,dest_fn)
    src_dir = os.path.dirname(project_file)
    for e in extras:
        source = os.path.abspath(os.path.join(src_dir,e))
        assert os.path.exists(source)
        dest = os.path.abspath(os.path.join(tmp,e))
        if os.path.isdir(source):
            shutil.copytree(source,dest)
        else:
            shutil.copyfile(source,dest)
    return tmp

@dask.delayed
def remove_copy(directory):
    shutil.rmtree(directory)

@dask.delayed
def scenario_info(port):
    v = Veneer(port)
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
    @dask.delayed
    def inner_wrapped(*args,**kwargs):
        import os
        import veneer
        worker_pid = os.getpid()
        port, directory = worker_map[worker_pid]
        inner_args = {
            'v': veneer.Veneer(port)
        }
        if copy_projects:
            inner_args['directory'] = directory
        return worker_pid, fn(*args,**kwargs,**inner_args)

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
    def __init__(self,cluster):
        self._cluster = cluster
        self._dummy_v = Veneer(0)

    def __getattr__(self,attrname):
        def wrapped(*args,**kwargs):
            @dask.delayed
            def fn(p):
                v = Veneer(p)
                method = getattr(v,attrname)
                return method(*args,**kwargs)
            return self._cluster.run_on_each(fn)
        return wrapped
    def __dir__(self):
        return self._dummy_v.__dir__()

class VeneerCluster(object):
    def __init__(self,project_file=None,veneer_exe=None,n_workers=None,debug=False,remote=False,
          script=True,overwrite_plugins=None,custom_endpoints=None,additional_plugins=None,
          copy=False,tempdir_prefix='source-cluster-',copy_extras=None,existing='raise',
          existing_cluster=None):
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
        '''

        self.wrap = partial(run_on_cluster,self)
        self.v = ClusterVeneerClient(self)

        if existing_cluster is not None:
            if os.path.exists(existing_cluster):
                logger.info('Using existing cluster at %s',existing_cluster)
                with open(existing_cluster,'r') as f:
                    existing_cluster = json.load(f)
            elif isinstance(existing_cluster,dict):
                logger.info('Using existing cluster with config')
            else:
                raise Exception('Invalid existing cluster: %s'%existing_cluster)

            self.dask_client = Client(existing_cluster['dask_scheduler'])
            self.dask_cluster = self.dask_client.cluster

            self.original_project_file = existing_cluster['project_file']
            self.name = existing_cluster['name']
            self.n_workers = existing_cluster['n_workers']
            self.veneer_ports = existing_cluster['veneer_ports']
            self.veneer_processes = [Process(p) for p in existing_cluster['veneer_processes']]
            self.temp_directories = existing_cluster['temp_directories']
            self.worker_affinity = {int(k):v for k,v in existing_cluster['worker_affinity'].items()}
            self.copy_projects = existing_cluster['copy_projects']
            return

        check_existing_cluster_temp_directory(tempdir_prefix,existing)
        self.original_project_file = project_file
        self.name = f'{n_workers} node cluster for {project_file}'

        if n_workers is None:
            n_workers = min(os.cpu_count(),MAX_DEFAULT_CLUSTER_SIZE)
        self.n_workers = n_workers or os.cpu_count()

        logger.info('Initialising DASK cluster with %d workers',self.n_workers)
        self.dask_cluster = LocalCluster(threads_per_worker=1,n_workers=self.n_workers)
        self.dask_client = Client(self.dask_cluster)

        if not copy and (copy_extras is not None) and len(copy_extras):
            logger.info('Copying project file for each instance to copy extras')
            copy = True
        self.copy_projects = copy

        if copy:
            logger.info('Creating %d copies of project file %s',n_workers,project_file)
            creation = [copy_project(tempdir_prefix,project_file,copy_extras or []) for _ in range(n_workers)]
            self.temp_directories = dask.compute(*creation)
            self.project_files = [os.path.join(t,os.path.basename(self.original_project_file)) for t in self.temp_directories]
        else:
            self.temp_directories = []
            self.project_files = [self.original_project_file] * self.n_workers

        logger.info('Starting %d Veneer instances',self.n_workers)
        veneer_processes, veneer_ports = start(
            n_instances=self.n_workers,debug=debug,remote=remote,
            script=script, veneer_exe=veneer_exe,overwrite_plugins=overwrite_plugins,
            additional_plugins=additional_plugins or [],custom_endpoints=custom_endpoints or [],
            projects=self.project_files
        )
        self.veneer_ports = veneer_ports
        self.veneer_processes = veneer_processes

        logger.info('Assigning Veneer instances to DASK workers')
        self.worker_affinity = {}
        worker_info = self.dask_cluster.workers
        veneer_info_map = self.run_on_each(scenario_info)

        self.worker_affinity = {w.pid:(info['port'],os.path.dirname(info['ProjectFullFilename'])) for w,info in zip(worker_info.values(),veneer_info_map)}

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
            'worker_affinity': self.worker_affinity,
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

    def run_jobs(self,jobs):
        '''
        Run a set of jobs on the cluster.

        jobs: list
            A list of tuples, each containing a dask delayed job. Note that the job need not involve Source/Veneer,
            but if veneer is required, the original function should be wrapped with self.wrap

        Returns a list of tuples, each containing the worker port, the temporary directory and the result of the job
        '''
        results = self.dask_client.compute(jobs,sync=True)
        workers = [self.worker_affinity[pid] for pid,_ in results]
        return list(zip(workers,[r for _,r in results]))

    def shutdown(self):
        '''
        Shutdown the cluster. Remove any temporary directories created for the project files
        '''
        kill_all_now(self.veneer_processes)

        if len(self.temp_directories):
            removal = [remove_copy(c) for c in self.temp_directories]
            _ = dask.compute(*removal,scheduler=self.dask_client.scheduler)

        self.dask_cluster.close()

