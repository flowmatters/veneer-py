from subprocess import Popen, PIPE
from psutil import Process
from queue import Queue, Empty  # python 3.x
from threading  import Thread
import sys
from time import sleep
import atexit
import os
import tempfile
import shutil
from .general import Veneer
from pathlib import Path

# Non blocking IO solution from http://stackoverflow.com/a/4896288
ON_POSIX = 'posix' in sys.builtin_module_names

VENEER_EXE_FN='FlowMatters.Source.VeneerCmd.exe'
MANY_VENEERS='D:\\src\\projects\\Veneer\\Compiled'
VENEER_EXE='D:\\src\\projects\\Veneer\\Output\\FlowMatters.Source.VeneerCmd.exe'
IGNORE_LOGS=[
    'log4net:ERROR Failed to find configuration section'
]

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

def create_command_line(veneer_path,source_version="4.1.1",
                        source_path='C:\\Program Files\\eWater',
                        dest=None,
                        force=True,
                        init_db=False):
    '''
    Copy all Veneer related files and all files from the relevant Source distribution to a third directory,
    for use as the veneer command line.

    veneer_path: Directory containing the Veneer files

    source_version: Version of Source to locate and copy

    source_path: Base installation directory for eWater Source

    dest: Destination to copy Source and Veneer to. If not provided, a temporary directory will be created.

    force: (default True) copy all files even if the directory already exists

    init_db: (default False) if True AND if the command line was created, start an instance in order to
             initialise the database, then terminate.

    Returns: Full path to FlowMatters.Source.VeneerCmd.exe for use with start()

    Note: It is your responsibility to delete the copy of Source and Veneer from the destination directory
    when you are finished with it! (Even if a temporary directory is used!)
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
        #print(available,versions,chosen_one)
        chosen_one_full = [product_ver for product_ver in available if chosen_one in product_ver][0]
    else:
        chosen_one_full = source_path

    source_dir = Path(source_path)/chosen_one_full
    source_files = list(source_dir.glob('*.*'))
    if not len(source_files):
        raise Exception('Source files not found at %s'%source_path)

    veneer_files = list(Path(veneer_path).glob('*.*'))
    if not len(veneer_files):
        raise Exception('Veneer files not found at %s'%veneer_path)

    for f in source_files:
        if not f.is_file():
            continue
        shutil.copyfile(str(f),str(Path(dest)/_basename(f)))

    if Path(veneer_path)!=dest:
        for f in veneer_files:
            if not f.is_file():
                continue
            shutil.copyfile(str(f),str(Path(dest)/_basename(f)))

    extra_dirs = ['x86','x64']
    for e in extra_dirs:
        extra = source_dir / e
        if not extra.exists():
            continue
        extra_dest = Path(dest) / e
        if not extra_dest.exists():
            extra_dest.mkdir()
        for f in extra.glob('*.*'):
            shutil.copy(f,extra_dest/_basename(f))

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

def overwrite_plugin_configuration(source_binaries,project_fn):
    source_binaries = Path(source_binaries)
    project_fn = Path(project_fn)
    plugin_fn = _find_plugins_file(project_fn)
    if not plugin_fn:
        # logger.warn('Unable to overwrite plugins. No Plugin.xml found')
        return
    print(plugin_fn)

    if source_binaries.is_file():
        source_binaries = Path(_dirname(source_binaries))/'RiverSystem.Forms.exe'

    source_version = '.'.join([str(v) for v in _get_version_number(source_binaries)[00:-1]])
    print(source_version)

    plugin_dir = Path('C:')/'Users'/os.environ['USERNAME']/'AppData'/'Roaming'/'Source'/source_version
    print(plugin_dir)
    if not plugin_dir.exists():
        plugin_dir.mkdir(parents=True)
    plugin_dest_file = Path(plugin_dir)/'Plugins.xml'
    shutil.copyfile(str(plugin_fn),str(plugin_dest_file))
    assert plugin_dest_file.exists()

def start(project_fn=None,n_instances=1,ports=9876,debug=False,remote=True,
          script=True, veneer_exe=None,overwrite_plugins=None,return_io=False,
          model=None,start_new_session=False,additional_plugins=[]):
    """
    Start one or more copies of the Veneer command line progeram with a given project file

    Parameters:

    - project_fn - Path to a Source project file (.rsproj)

    - n_instances - Number of copies of the Veneer command line to start (default: 1)

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

    - additional_plugins - List of plugin files (DLLs) to load

    returns processes, ports
       processes - list of process objects that can be used to terminate the servers
       ports - the port numbers used for each copy of the server
       ((stdout_queues,stdout_threads),(stderr_queues,stderr_threads)) if return_io
    """
    if not hasattr(ports,'__len__'):
        ports = list(range(ports,ports+n_instances))

    if not veneer_exe:
        veneer_exe = find_veneer_cmd_line_exe(project_fn)

    if project_fn:
        project_fn = Path(project_fn).absolute()
        if overwrite_plugins:
            overwrite_plugin_configuration(veneer_exe,project_fn)

    extras = ''
    if remote: extras += '-r '
    if script: extras += '-s '
    if model is not None:
        if isinstance(model,int) and model <= 0:
            raise Exception('Model index must be 1-based')

        model = str(model)
        if ' ' in model:
            model = '"%s"'%model

        extras += '-m %s '%model

    if len(additional_plugins):
        extras += '-l '+ ','.join(additional_plugins)

    cmd_line = '%s -p %%d %s '%(veneer_exe,extras)
    if project_fn:
        cmd_line += '"%s"'%str(project_fn)
    cmd_lines = [cmd_line%port for port in ports]
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
    processes = [Popen(cmd_line%port,stdout=PIPE,stderr=PIPE,bufsize=1, close_fds=ON_POSIX, **kwargs) for port in ports]
    std_out_queues,std_out_threads = configure_non_blocking_io(processes,'stdout')
    std_err_queues,std_err_threads = configure_non_blocking_io(processes,'stderr')

    ready = [False for p in range(n_instances)]
    failed = [False for p in range(n_instances)]

    all_ready = False
    any_failed = False
    actual_ports = ports[:]
    while not (all_ready or any_failed):       
        for i in range(n_instances):
            if not processes[i].poll() is None:
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
    #               print('.')
    #               sys.stdout.flush()
                else:
                    if 'Started Source RESTful Service on port' in line:
                        actual_ports[i] = int(line.split(':')[-1])

                    if line.startswith('Server started. Ctrl-C to exit'):
                        ready[i] = True
                        end = True
                        print('Server %d on port %d is ready'%(i,ports[i]))
                        sys.stdout.flush()
                    elif line.startswith('Cannot find project') or line.startswith('Unhandled exception'):
                        end = True
                        failed[i] = True
                        print('Server %d on port %d failed to start'%(i,ports[i]))
                        sys.stdout.flush()
                    if debug and not line.startswith('Unable to delete'):
                        print("[%d] %s"%(i,line))
                        sys.stdout.flush()
        all_ready = len([r for r in ready if not r])==0
        any_failed = len([f for f in failed if f])>0

    if any_failed:
        for p in processes:
            p.kill()
        raise Exception('One or more instances of Veneer failed to start. Try again with debug=True to see output')

    kill_all_on_exit(processes)

    if return_io:
        return processes,actual_ports,((std_out_queues,std_out_threads),(std_err_queues,std_err_threads))
    return processes,actual_ports

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
