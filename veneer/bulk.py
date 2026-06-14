
try:
    from urllib import quote
except:
    from urllib.parse import quote
import argparse
import requests
import json
import shutil
import os
import sys
from glob import glob
from veneer.general import MODEL_TABLES
from veneer.actions import get_big_data_source

STANDARD_RASTERS=[('DEM',False),('FunctionalUnitMap',True),('StreamMap',False)]


def _long_path(path):
    '''
    On Windows, rewrite ``path`` to its extended-length (``\\\\?\\``) form so
    paths longer than the legacy 260-character MAX_PATH limit can be created
    and written. Source models can generate very long network/variable names,
    which push archived time-series paths past that limit. No-op elsewhere.
    '''
    if os.name != 'nt':
        return path
    abs_path = os.path.abspath(path)  # also normalises forward slashes to '\\'
    if abs_path.startswith('\\\\?\\'):
        return abs_path
    if abs_path.startswith('\\\\'):
        # UNC path: \\server\share -> \\?\UNC\server\share
        return '\\\\?\\UNC\\' + abs_path[2:]
    return '\\\\?\\' + abs_path

class VeneerRetriever(object):
    '''
    Retrieve all information from a Veneer web service and write it out to disk in the same path structure.

    Typically used for creating/archiving static dashboards from an existing Veneer web application.
    '''
    def __init__(self,destination,port=9876,host='localhost',protocol='http',
                 retrieve_daily=True,retrieve_monthly=True,retrieve_annual=True,
                 retrieve_slim_ts=True,retrieve_single_ts=True,
                 retrieve_single_runs=True,retrieve_daily_for=[],
                 retrieve_ts_json=True,retrieve_ts_csv=False,
                 retrieve_data_sources=True,retrieve_spatial=False,
                 print_all = False, print_urls = False,
                 update_frequency = -1, logger=None,
                 trust_env=None, proxies=None):
        from .general import Veneer,log
        self.destination = destination
        self.port = port
        self.host = host
        self.protocol = protocol
        self.retrieve_daily = retrieve_daily
        self.retrieve_monthly = retrieve_monthly
        self.retrieve_annual = retrieve_annual
        self.retrieve_slim_ts = retrieve_slim_ts
        self.retrieve_single_ts = retrieve_single_ts
        self.retrieve_single_runs = retrieve_single_runs
        self.retrieve_daily_for = retrieve_daily_for
        self.retrieve_ts_json=retrieve_ts_json
        self.retrieve_ts_csv=retrieve_ts_csv
        self.retrieve_data_sources=retrieve_data_sources
        self.retrieve_spatial=retrieve_spatial
        self.print_all = print_all
        self.print_urls = print_urls
        self.base_url = "%s://%s:%d" % (protocol,host,port)
        self._veneer = Veneer(host=self.host,port=self.port,protocol=self.protocol,
                              trust_env=trust_env, proxies=proxies)
        self.log = logger or log
        self.last_update_at = 0
        self.update_frequency = update_frequency

    def mkdirs(self,directory):
        directory = _long_path(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def make_dest(self,base_name,ext):
        import os
        full_name = os.path.join(self.destination,base_name + "."+ext)
        directory = os.path.dirname(full_name)
        self.mkdirs(directory)
        return full_name

    def save_data(self,base_name,data,ext,mode="b"):
        full_name = self.make_dest(base_name,ext)
        # Use the extended-length path for the actual write so over-long names
        # don't fail, and log+skip (rather than aborting the whole archive) on
        # any other filesystem error, mirroring retrieve_json/retrieve_csv.
        try:
            with open(_long_path(full_name),"w"+mode) as f:
                f.write(data)
        except OSError as e:
            self.log("Couldn't write %s: %s"%(full_name, str(e)))
    
    def retrieve_json(self,url,**kwargs):
        if self.print_urls:
            print("*** %s ***" % (url))
    
        try:
            response = self._veneer._session.get(self.base_url + quote(url))
            response.raise_for_status()
            text = response.text
        except Exception as e:
            self.log("Couldn't retrieve %s: %s"%(url, str(e)))
            return None

        self.save_data(url[1:],bytes(text,'utf-8'),"json")
    
        if self.print_all:
            print(json.loads(text))
            print("")
        return json.loads(text)
    
    def retrieve_csv(self,url):
        # Resilient like retrieve_json: a missing/unsupported table (e.g. the
        # 'fus' table on a model with no functional units) is logged and skipped
        # rather than aborting the whole retrieval.
        try:
            text = self._veneer.retrieve_csv(url)
        except Exception as e:
            self.log("Couldn't retrieve %s: %s"%(url, str(e)))
            return None
        self.save_data(url[1:],bytes(text,'utf-8'),"csv")

    def retrieve_resource(self,url,ext,optional=False):
        if self.print_urls:
            print("*** %s ***" % (url))

        try:
            response = self._veneer._session.get(self.base_url+quote(url))
            response.raise_for_status()
        except Exception as e:
            if optional:
                self.log("Couldn't retrieve %s: %s"%(url, str(e)))
                return None
            raise
        self.save_data(url[1:],response.content,ext,mode="b")

    # Process Run list and results
    def retrieve_runs(self):
        current_progress = 5
        total_progress = 85
        run_list = self.retrieve_json("/runs")
        all_results = []
        for ix,run in enumerate(run_list):
            prog_per_run = total_progress / len(run_list)

            run_results = self.retrieve_json(run['RunUrl'])
            ts_results = run_results['Results']
            all_results += ts_results

            if not self.retrieve_single_runs:
                continue

            if self.retrieve_single_ts:
                for jx,result in enumerate(ts_results):
                    prog_per_ts = prog_per_run / len(ts_results)
                    self.retrieve_ts(result['TimeSeriesUrl'])
                    self.update('Downloaded %d/%d results in run %d'%(jx,len(ts_results),ix),current_progress+jx*prog_per_ts)

            if self.retrieve_slim_ts:
                self.retrieve_multi_ts(ts_results)

            current_progress += prog_per_run
            self.update('Downloaded %d/%d runs'%(ix,len(run_list)),current_progress)

        if self.retrieve_slim_ts and len(run_list):
            all_results = self.unique_results_across_runs(all_results)
            self.retrieve_multi_ts(all_results,run="__all__")
            self.retrieve_across_runs(all_results)

    def unique_results_across_runs(self,all_results):
        result = {}
        for ts in all_results:
            generic_url = self.translate_url(ts['TimeSeriesUrl'],run='__all__')
            if not generic_url in result:
                result[generic_url] = ts
        return result.values()

    def translate_url(self,orig,run=None,loc=None,elem=None,var=None):
        url = orig.split('/')
        if not run is None:
            url[2] = run
        if not loc is None:
            url[4] = loc
        if not elem is None:
            url[6] = elem
        if not var is None:
            url[8] = var
        return '/'.join(url)

    def retrieve_multi_ts(self,ts_results,run=None):
        recorders = list(set([(r['RecordingElement'],r['RecordingVariable']) for r in ts_results]))
        for r in recorders:
            for option in ts_results:
                if option['RecordingElement'] == r[0] and option['RecordingVariable'] == r[1]:
                    url = self.translate_url(option['TimeSeriesUrl'],run=run,loc='__all__')
                    self.retrieve_ts(url)
                    break

    def retrieve_across_runs(self,results_set):
        for option in results_set:
            url = self.translate_url(option['TimeSeriesUrl'],run='__all__')
            self.retrieve_ts(url)

    def retrieve_this_daily(self,ts_url):
        if self.retrieve_daily: return True

        splits = ts_url.split('/')
        run = splits[2]
        loc = splits[4]
        ele = splits[6]
        var = splits[8]

        for exception in self.retrieve_daily_for:
            matched = True
            for key,val in exception.items():
                if key=='NetworkElement' and val!=loc: 
                    matched=False;
                    break
                if key=='RecordingElement' and val!=ele:
                    matched=False;
                    break
                if key=='RecordingVariable' and val!=var:
                    matched=False;
                    break
            if matched: return True
        return False

    def retrieve_ts(self,ts_url):
        urls = []

        if self.retrieve_this_daily(ts_url):
            urls.append(ts_url)
        if self.retrieve_monthly:
            urls.append(ts_url + "/aggregated/monthly")
        if self.retrieve_annual:
            urls.append(ts_url + "/aggregated/annual")

        for url in urls:
            if self.retrieve_ts_json:
                self.retrieve_json(url)
            if self.retrieve_ts_csv:
                self.retrieve_csv(url)
    
    def retrieve_variables(self):
        variables = self.retrieve_json("/variables")
        for var in variables:
            if var['TimeSeries']: self.retrieve_json(var['TimeSeries'])
            if var['PiecewiseFunction']: self.retrieve_json(var['PiecewiseFunction'])

    def retrieve_data_sources_values(self):
        data_sources = self._veneer.data_sources()
        for ds_spec in data_sources:
            ds_name = ds_spec['Name']
            try:
                ds = get_big_data_source(self._veneer,ds_name,data_sources)
            except Exception as e:
                self.log(f'Unable to retrieve datasource: {ds_name}. Skipping')
                self.log(str(e))
                continue
            fn = self.make_dest(f'dataSources/{ds_name}','csv')
            ds.to_csv(fn)

    def retrieve_tables(self):
        # Discover model tables dynamically from the /tables index endpoint.
        # Older Veneer instances lack this endpoint (retrieve_json returns None),
        # in which case fall back to the known MODEL_TABLES. When the endpoint IS
        # present we trust its (possibly empty) table list: a model with no
        # functional units legitimately reports zero tables and must not be
        # forced to fetch the non-existent 'fus' table.
        index = self.retrieve_json('/tables')
        if index is None:
            table_names = MODEL_TABLES
        else:
            table_names = [tbl['Name'] for tbl in index.get('Tables', [])]
        for tbl in table_names:
            self.retrieve_csv('/tables/%s'%tbl)

    def retrieve_network(self):
        network = self.retrieve_json("/network")
        icons_retrieved = []
        for f in network['features']:
            #retrieve_json(f['id'])
            if not f['properties']['feature_type'] == 'node': continue
            if f['properties']['icon'] in icons_retrieved: continue
            self.retrieve_resource(f['properties']['icon'],'png')
            icons_retrieved.append(f['properties']['icon'])

        # Newer Veneer endpoints. These degrade gracefully on older instances:
        # retrieve_json swallows 404s, and the schematic SVG is fetched as an
        # optional resource (absent when the scenario has no schematic).
        self.retrieve_json("/network/geographic")
        self.retrieve_resource("/network/schematic.svg","svg",optional=True)
        self.retrieve_json("/network/schematic.svg/tags")

    def retrieve_spatial_data(self):
        # List of rasters. Check which are present and write. Check which have attribute tables and write
        # List of vectors. Write... (columns?)

        for r,categories in STANDARD_RASTERS:
            if not self._veneer.model.spatial.has_raster(r):
                continue

            self._veneer.model.spatial.save_raster(r,self.make_dest(f'spatial/{r}','asc'),categories)

    def retrieve_all(self,clean=False):
        if os.path.exists(self.destination):
            if clean:
                shutil.rmtree(self.destination)
            else:
                raise Exception("Destination (%s) already exists. Use clean=True to overwrite"%self.destination)
        self.mkdirs(self.destination)

        self.retrieve_tables()

        if self.retrieve_data_sources:
            self.retrieve_data_sources_values()

        if self.retrieve_spatial:
            self.retrieve_spatial_data()

        self.update('Downloading runs',5)
        self.retrieve_runs()
        self.update('Downloaded runs',90)

        self.retrieve_json("/functions")
        self.retrieve_variables()
        self.retrieve_json("/inputSets")
        self.retrieve_json("/")

        self.retrieve_network()
        self.update('Finished ',100)
    
    def update(self,msg,prog):
        if self.update_frequency < 0:
            return
        if (prog >= 100) or (self.last_update_at + self.update_frequency < prog):
            self.log(msg)
            self.last_update_at = prog

class PruneVeneer(object):
    def __init__(self,path,dry_run=False):
        self.path = path
        self.removals = []

    def remove_variable(self,v,daily=True,aggregate=True):
        self.removals.append(({'RecordingVariable':v},dict(daily=daily,aggregate=aggregate)))

    def remove_element(self,e,daily=True,aggregate=True):
        self.removals.append(({'RecordingElement':e},dict(daily=daily,aggregate=aggregate)))

    def glob(self,f,recursive=False):
        fn = os.path.join(self.path,f)
        return glob(fn,recursive=recursive)

    def prune(self):
        files_to_remove = []
        files_to_deindex = []
        ts_template = 'runs/*/location/%s/element/%s/variable/%s'
        for r,opt in self.removals:
            loc=r.get('NetworkElement','*')
            ele=r.get('RecordingElement','*')
            var=r.get('RecordingVariable','*')
            search = ts_template%(loc,ele,var)
            #print(search)
            if opt['daily']:
                matches = self.glob(search+'.json')
                files_to_remove += matches
                if opt['aggregate']:
                    files_to_deindex += matches
            if opt['aggregate']:
                files_to_remove += self.glob(search+'/**',recursive=True)

        print('Found %d files and folders to remove'%len(files_to_remove))
        print('Found %d time series to de-index'%len(files_to_deindex))
        print('Cleaning up run files')
        self.clean_up_results_files(files_to_deindex)
        print('Removing time series files')
        for f in files_to_remove:
            if os.path.isfile(f):
                os.remove(f)
        print('Pruning empty directories')
        os.system('find %s -type d -empty -delete'%self.path)

    def clean_up_results_files(self,files):
        files_per_run = {}
        for fn in files:
            relative = fn[len(self.path):]
            run = int(relative.split('/')[2])
            if not run in files_per_run:
                files_per_run[run] = []
            files_per_run[run].append(relative[:-5])
        for run,files in files_per_run.items():
            self.clean_up_run(run,files)
        
    def clean_up_run(self,run_number,files):
        run_fn = os.path.join(self.path,'runs','%d.json'%run_number)
        run = json.load(open(run_fn))
        len_before = len(run['Results'])
        run['Results'] = [res for res in run['Results'] if not res['TimeSeriesUrl'] in files]
        
        json.dump(run,open(run_fn,'w'))


DEFAULT_DESTINATION = 'C:\\temp\\veneer_download'


def build_arg_parser():
    '''
    Build the command line argument parser for the bulk retriever.

    Every overridable option defaults to None so that, after parsing, we can
    tell whether the user actually supplied a flag. Resolution order for each
    setting is: explicit CLI flag -> value from --config JSON -> built-in default.
    '''
    bool_action = argparse.BooleanOptionalAction
    parser = argparse.ArgumentParser(
        prog='python -m veneer.bulk',
        description='Bulk-retrieve all data from a running Veneer instance and '
                    'write it to disk (time series, tables, data sources, '
                    'network and optionally spatial data).')

    parser.add_argument('destination', nargs='?', default=None,
                        help='Output directory for the retrieved data '
                             '(default: %s, or "destination" from --config).' % DEFAULT_DESTINATION)
    parser.add_argument('--config', metavar='PATH', default=None,
                        help='JSON config file providing default settings. '
                             'Explicit CLI flags override values from this file. '
                             'Use the config for structured/niche settings such '
                             'as slack, proxies, retrieve_daily_for and print flags.')

    parser.add_argument('--host', default=None,
                        help='Veneer host (default: localhost).')
    parser.add_argument('--port', type=int, default=None,
                        help='Veneer port (default: 9876).')
    parser.add_argument('--protocol', default=None, choices=['http', 'https'],
                        help='Protocol (default: http).')
    parser.add_argument('--clean', action=bool_action, default=None,
                        help='Delete the destination directory first if it '
                             'already exists (default: off).')
    parser.add_argument('--update-frequency', dest='update_frequency', type=int, default=None,
                        help='Progress logging frequency as a percent step; '
                             'negative to silence progress (default: 5).')
    parser.add_argument('--run', action=bool_action, default=None,
                        help='Trigger a simulation run with default settings '
                             'before retrieving (default: off).')
    parser.add_argument('--drop-runs', dest='drop_runs', action=bool_action, default=None,
                        help='Drop all existing runs before triggering the run, '
                             'so the archive reflects only the fresh run. Only '
                             'applies with --run (default: off).')

    parser.add_argument('--daily', dest='retrieve_daily', action=bool_action, default=None,
                        help='Retrieve daily time series (default: on).')
    parser.add_argument('--monthly', dest='retrieve_monthly', action=bool_action, default=None,
                        help='Retrieve monthly-aggregated time series (default: on).')
    parser.add_argument('--annual', dest='retrieve_annual', action=bool_action, default=None,
                        help='Retrieve annually-aggregated time series (default: on).')
    parser.add_argument('--data-sources', dest='retrieve_data_sources', action=bool_action, default=None,
                        help='Retrieve data sources (default: on).')
    parser.add_argument('--spatial', dest='retrieve_spatial', action=bool_action, default=None,
                        help='Retrieve spatial/raster data (default: off).')
    parser.add_argument('--ts-json', dest='retrieve_ts_json', action=bool_action, default=None,
                        help='Write time series as JSON (default: on).')
    parser.add_argument('--ts-csv', dest='retrieve_ts_csv', action=bool_action, default=None,
                        help='Write time series as CSV (default: off).')

    return parser


# CLI flags whose resolved values are passed straight through to VeneerRetriever
# as keyword arguments (and may also appear in a config file's "options" dict).
_OPTION_FLAGS = [
    'host', 'protocol',
    'retrieve_daily', 'retrieve_monthly', 'retrieve_annual',
    'retrieve_data_sources', 'retrieve_spatial',
    'retrieve_ts_json', 'retrieve_ts_csv',
]


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    # Start from the config's "options" dict; CLI flags layer on top of it.
    options = dict(config.get('options', {}))

    port = args.port if args.port is not None else config.get('port', 9876)
    dest = args.destination if args.destination is not None \
        else config.get('destination', DEFAULT_DESTINATION)
    clean = args.clean if args.clean is not None else config.get('clean', False)
    run = args.run if args.run is not None else config.get('run', False)
    drop_runs = args.drop_runs if args.drop_runs is not None else config.get('drop_runs', False)

    # update_frequency historically defaulted to 5 in the CLI (not the
    # constructor's -1); preserve that. CLI flag > config options > 5.
    # Always pop it from options so it can't collide with the explicit kwarg.
    config_update_frequency = options.pop('update_frequency', 5)
    if args.update_frequency is not None:
        update_frequency = args.update_frequency
    else:
        update_frequency = config_update_frequency

    for flag in _OPTION_FLAGS:
        value = getattr(args, flag)
        if value is not None:
            options[flag] = value

    slack = config.get('slack', None)
    if slack is not None:
        import nbslack
        nbslack.notifying(slack)

    def notify(msg):
        print(msg)
        if slack is not None:
            import nbslack
            nbslack.notify(msg)

    vr = VeneerRetriever(dest, port, update_frequency=update_frequency, logger=notify, **options)

    if run:
        if drop_runs:
            notify('Dropping existing runs')
            vr._veneer.drop_all_runs()
        notify('Triggering simulation run with default settings')
        vr._veneer.run_model()
        notify('Simulation run complete')

    vr.retrieve_all(clean=clean)


if __name__ == '__main__':
    main()

