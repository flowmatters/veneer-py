
try:
    from urllib2 import urlopen, quote
except:
    from urllib.request import urlopen, quote, Request
import json
import shutil
import os
import sys
from glob import glob
from veneer.general import MODEL_TABLES
from veneer.actions import get_big_data_source

STANDARD_RASTERS=[('DEM',False),('FunctionalUnitMap',True),('StreamMap',False)]

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
                 update_frequency = -1, logger=None):
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
        self._veneer = Veneer(host=self.host,port=self.port,protocol=self.protocol)
        self.log = logger or log
        self.last_update_at = 0
        self.update_frequency = update_frequency

    def mkdirs(self,directory):
        import os
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
        f = open(full_name,"w"+mode)
        f.write(data)
        f.close()
    
    def retrieve_json(self,url,**kwargs):
        if self.print_urls:
            print("*** %s ***" % (url))
    
        try:
            text = urlopen(self.base_url + quote(url)).read().decode('utf-8')
        except:
            self.log("Couldn't retrieve %s"%url)
            return None

        self.save_data(url[1:],bytes(text,'utf-8'),"json")
    
        if self.print_all:
            print(json.loads(text))
            print("")
        return json.loads(text)
    
    def retrieve_csv(self,url):
        text = self._veneer.retrieve_csv(url)
        self.save_data(url[1:],bytes(text,'utf-8'),"csv")

    def retrieve_resource(self,url,ext):
        if self.print_urls:
            print("*** %s ***" % (url))
    
        self.save_data(url[1:],urlopen(self.base_url+quote(url)).read(),ext,mode="b")

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

        for tbl in MODEL_TABLES:
            self.retrieve_csv('/tables/%s'%tbl)

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

        network = self.retrieve_json("/network")
        icons_retrieved = []
        for f in network['features']:
            #retrieve_json(f['id'])
            if not f['properties']['feature_type'] == 'node': continue
            if f['properties']['icon'] in icons_retrieved: continue
            self.retrieve_resource(f['properties']['icon'],'png')
            icons_retrieved.append(f['properties']['icon'])
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


if __name__ == '__main__':
    if len(sys.argv)>1:
        config_fn = sys.argv[1]
        config = json.load(open(config_fn,'r'))
    else:
        config = {}

    port = config.get('port',9876)
    slack = config.get('slack',None)

    if slack is not None:
        import nbslack
        nbslack.notifying(slack)

    def notify(msg):
        print(msg)
        if slack is not None:
            import nbslack
            nbslack.notify(msg)
    
    options = config.get('options',{})
    dest = config.get('destination','C:\\temp\\veneer_download')
    clean = config.get('clean',False)

    vr = VeneerRetriever(dest,port,update_frequency=5,logger=notify,**options)
    vr.retrieve_all(clean=clean)

