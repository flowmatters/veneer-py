try:
    from urllib2 import quote
    import httplib as hc
except:
    from urllib.request import quote
    import http.client as hc

import json
import re
from .bulk import VeneerRetriever
from .server_side import VeneerIronPython
from .utils import SearchableList,_stringToList,read_veneer_csv,objdict
import pandas as pd
# Source
from . import extensions

PRINT_URLS=False
PRINT_ALL=False
PRINT_SCRIPTS=False

def name_time_series(result):
    '''
    Name the retrieved time series based on the full name of the time series (including variable and location)
    '''
    return result.get('TimeSeriesName',result.get('Name','%s/%s/%s'%(result['NetworkElement'],result['RecordingElement'],result['RecordingVariable'])))

def name_element_variable(result):
    element = result['NetworkElement']
    variable = result['RecordingVariable'].split(' - ')[-1]
    if variable=='Flow':
        variable = result['RecordingElement'].split(' - ')[-1]
    return '%s:%s'%(element,variable)

def name_for_variable(result):
    '''
    Name the retrieved time series based on the variable only.

    Useful when retrieving multiple time series from one network location.
    '''
    return result['RecordingVariable']

def name_for_end_variable(result):
    '''
    Name the retrieved time series based on the final part of the RecordingVariable (eg after the last @)
    '''
    return name_for_variable(result).split('@')[-1]

def name_for_location(result):
    '''
    Name the retrieved time series based on the network location only.

    Useful when retrieving the same variable from multiple locations.
    '''
    return result['NetworkElement']

def name_for_fu_and_sc(result):
    '''
    For a FU-based time series, name based on the FU and the subcatchment.

    Note, when retrieving FU based results, you should limit the query to a single FU (or 'Total').
    Due to a quirk in the Source recording system, you'll get the results for all FUs anyway.
    If you don't specify a single FU, the system will make separate requests for each FU and get multiple
    results from each requests (essentially transferring n^2 data).
    '''
    char = ':'
    if not 'TimeSeriesName' in result:
        char = '/'
    return ':'.join(name_time_series(result).split(':')[:2])
    

def name_for_fu(result):
    '''
    For a FU-based time series, name based on the FU.

    Note, when retrieving FU based results, you should limit the query to a single FU (or 'Total').
    Due to a quirk in the Source recording system, you'll get the results for all FUs anyway.
    If you don't specify a single FU, the system will make separate requests for each FU and get multiple
    results from each requests (essentially transferring n^2 data).
    '''
    return name_for_fu_and_sc(result).split(':')[0]

def log(text):
    import sys
    print('\n'.join(_stringToList(text)))
    sys.stdout.flush()

def _veneer_url_safe_id_string(s):
    return s.replace('#','').replace('/','%2F').replace(':','')

class Veneer(object):
    '''
    Acts as a high level client to the Veneer web service within eWater Source.
    '''
    def __init__(self,port=9876,host='localhost',protocol='http',prefix='',live=True):
        '''
        Instantiate a new Veneer client.

        Parameters:

        port, host, protocol: Connection information for running Veneer service (default 9876, localhost, http)

        prefix: path prefix for all queries. Useful if Veneer is running behind some kind of proxy

        live: Connecting to a live Veneer service or a statically served copy of the results? Default: True
        '''
        self.port=port
        self.host=host
        self.protocol=protocol
        self.prefix=prefix
        self.base_url = "%s://%s:%d%s" % (protocol,host,port,prefix)
        self.live_source=live
        if self.live_source:
            self.data_ext=''
        else:
            if protocol=='file':
                self.base_url = '%s://%s'%(protocol,prefix)
            self.data_ext='.json'
        self.model = VeneerIronPython(self)

    def shutdown(self):
        '''
        Stop the Veneer server (and shutdown the command line if applicable)
        '''
        try:
            self.post_json('/shutdown')
        except ConnectionResetError:
            return
        raise Exception("Connection didn't reset. Shutdown may not have worked")

    def _replace_inf(self,text):
        return re.sub('":(-?)INF','":\\1Infinity',text)

    def retrieve_json(self,url):
        '''
        Retrieve data from the Veneer service at the given url path.

        url: Path to required resource, relative to the root of the Veneer service.
        '''
        query_url = self.prefix+url+self.data_ext
        if PRINT_URLS:
            print("*** %s - %s ***" % (url, query_url))
        if self.protocol=='file':
            text = open(query_url).read()
        else:
            conn = hc.HTTPConnection(self.host,port=self.port)
            conn.request('GET',quote(query_url))
            resp = conn.getresponse()
            text = resp.read().decode('utf-8')

        text = self._replace_inf(text)
        if PRINT_ALL:
            print(json.loads(text))
            print("")
        try:
            return json.loads(text)
        except Exception as e:
            raise Exception('Error parsing response as JSON. Retrieving %s and received:\n%s'%(url,text[:100]))

    def retrieve_csv(self,url):
        '''
        Retrieve data from the Veneer service, at the given url path, in CSV format.

        url: Path to required resource, relative to the root of the Veneer service.

        NOTE: CSV responses are currently only available for time series results
        '''
        if PRINT_URLS:
            print("*** %s ***" % (url))

        conn = hc.HTTPConnection(self.host,port=self.port)
        conn.request('GET',quote(url+self.data_ext),headers={"Accept":"text/csv"})
        resp = conn.getresponse()
        text = resp.read().decode('utf-8')

        result = read_veneer_csv(text)
        if PRINT_ALL:
            print(result)
            print("")
        return result

    def update_json(self,url,data,async=False):
        '''
        Issue a PUT request to the Veneer service to update the data held at url

        url: Path to required resource, relative to the root of the Veneer service.

        data: Data to update.

        NOTE: This method will typically be used internally, by other Veneer methods.
        Usually, you will want to call one of these other methods to update something specific.
        For example, configure_recording to enable and disable time series recorders in the model.
        '''
        return self.send_json(url,data,'PUT',async)

    def send_json(self,url,data,method,async=False):
        payload = json.dumps(data)
        headers={'Content-type':'application/json','Accept':'application/json'}
        return self.send(url,method,payload,headers,async)

    def post_json(self,url,data=None,async=False):
        return self.send_json(url,data,'POST',async)

    def send(self,url,method,payload=None,headers={},async=False):
        conn = hc.HTTPConnection(self.host,port=self.port)
        conn.request(method,url,payload,headers=headers)
        if async:
            return conn
        resp = conn.getresponse()
        code = resp.getcode()
        if code==302:
            return code,resp.getheader('Location')
        elif code==200:
            resp_body = resp.read().decode('utf-8')
            return code,(json.loads(resp_body) if len(resp_body) else None)
        else:
            return code,resp.read().decode('utf-8')

        return conn

    def run_server_side_script(self,script,async=False):
        '''
        Run an IronPython script within Source.

        Requires Veneer to be running with 'Allow Scripts' option.

        script: the full text of an IronPython script to execute from within Source.

        NOTE: In many cases, it is possible (and desirable) to call helper methods within Veneer.model,
        rather than write your own IronPython script.
        '''
        if PRINT_SCRIPTS: print(script)
        result = self.post_json('/ironpython',{'Script':script},async=async)
        if async:
            return result
        code,data = result
        if code == 403:
            raise Exception('Script disabled. Enable scripting in Veneer')
        return data

    def configure_recording(self,enable=[],disable=[]):
        '''
        Enabled and disable time series recording in the Source model.

        enable: List of time series selectors to enable,

        disable: List of time series selectors to disable

        Note: Each time series selector is a python dictionary object with up to three keys:
          * NetworkElement
          * RecordingElement
          * RecordingVariable

        These are used to match time series available from the Source model. A given selector may match
        multiple time series. For example, a selector of {'RecordingVariable':'Downstream Flow Volume'}
        will match Downstream Flow Volume from all nodes and links.

        Any empty dictionary {} will match ALL time series in the model.

        So, for example, you could disable ALL recording in the model with

        v = Veneer()
        v.configure_recording(disable=[{}])

        Note, the time series selectors in enable and disable may both match the same time series in some cases.
        In this case, the 'enable' will take effect.
        '''
        def get_many(src,keys,default):
            return [src.get(k,default) for k in keys]

        def translate(rule):
            keys = ['NetworkElement','RecordingElement','RecordingVariable']
            vals = get_many(rule,keys,'')
            if vals[2]=='':vals[2]=vals[1]

            return 'location/%s/element/%s/variable/%s'%tuple(vals)

        modifier = {'RecordNone':[translate(r) for r in disable],
                    'RecordAll':[translate(r) for r in enable]}
        self.update_json('/recorders',modifier)

    def run_model(self,params=None,start=None,end=None,async=False,name=None,**kwargs):
        '''
        Trigger a run of the Source model

        params: Python dictionary of parameters to pass to Source. Should match the parameters expected
                of the running configuration. (If you just want to set the start and end date of the simulation,
                use the start and end parameters

        start, end: The start and end date of the simulation. Should be provided as Date objects or as text in the dd/mm/yyyy format

        async: (default False). If True, the method will return immediately rather than waiting for the simulation to finish.
               Useful for triggering parallel runs. Method will return a connection object that can then be queried to know
               when the run has finished.

        name: Name to assign to run in Source results (default None: let Source name using default strategy)

        kwargs: optional named parameters to be used to update the params dictionary

        In the default behaviour (async=False), this method will return once the Source simulation has finished, and will return
        the URL of the results set in the Veneer service
        '''
        conn = hc.HTTPConnection(self.host,port=self.port)

        if params is None:
            params = {}

        params.update(kwargs)

        if not start is None:
            params['StartDate'] = to_source_date(start)
        if not end is None:
            params['EndDate'] = to_source_date(end)

        if not name is None:
            params['_RunName'] = name

    #   conn.request('POST','/runs',json.dumps({'parameters':params}),headers={'Content-type':'application/json','Accept':'application/json'})
        conn.request('POST','/runs',json.dumps(params),headers={'Content-type':'application/json','Accept':'application/json'})
        if async:
            return conn

        resp = conn.getresponse()
        code = resp.getcode()
        if code==302:
            return code,resp.getheader('Location')
        elif code==200:
            return code,None
        elif code==500:
            error = json.loads(resp.read().decode('utf-8'))
            raise Exception('\n'.join([error['Message'],error['StackTrace']]))
        else:
            return code,resp.read().decode('utf-8')

    def drop_run(self,run='latest'):
        '''
        Tell Source to drop/delete a specific set of results from memory.

        run: Run number to delete. Default ='latest'. Valid values are 'latest' and integers from 1
        '''
        assert self.live_source
        conn = hc.HTTPConnection(self.host,port=self.port)
        conn.request('DELETE','/runs/%s'%str(run))
        resp = conn.getresponse()
        code = resp.getcode()
        return code

    def drop_all_runs(self):
        '''
        Tell Source to drop/delete ALL current run results from memory
        '''
        while len(self.retrieve_runs())>0:
            self.drop_run()

    def retrieve_runs(self):
        '''
        Retrieve the list of available runs.

        Individual runs can be used with retrieve_run to retrieve a summary of results
        '''
        return self.retrieve_json('/runs')

    def retrieve_run(self,run='latest'):
        '''
        Retrieve a results summary for a particular run.

        This will include references to all of the time series results available for the run.

        run: Run to retrieve. Either 'latest' (default) or an integer run number from 1
        '''
        run = run.split('/')[-1]
        if run=='latest' and not self.live_source:
            all_runs = self.retrieve_json('/runs')
            result = self.retrieve_json(all_runs[-1]['RunUrl'])
        else:
            result = self.retrieve_json('/runs/%s'%str(run))
        result['Results'] = SearchableList(result['Results'])
        return result

    def network(self):
        '''
        Retrieve the network from Veneer.

        The result will be a Python dictionary in GeoJSON conventions.

        The 'features' key of the returned dictionary will be a SearchableList, suitable for querying for
        different properties - eg to filter out just nodes, or links, or catchments.

        Example: Find all the node names in the current Source model

        v = Veneer()
        network = v.network()
        nodes = network['features'].find_by_feature_type('node')
        node_names = nodes._unique_values('name')
        '''
        result = objdict(self.retrieve_json('/network'))
        result['features'] = SearchableList(result['features'],['geometry','properties'])
        extensions.add_network_methods(result)
        return result

    def functions(self):
        '''
        Return a SearchableList of the functions in the Source model.
        '''
        return SearchableList(self.retrieve_json('/functions'))

    def update_function(self,fn,value):
        '''
        Update a function within Source

        fn: str, name of function to update.
        '''
        fn = fn.split('/')[-1]
        url = '/functions/' + fn.replace('$','')
        payload = {
            'Name':fn,
            'Expression':str(value)
        }
        return self.update_json(url,payload)

    def variables(self):
        '''
        Return a SearchableList of the function variables in the Source model
        '''
        return SearchableList(self.retrieve_json('/variables'))

    def variable(self,name):
        '''
        Returns details of a particular variable
        '''
        name = name.replace('$','')
        return self.retrieve_json('/variables/%s'%name)

    def variable_time_series(self,name):
        '''
        Returns time series for a particular variable
        '''
        name = name.replace('$','')
        url = '/variables/%s/TimeSeries'%name
        result = self.retrieve_json(url)
        df = pd.DataFrame(self.convert_dates(result['Events'])).set_index('Date').rename({'Value':result['Name']})
        extensions._apply_time_series_helpers(df)
        return df

    def update_variable_time_series(self,name,timeseries):
        name = name.replace('$','')
        url = '/variables/%s/TimeSeries'%name

        if hasattr(timeseries,'columns'):
            date_format = '%m/%d/%Y'
            payload = {}
            events = zip(timeseries.index,timeseries[timeseries.columns[0]])
            payload['Events'] = [{'Date':d.strftime(date_format),'Value':v} for d,v in events]
            payload['StartDate'] = timeseries.index[0].strftime(date_format)
            payload['EndDate'] = timeseries.index[-1].strftime(date_format)
            timeseries = payload

        return self.update_json(url,timeseries)

    def variable_piecewise(self,name):
        '''
        Returns piecewise linear function for a particular variable
        '''
        name = name.replace('$','')
        url = '/variables/%s/Piecewise'%name
        result = self.retrieve_json(url)
        return pd.DataFrame(result['Entries'],columns=[result[c] for c in ['XName','YName']])

    def update_variable_piecewise(self,name,values):
        '''
        Update piecewise linear function for a given variable.

        name: str, variable name to update.
        '''
        name = name.replace('$','')
        url = '/variables/%s/Piecewise'%name
        if hasattr(values,'columns'):
            payload = {}
            entries = list(zip(values[values.columns[0]],values[values.columns[1]]))
            payload['Entries'] = [[float(x),float(y)] for (x,y) in entries]
            payload['XName'] = values.columns[0]
            payload['YName'] = values.columns[1]
            values = payload

        print(values)
        return self.update_json(url,values)


    def data_sources(self):
        '''
        Return a SearchableList of the data sources in the Source model

        Note: Returns a summary (min,max,mean,etc) of individual time series - NOT the full record.

        You can get the time series by retrieving individual data sources (`data_source` method)
        '''
        return SearchableList(self.retrieve_json('/dataSources'))

    def data_source(self,name):
        '''
        Return an individual data source, by name.

        Note: Will include the each time series associated with the data source IN FULL
        '''
        prefix = '/dataSources/'
        if not name.startswith(prefix):
            name = prefix+name
        result = self.retrieve_json(name)

        def _transform_details(details):
            if 'Events' in details[0]['TimeSeries']:
                data_dict = {d['Name']:d['TimeSeries']['Events'] for d in details}
                return self._create_timeseries_dataframe(data_dict,common_index=False)

            # Slim Time Series...
            ts = details[0]['TimeSeries']
            start_t = self.parse_veneer_date(ts['StartDate'])
            end_t = self.parse_veneer_date(ts['EndDate'])
            freq = ts['TimeStep'][0]
            index = pd.date_range(start_t,end_t,freq=freq)
            data_dict = {d['Name']:d['TimeSeries']['Values'] for d in details}
            df = pd.DataFrame(data_dict,index=index)
            extensions._apply_time_series_helpers(df)
            return df

        def _transform_data_source_item(item):
            item['Details'] = _transform_details(item['Details'])
            return item

        result['Items'] = SearchableList([_transform_data_source_item(i) for i in result['Items']])
        return result

    def create_data_source(self,name,data,units='mm/day',precision=3):
        '''
        Create a new data source (name) using a Pandas dataframe (data)
        '''
        dummy_data_group = {}
        dummy_data_group['Name']=name
        dummy_item = {}
        dummy_item['Name']='Item for %s'%name
        dummy_item['InputSets']=['Default Input Set']

        dummy_detail = {}
        dummy_detail['Name'] = 'Details for %s'%name
        dummy_detail['TimeSeries']={}

        #dummy_item['Details'] = [dummy_detail]
        dummy_item['DetailsAsCSV']=data.to_csv(float_format='%%.%df'%precision)
        dummy_item['UnitsForNewTS']=units
        dummy_data_group['Items']=[dummy_item]

        return self.post_json('/dataSources',data=dummy_data_group)

    def data_source_item(self,source,name=None,input_set='__all__'):
        if name:
            source = '/'.join([source,input_set,_veneer_url_safe_id_string(name)])
        else:
            name = source

        prefix = '/dataSources/'
        if not source.startswith(prefix):
            source = prefix+source
        result = self.retrieve_json(source)

        def _transform(res):
            if 'TimeSeries' in res:
                return self._create_timeseries_dataframe({name:res['TimeSeries']['Events']},common_index=False)
            elif 'Items' in res:
                data_dict = {}
                suffix = ''
                for item in res['Items']:
                    if len(res['Items'])>1:
                        suffix = " (%s)"%item['Name']

                    if 'Details' in item:
                        update = {("%s%s"%(d['Name'],suffix)):d['TimeSeries']['Events'] for d in item['Details']}

                        data_dict.update(update)

                return self._create_timeseries_dataframe(data_dict,common_index=False)
            return res

        if isinstance(result,list):
            if len(result)==1:
                result = result[0]
            else:
                return [_transform(r) for r in result]
        return _transform(result)

    def result_matches_criteria(self,result,criteria):
        import re
#        MATCH_ALL='__all__'
        for key,pattern in criteria.items():
#            if pattern==MATCH_ALL: continue
            if not re.match(pattern,result[key]):
                return False
        return True

    def input_sets(self):
        '''
        Return a SearchableList of the input sets in the Source model

        Each input set will be a Python dictionary representing the different information in the input set
        '''
        return SearchableList(self.retrieve_json('/inputSets'))

    def update_input_set(self,name,input_set):
        '''
        Modify the input set and send to Source.

        name: str, name of input set
        input_set: A Python dictionary representing the updated input set. Should contain the same fields as the input set
                   returned from the input_sets method.
        '''
        return self.send_json('/inputSets/%s'%(name.replace(' ','%20')),method='PUT',data=input_set)

    def create_input_set(self,input_set):
        '''
        Create a new input set in Source model.

        input_set: A Python dictionary representing the updated input set. Should contain the same fields as the input set
                   returned from the input_sets method. (eg Configuration,Filename,Name,ReloadOnRun)
        '''
        return self.post_json('/inputSets',data=input_set)
        
    def apply_input_set(self,name):
        '''
        Have Source apply a given input set
        '''
        return self.send('/inputSets/%s/run'%(name.replace('%','%25').replace(' ','%20')),'POST')


    def retrieve_multiple_time_series(self,run='latest',run_data=None,criteria={},timestep='daily',name_fn=name_element_variable):
        """
        Retrieve multiple time series from a run according to some criteria.

        Return all time series in a single Pandas DataFrame with date time index.

        you can an index of run results via run_data. If you don't the method will first retrieve an
        index based on the value of the run parameter (default='latest')

        criteria should be regexps for the fields in a Veneer time series record:
          * NetworkElement
          * RecordingElement
          * RecordingVariable
          * TimeSeriesName
          * TimeSeriesUrl

        These criteria are used to identify which time series to retrieve.

        timestep should be one of 'daily' (default), 'monthly', 'annual'.
        *WARNING*: The monthly and annual option uses the corresponding option in the Veneer plugin, which ALWAYS SUMS values,
        regardless of units. So, if you retrieve a rate variable (eg m^3/s) those values will be summed and you will need to 
        correct this manually in the returned DataFrame.

        All retrieved time series are returned in a single Data Frame.

        You can specify a function for naming the columns of the Data Frame using name_fn. This function should take
        the results summary (from the index) and return a string. Example functions include:
          * veneer.name_time_series (uses the full name of the time series, as provided by Source)
          * veneer.name_element_variable (DEFAULT: users the name of the network element and the name of the variable)
          * veneer.name_for_location (just use the name of the network element)
          * veneer.name_for_variable (just use the name of the variable)
        """
        if timestep=="daily":
            suffix = ""
        else:
            suffix = "/aggregated/%s"%timestep

        if run_data is None:
            run_data = self.retrieve_run(run)

        retrieved={}
        def name_column(result):
            col_name = name_fn(result)
            if col_name in retrieved:
                i = 1
                alt_col_name = '%s %d'%(col_name,i)
                while alt_col_name in retrieved:
                    i += 1
                    alt_col_name = '%s %d'%(col_name,i)
                col_name = alt_col_name
            return col_name

        units_store = {}
        for result in run_data['Results']:
            if self.result_matches_criteria(result,criteria):
                d = self.retrieve_json(result['TimeSeriesUrl']+suffix)
                result.update(d)
                col_name = name_column(result)
#                    raise Exception("Duplicate column name: %s"%col_name)
                if 'Events' in d:
                    retrieved[col_name] = d['Events']
                    units_store[col_name] = result['Units']
                else:
                    all_ts = d['TimeSeries']
                    for ts in all_ts:
                        col_name = name_column(ts)
                        units_store[col_name] = ts['Units']

                        vals = ts['Values']
                        s = self.parse_veneer_date(ts['StartDate'])
                        e = self.parse_veneer_date(ts['EndDate'])
                        if ts['TimeStep']=='Daily':
                            f='D'
                        elif ts['TimeStep']=='Monthly':
                            f='M'
                        elif ts['TimeStep']=='Annual':
                            f='A'
                        dates = pd.date_range(s,e,freq=f)
                        retrieved[col_name] = [{'Date':d,'Value':v} for d,v in zip(dates,vals)]
                    # Multi Time Series!

        result = self._create_timeseries_dataframe(retrieved)
        for k,u in units_store.items():
            result[k].units = u

        return result

    def parse_veneer_date(self,txt):
        if hasattr(txt,'strftime'):
            return txt
        return pd.datetime.strptime(txt,'%m/%d/%Y %H:%M:%S')

    def convert_dates(self,events):
        return [{'Date':self.parse_veneer_date(e['Date']),'Value':e['Value']} for e in events]

    def _create_timeseries_dataframe(self,data_dict,common_index=True):
        if len(data_dict) == 0:
            df = pd.DataFrame()
        elif common_index:
            index = [self.parse_veneer_date(event['Date']) for event in list(data_dict.values())[0]]
            data = {k:[event['Value'] for event in result] for k,result in data_dict.items()}
            df = pd.DataFrame(data=data,index=index)
        else:
            from functools import reduce
            dataFrames = [pd.DataFrame(self.convert_dates(ts)).set_index('Date').rename(columns={'Value':k}) for k,ts in data_dict.items()]
            df = reduce(lambda l,r: l.join(r,how='outer'),dataFrames)
        extensions._apply_time_series_helpers(df)
        return df

def read_sdt(fn):
    ts = pd.read_table(fn,sep=' +',engine='python',names=['Year','Month','Day','Val'])
    ts['Date'] = ts.apply(lambda row: pd.datetime(int(row.Year),int(row.Month),int(row.Day)),axis=1)
    ts = ts.set_index(ts.Date)
    return ts.Val

def to_source_date(the_date):
    if hasattr(the_date,'strftime'):
        return the_date.strftime('%d/%m/%Y')
    return the_date

def read_rescsv(fn):
    '''
    Read a .res.csv file saved from Source

    Returns
      * attributes - Pandas Dataframe of the various metadata attributes in the file
      * data - Pandas dataframe of the time series
    '''
    import pandas as pd
    import re
    import io
    text = open(fn,'r').read()

    r = re.compile('\nEOH\n')
    header,body = r.split(text)

    r = re.compile('\nEOC\n')
    config,headers = r.split(header)

    attribute_names = config.splitlines()[-1].split(',')
    attributes = pd.DataFrame([dict(zip(attribute_names,line.split(','))) for line in headers.splitlines()[1:]])

    columns = ['Date'] + list(attributes.Name)
    data = pd.read_csv(io.StringIO(body),header=None,index_col=0,parse_dates=True,dayfirst=True,names=columns)

    return attributes, data

if __name__ == '__main__':
    # Output
    destination = sys.argv[1] if len(sys.argv)>1 else "C:\\temp\\veneer_download\\"
    print("Downloading all Veneer data to %s"%destination)
    retriever = VeneerRetriever(destination)
    retriever.retrieve_all(destination)
