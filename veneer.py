try:
    from urllib2 import urlopen, quote
except:
    from urllib.request import urlopen, quote

import json
import http.client as hc

# Source

PRINT_URLS=True
PRINT_ALL=False
PRINT_SCRIPTS=False

def name_time_series(result):
    return result['TimeSeriesName']

def name_element_variable(result):
    element = result['NetworkElement']
    variable = result['RecordingVariable'].split(' - ')[-1]
    return '%s:%s'%(element,variable)

def name_for_variable(result):
    return result['RecordingVariable']

def name_for_location(result):
    return result['NetworkElement']

def _stringToList(string_or_list):
    if isinstance(string_or_list,str):
        return [string_or_list]
    return string_or_list

def log(text):
    import sys
    print('\n'.join(_stringToList(text)))
    sys.stdout.flush()

class Veneer(object):
    def __init__(self,port=9876,host='localhost',protocol='http',prefix='',live=True):
        self.port=port
        self.host=host
        self.protocol=protocol
        self.prefix=prefix
        self.base_url = "%s://%s:%d%s" % (protocol,host,port,prefix)
        self.live_source=live
        if self.live_source:
            self.data_ext=''
        else:
            self.data_ext='.json'
        self.model = VeneerIronPython(self)

#   def retrieve_resource(self,url,ext):
#       if PRINT_URLS:
#           print("*** %s ***" % (url))
#
#       save_data(url[1:],urlopen(base_url+quote(url)).read(),ext,mode="b")

    def retrieve_json(self,url):
        if PRINT_URLS:
            print("*** %s ***" % (url))

        text = urlopen(self.base_url + quote(url+self.data_ext)).read().decode('utf-8')
        
        if PRINT_ALL:
            print(json.loads(text))
            print("")
        return json.loads(text)

    def update_json(self,url,data,async=False):
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
        if PRINT_SCRIPTS: print(script)
        result = self.post_json('/ironpython',{'Script':script},async=async)
        if async:
            return result
        code,data = result
        if code == 403:
            raise Exception('Script disabled. Enable scripting in Veneer')
        return data

    def configure_recording(self,enable=[],disable=[]):
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

    def run_model(self,params={},start=None,end=None,async=False):
        conn = hc.HTTPConnection(self.host,port=self.port)

        if not start is None:
            params['StartDate'] = to_source_date(start)
        if not end is None:
            params['EndDate'] = to_source_date(end)

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
        else:
            return code,resp.read().decode('utf-8')

    def drop_run(self,run='latest'):
        assert self.live_source
        conn = hc.HTTPConnection(self.host,port=self.port)
        conn.request('DELETE','/runs/%s'%str(run))
        resp = conn.getresponse()
        code = resp.getcode()
        return code

    def drop_all_runs(self):
        while len(self.retrieve_runs())>0:
            self.drop_run()

    def retrieve_runs(self):
        return self.retrieve_json('/runs')

    def retrieve_run(self,run='latest'):
        if run=='latest' and not self.live_source:
            all_runs = self.retrieve_json('/runs')
            result = self.retrieve_json(all_runs[-1]['RunUrl'])
        else:
            result = self.retrieve_json('/runs/%s'%str(run))
        result['Results'] = SearchableList(result['Results'])
        return result

    def network(self):
        result = self.retrieve_json('/network')
        result['features'] = SearchableList(result['features'],['geometry','properties'])
        return result

    def functions(self):
        return SearchableList(self.retrieve_json('/functions'))

    def variables(self):       
        return SearchableList(self.retrieve_json('/variables'))

    def result_matches_criteria(self,result,criteria):
        import re
#        MATCH_ALL='__all__'
        for key,pattern in criteria.items():
#            if pattern==MATCH_ALL: continue
            if not re.match(pattern,result[key]):
                return False
        return True

    def input_sets(self):
        return SearchableList(self.retrieve_json('/inputSets'))

    def apply_input_set(self,name):
        return self.send('/inputSets/%s/run'%(name.replace(' ','%20')),'POST')

    def retrieve_multiple_time_series(self,run='latest',run_data=None,criteria={},timestep='daily',name_fn=name_element_variable):
        """
        Retrieve multiple time series from a run according to some criteria.

        Return all time series in a single Pandas DataFrame with date time index.

        Crtieria should be regexps for the fields in a Veneer time series record:
          * RecordingElement
          * RecordingVariable
          * TimeSeriesName
          * TimeSeriesUrl

        timestep should be one of 'daily' (default), 'monthly', 'annual'
        """
        from pandas import DataFrame
        if timestep=="daily":
            suffix = ""
        else:
            suffix = "/aggregated/%s"%timestep

        if run_data is None:
            run_data = self.retrieve_run(run)

        retrieved={}

        for result in run_data['Results']:
            if self.result_matches_criteria(result,criteria):
                retrieved[name_fn(result)] = self.retrieve_json(result['TimeSeriesUrl']+suffix)['Events']

        if len(retrieved) == 0:
            return DataFrame()
        else:
            index = [event['Date'] for event in list(retrieved.values())[0]]
            data = {k:[event['Value'] for event in result] for k,result in retrieved.items()}
            return DataFrame(data=data,index=index)

class VeneerIronPython(object):
    """
    Helper functions for manipulating the internals of the Source model itself.

    These features rely on 'Allow Scripting' being enabled in Veneer in order to
    post custom IronPython scripts to Source.
    """
    def __init__(self,veneer):
        self._veneer = veneer
        self.ui = VeneerSourceUIHelpers(self)
        self._generator = VeneerScriptGenerators(self)
        self.catchment = VeneerCatchmentActions(self)
        self.link = VeneerLinkActions(self)

    def _initScript(self,namespace=None):
        script = "# Generated Script\n"
        if not namespace is None:
            script += "import %s\n"%namespace
        script += "import clr\n"
        script += "import System\n"
        script += "import FlowMatters.Source.Veneer.RemoteScripting.ScriptHelpers as H\n"
        script += "clr.ImportExtensions(System.Linq)\n"
        return script

    def runScript(self,script,async=False):
        return self.run_script(script,async)

    def run_script(self,script,async=False):
        return self._veneer.run_server_side_script(script,async)

    def sourceHelp(self,theThing='scenario',namespace=None):
        """
        Get some help on what you can do with theThing,
        where theThing is something you can access from a scenario, or the scenario itself.

        If theThing is a method, returns details on how to call the method
        If theThing is an object, returns a list of public methods and properties

        eg
        v.model.sourceHelp() # returns help on the scenario object
        v.model.sourceHelp('scenario.CurrentConfiguration')
        """
        #theThing = theThing.replace('.*','.First().')

        script = self._initScript(namespace)
        innerLoop = "theThing = %s%s\n"
        innerLoop += "if hasattr(theThing, '__call__'):\n"
        innerLoop += "    result = 'function'\n"
        innerLoop += "    help(theThing)\n"
        innerLoop += "else:\n"
        innerLoop += "    result = dir(theThing)"
        script += self._generateLoop(theThing,innerLoop,first=True)
        data = self.run_script(script)
        if not data['Exception'] is None:
            raise Exception(data['Exception'])
        if data['Response'] is None:
            raise Exception('Could not find anything matching %s'%theThing)
        if data['Response']['Value']=='function':
            print(data['StandardOut'])
        else:
            return [d['Value'] for d in data['Response']['Value']]

    def _generateLoop(self,theThing,innerLoop,first=False):
        script = ''
        script += "have_succeeded = False\n"
        script += "ignoreExceptions = True\n"
        indent = 0
        indentText = ''
        levels = theThing.split('.*')
        prevLoop = ''
        for level in levels[0:-1]:
            loopVar = "i_%d"%indent
            script += indentText+'for %s in %s%s:\n'%(loopVar,prevLoop,level)
            indent += 1
            indentText = ' '*(indent*4) 
            script += indentText+'try:\n'
            indent += 1
            indentText = ' '*(indent*4)
            prevLoop = loopVar+'.'
        script += indentText
        substCount = innerLoop.count('%s')//2
        script += innerLoop.replace('\n','\n'+indentText)%tuple([prevLoop,levels[-1]]*substCount)
        script += '\n'
        script += indentText + "have_succeeded = True\n"
        while indent >0:
            indent -= 1
            indentText = ' '*(indent*4) 
            script += indentText + "except:\n"
            script += indentText + '  if not ignoreExceptions: raise\n'
            if first:
                script += indentText + "if have_succeeded: break\n"
            indent -= 1     

        return script

    def find_model_type(self,model_type):
        script = self._initScript('TIME.Management.Finder as Finder')
        script += 'import TIME.Core.Model as Model\n'
        script += 'f = Finder(Model)\n'
        script += 'types = f.types()\n'
        script += 's = "%s"\n'%model_type
        script += 'result = types.Where(lambda t:t.Name.ToLower().Contains(s.ToLower()))'
        script += '.Select(lambda tt:tt.FullName)\n'
        res = self._safe_run(script)
        return [v['Value'] for v in res['Response']['Value']]

    def _find_members_with_attribute_in_type(self,model_type,attribute):
        script = self._initScript(model_type)
        script += 'from TIME.Core.Metadata import %s\n'%attribute
        script += 'result = []\n'
        script += 'tmp = %s()\n'%model_type
        script += 'typeObject = tmp.GetType()\n'
        script += 'for member in dir(tmp):\n'
        script += '  try:\n'
        script += '    if typeObject.GetMember(member)[0].IsDefined(%s,True):\n'%attribute
        script += '      result.append(member)\n'
        script += '  except: pass'
        res = self._safe_run(script)
        return [v['Value'] for v in res['Response']['Value']]

    def _find_members_with_attribute_for_types(self,model_types,attribute):
        model_types = list(set(_stringToList(model_types)))
        result = {}
        for t in model_types:
            result[t] = self._find_members_with_attribute_in_type(t,attribute)
        if len(model_types)==1:
            return result[model_types[0]]
        return result

    def find_parameters(self,model_types):
        """
        Find the parameter names for a given model type or list of model types

        Returns:
         * A list of parameters (if a single model type is provided)
         * A dictionary model type name -> list of parameters (if more than model type)
        """
        return self._find_members_with_attribute_for_types(model_types,'ParameterAttribute')

    def find_inputs(self,model_types):
        """
        Find the input names for a given model type or list of model types

        Returns:
         * A list of inputs (if a single model type is provided)
         * A dictionary model type name -> list of inputs (if more than model type)
        """
        return self._find_members_with_attribute_for_types(model_types,'InputAttribute')

    def get(self,theThing,namespace=None):
        """
        Retrieve a value, or list of values from Source using theThing as a query string.

        Query should either start with `scenario` OR from a class imported using `namespace`
        """
        script = self._initScript(namespace)
        listQuery = theThing.find(".*") != -1
        if listQuery:
            script += 'result = []\n'
            innerLoop = 'result.append(%s%s)'
            script += self._generateLoop(theThing,innerLoop)
        else:
            script += "result = %s\n"%theThing
#       return script

        resp = self.run_script(script)
        if not resp['Exception'] is None:
            raise Exception(resp['Exception'])
        data = resp['Response']['Value']
        if listQuery:
            return [d['Value'] for d in data]
        return data

    def _assignment(self,theThing,theValue,namespace=None,literal=False,
                    fromList=False,instantiate=False,
                    assignment="",post_assignment="",
                    print_script=False):
        val_transform='()' if instantiate else ''
        if literal and isinstance(theValue,str):
            theValue = "'"+theValue+"'"
        if fromList:
            if literal:
                theValue = [("'"+v+"'") if isinstance(v,str) else v for v in theValue]
            theValue = '['+(','.join([str(s) for s in theValue]))+']'
        script = self._initScript(namespace)
        script += 'origNewVal = %s\n'%theValue
        if fromList:
            script += 'origNewVal.reverse()\n'
            script += 'newVal = origNewVal[:]\n'
        else:
            script += 'newVal = origNewVal\n'

        innerLoop = 'ignoreExceptions = True\n'
        innerLoop += 'checkValueExists = %s%s\n'
        innerLoop += 'ignoreExceptions = False\n'
        innerLoop += assignment
        if fromList:
            innerLoop += '.pop()'+ val_transform + post_assignment +'\n'
            innerLoop += 'if len(newVal)==0: newVal = origNewVal[:]\n'
        else:
            innerLoop += val_transform + post_assignment
        script += self._generateLoop(theThing,innerLoop)
        script += 'result = have_succeeded\n'
#       return script
#        return None
        result = self.run_script(script)
        if not result['Exception'] is None:
            raise Exception(result['Exception'])
        return result['Response']['Value']

    def set(self,theThing,theValue,namespace=None,literal=False,fromList=False,instantiate=False):
        return self._assignment(theThing,theValue,namespace,literal,fromList,instantiate,"%s%s = newVal","")

    def add_to_list(self,theThing,theValue,namespace=None,literal=False,
                    fromList=False,instantiate=False,allow_duplicates=False):
        if allow_duplicates:
            assignment = "%s%s.Add(newVal"
        else:
            assignment = "theList=%s%s\nif not H.ListContainsInstance(theList,newVal"
            if instantiate: assignment += "()"
            assignment +="): theList.Add(newVal"

        return self._assignment(theThing,theValue,namespace,literal,fromList,instantiate,assignment,")")

    def assign_time_series(self,theThing,theValue,column=0,from_list=False,literal=True,
                           data_group=None):
        ns = None
        assignment = "H.AssignTimeSeries(scenario,%s__init__.__self__,'%s','"+data_group+"',newVal"
        post_assignment = ",%d)"%column
        return self._assignment(theThing,theValue,ns,literal,from_list,False,assignment,post_assignment)

    def sourceScenarioOptions(self,optionType,option=None,newVal = None):
        self.source_scenario_options(optionType,option,newVal)

    def source_scenario_options(self,optionType,option=None,newVal = None):
        script = self._initScript('RiverSystem.ScenarioConfiguration.%s as %s'%(optionType,optionType))
        retrieve = "scenario.GetScenarioConfiguration[%s]()"%optionType
        if option is None:
            script += "result = dir(%s)\n"%retrieve
        else:
            if not newVal is None:
                script += "%s.%s = %s\n"%(retrieve,option,newVal)
            script += "result = %s.%s\n"%(retrieve,option)
        res = self._safe_run(script)
        if newVal is None:
            return res

    def _safe_run(self,script):
        result = self.run_script(script)
        if not result['Exception'] is None:
            raise Exception(result['Exception'])
        return result

    def get_constituents(self):
        s = self._initScript()
        s += 'result = scenario.SystemConfiguration.Constituents.Select(lambda c: c.Name)\n'
        return self._safe_run(s)['Response']['Value']

    def add_constituent(self,new_constituent):
        s = self._initScript(namespace='RiverSystem.Constituents.Constituent as Constituent')
        s += 'scenario.Network.ConstituentsManagement.Config.ProcessConstituents = True\n' 
        s += 'theList = scenario.SystemConfiguration.Constituents\n'
        s += 'if not theList.Any(lambda c: c.Name=="%s"):\n'%new_constituent
        s += '  c=Constituent("%s")\n'%new_constituent
        s += '  theList.Add(c)\n'
        s += '  H.InitialiseModelsForConstituent(scenario,c)\n'
#        s += 'nw = scenario.Network\n'
#        s += 'nw.ConstituentsManagement.Reset(scenario.CurrentConfiguration.StartDate)\n'
        return self._safe_run(s)

class VeneerScriptGenerators(object):
    def __init__(self,ironpython):
        self._ironpy = ironpython

    def find_feature_by_name(self,name):
        script = self._ironpy._initScript()
        script += "def find_feature_by_name(searchTerm):\n"
        script += "  for n in scenario.Network.Nodes:\n"
        script += "    if n.Name.startswith(searchTerm):\n"
        script += "      return n\n"
        script += "  for l in scenario.Network.Links:\n"
        script += "    if l.Name.startswith(searchTerm):\n"
        script += "      return l\n"
        script += "  return None\n\n"
        return script

class VeneerSourceUIHelpers(object):
    def __init__(self,ironpython):
        self._ironpy = ironpython

    def open_editor(self,name_of_element):
        script = self._ironpy._initScript(namespace="RiverSystem.Controls.Controllers.FeatureEditorController as FeatureEditorController")
        script += self._ironpy._generator.find_feature_by_name(name_of_element)
        script += "f = find_feature_by_name('%s')\n"%name_of_element
        script += "if not f is None:\n"
        script += "  ctrl = FeatureEditorController(f)\n"
        script += "  ctrl.Initialise(scenario)\n"
        script += "  ctrl.Show(None)\n"
        
        self._ironpy._safe_run(script)

class VeneerNetworkElementActions(object):
    def __init__(self,ironpython):
        self._ironpy = ironpython
        self._ns = None

    def _instantiation_namespace(self,types):
        ns = ','.join(set(types))
        if not self._ns is None:
            ns += ','+self._ns
        return ns

    def get_models(self,**kwargs):
        return self.get_param_values('GetType().FullName',**kwargs)

    def get_param_values(self,parameter,**kwargs):
        accessor = self._build_accessor(parameter,**kwargs)
        return self._ironpy.get(accessor,kwargs.get('namespace',self._ns))

    def set_models(self,models,fromList=False,**kwargs):
        return self.set_param_values(None,models,fromList=fromList,instantiate=True,**kwargs)

    def set_param_values(self,parameter,values,literal=False,fromList=False,instantiate=False,**kwargs):
        accessor = self._build_accessor(parameter,**kwargs)
        ns = self._ns
        if instantiate:
            values = _stringToList(values)
            ns = self._instantiation_namespace(values)
            fromList = True
        return self._ironpy.set(accessor,values,ns,literal=literal,fromList=fromList,instantiate=instantiate)

class VeneerFunctionalUnitActions(VeneerNetworkElementActions):
    def __init__(self,catchment):
        self._catchment = catchment
        super(VeneerFunctionalUnitActions,self).__init__(catchment._ironpy)

    def _build_accessor(self,parameter=None,catchments=None,fus=None):
        accessor = 'scenario.Network.Catchments'

        if not catchments is None:
            catchments = _stringToList(catchments)
            accessor += '.Where(lambda c: c.DisplayName in %s)'%catchments

        accessor += '.*FunctionalUnits'

        if not fus is None:
            fus = _stringToList(fus)
            accessor += '.Where(lambda fu: fu.DisplayName in %s)'%fus

        if not parameter is None:
            accessor += '.*%s'%parameter

        return accessor

class VeneerCatchmentActions(VeneerFunctionalUnitActions):
    def __init__(self,ironpython):
        self._ironpy = ironpython
        self.runoff = VeneerRunoffActions(self)
        self.generation = VeneerCatchmentGenerationActions(self)
        self.subcatchment = VeneerSubcatchmentActions(self)
        self._ns = None

    def get_areas(self,catchments=None):
        return self._ironpy.get('scenario.Network.Catchments.*characteristics.areaInSquareMeters')

    def get_functional_unit_areas(self,catchments=None,fus=None):
        return self.get_param_values('areaInSquareMeters',catchments=catchments,fus=fus)

    def set_functional_unit_areas(self,values,catchments=None,fus=None):
        return self.set_param_values('areaInSquareMeters',values,fromList=True,catchments=catchments,fus=fus)

    def get_functional_unit_types(self,catchments=None,fus=None):
        return self.get_param_values('definition.Name')

class VeneerRunoffActions(VeneerFunctionalUnitActions):
    def __init__(self,catchment):
        super(VeneerRunoffActions,self).__init__(catchment)

    def _build_accessor(self,parameter=None,catchments=None,fus=None):
        accessor = super(VeneerRunoffActions,self)._build_accessor('rainfallRunoffModel',catchments,fus)

        if not parameter is None:
            accessor += '.%s'%parameter

        return accessor

#    def get_models(self,catchments=None,fus=None):
#        return self.get_param_values('GetType().FullName',catchments,fus)

#    def get_param_values(self,parameter,catchments=None,fus=None):
#        accessor = self._build_accessor(parameter,catchments,fus)
#        return self._ironpy.get(accessor)

    def assign_time_series(self,parameter,values,data_group,column=0,
                           catchments=None,fus=None,literal=True,fromList=False):
        accessor = self._build_accessor(parameter,catchments,fus)
        return self._ironpy.assign_time_series(accessor,values,from_list=fromList,
                                               literal=literal,column=column,
                                               data_group=data_group)

class VeneerCatchmentGenerationActions(VeneerFunctionalUnitActions):
    def __init__(self,catchment):
        super(VeneerCatchmentGenerationActions,self).__init__(catchment)
        self._ns = 'RiverSystem.Constituents.CatchmentElementConstituentData as CatchmentElementConstituentData'

    def _build_accessor(self,parameter,catchments=None,fus=None,constituents=None):
        accessor = 'scenario.Network.ConstituentsManagement.Elements' + \
                    '.OfType[CatchmentElementConstituentData]()' + \
                    '.*FunctionalUnitData'

        if not fus is None:
            fus = _stringToList(fus)
            accessor += '.Where(lambda fuData: fuData.DisplayName in %s)'%fus

        accessor += '.*ConstituentModels'

        if not constituents is None:
            constituents = _stringToList(constituents)
            accessor +=  '.Where(lambda x: x.Constituent.Name in %s)'%constituents

        accessor += '.*ConstituentSources.*GenerationModel'
        if not parameter is None:
            accessor += '.%s'%parameter

        return accessor

class VeneerSubcatchmentActions(VeneerNetworkElementActions):
    def __init__(self,catchment):
        self._catchment = catchment
        super(VeneerSubcatchmentActions,self).__init__(catchment._ironpy)

    def _build_accessor(self,parameter,**kwargs):
        accessor = 'scenario.Network.Catchments'

        if not kwargs.get('catchments') is None:
            catchments = _stringToList(kwargs['catchments'])
            accessor += '.Where(lambda sc: sc.DisplayName in %s)'%catchments

        accessor += '.*CatchmentModels'

        if not parameter is None:
            accessor += '.*%s'%parameter
        return accessor

#    def get_param_values(self,parameter,**kwargs):
#        accessor = self._build_accessor(parameter,**kwargs)
#        return self._ironpy.get(accessor)

#    def get_models(self,**kwargs):
#        return self.get_param_values('GetType().FullName',**kwargs)

    def add_model(self,model_type,add_if_existing=False,catchments=None,allow_duplicates=False):
        accessor = self._build_accessor(parameter=None,catchments=catchments)
        return self._ironpy.add_to_list(accessor,model_type,model_type,
                                        instantiate=True,allow_duplicates=allow_duplicates)

class VeneerLinkActions(object):
    def __init__(self,ironpython):
        self._ironpy = ironpython
        self.constituents = VeneerLinkConstituentActions(self)
        self.routing = VeneerLinkRoutingActions(self)

class VeneerLinkConstituentActions(VeneerNetworkElementActions):
    def __init__(self,link):
        self._link = link
        super(VeneerLinkConstituentActions,self).__init__(link._ironpy)
        self._ns = 'RiverSystem.Constituents.LinkElementConstituentData as LinkElementConstituentData'

    def _build_accessor(self,parameter=None,links=None,constituents=None):
        accessor = 'scenario.Network.ConstituentsManagement.Elements' + \
                    '.OfType[LinkElementConstituentData]()'

        if not links is None:
            links = _stringToList(links)
            accessor += '.Where(lambda lecd: lecd.Element.DisplayName in %s)'%links

        accessor += '.*Data' + \
                    '.ProcessingModels'

        if not constituents is None:
            constituents = _stringToList(constituents)
            accessor += '.Where(lambda c: c.Constituent.Name in %s)'%constituents

        accessor += '.*Model'

        if not parameter is None:
            accessor += '.%s'%parameter
        return accessor

class VeneerLinkRoutingActions(VeneerNetworkElementActions):
    def __init__(self,link):
        self._link = link
        super(VeneerLinkRoutingActions,self).__init__(link._ironpy)

    def _build_accessor(self,parameter=None,links=None):
        accessor = 'scenario.Network.Links'
        if not links is None:
            links = _stringToList(links)
            accessor += '.Where(lambda l:l.DisplayName in %s'%links
        accessor += '.*FlowRouting'
        return accessor

#    def set_model(self,theThing,theValue,namespace=None,literal=False,fromList=False,instantiate=False):
    def set_models(self,models,fromList=False,**kwargs):

        models = _stringToList(models)
        assignment = "theLink=%s%s\n"
        assignment += "val = newVal"

        post_assignment = "\n"
        post_assignment += "is_sr = 'StorageRouting' in val.__name__\n"
        post_assignment += "theLink.FlowRouting = val(theLink) if is_sr else val()"

        accessor = self._build_accessor()[:-13]+'.*__init__.__self__'
        namespace = self._instantiation_namespace(models)
        return self._ironpy._assignment(accessor,models,namespace,literal=False,fromList=True,
                                       instantiate=False,
                                       assignment=assignment,
                                       post_assignment=post_assignment)

class SearchableList(object):
    def __init__(self,the_list,nested=[]):
        self._list = the_list
        self._nested = nested

    def __repr__(self):
        return self._list.__repr__()

    def __len__(self):
        return len(self._list)

    def __getitem__(self,i):
        return self._list[i]

    def __iter__(self):
        return self._list.__iter__()

    def __reversed__(self):
        return SearchableList(reversed(self._list))

    def __contains__(self,item):
        return self._list.__contains__(item)

    def _search_all(self,key,val,entry):
        if (key in entry) and entry[key]==val: return True
        for nested in self._nested:
            if not nested in entry: continue
            if not key in entry[nested]: continue
            if entry[nested][key]==val: return True
        return False

    def _nested_retrieve(self,key,entry):
        if (key in entry): return entry[key]
        for nested in self._nested:
            if not nested in entry: continue
            if key in entry[nested]: return entry[nested][key]
        return None

    def _unique_values(self,key):
        return set(self._all_values(key))

    def _all_values(self,key):
        return [self._nested_retrieve(key,e) for e in self._list]

    def _select(self,keys,transforms={}):
        result = [{k:self._nested_retrieve(k,e) for k in keys} for e in self]

        for key,fn in transforms.items():
            for r,e in zip(result,self):
                r[key] = fn(e)

        if len(result)==0: SearchableList([])
        elif len(result[0])==1:
            key = list(result[0].keys())[0]
            return [r[key] for r in result]

        return SearchableList(result)

    def __getattr__(self,name):
        FIND_PREFIX='find_by_'
        if name.startswith(FIND_PREFIX):
            field_name = name[len(FIND_PREFIX):]
            return lambda x: SearchableList(list(filter(lambda y: self._search_all(field_name,x,y),self._list)),self._nested)

        GROUP_PREFIX='group_by_'
        if name.startswith(GROUP_PREFIX):
            field_name = name[len(GROUP_PREFIX):]
            return lambda: {k:self.__getattr__(FIND_PREFIX+field_name)(k) for k in self._unique_values(field_name)}
        raise AttributeError(name + ' not allowed')

def read_sdt(fn):
    import pandas as pd
    ts = pd.read_table(fn,sep=' +',engine='python',names=['Year','Month','Day','Val'])
    ts['Date'] = ts.apply(lambda row: pd.datetime(int(row.Year),int(row.Month),int(row.Day)),axis=1)
    ts = ts.set_index(ts.Date)
    return ts.Val

def to_source_date(the_date):
    if hasattr(the_date,'strftime'):
        return the_date.strftime('%d/%m/%Y')
    return the_date

class VeneerRetriever(object):
    def __init__(self,destination,port=9876,host='localhost',protocol='http',
                 retrieve_daily=True,retreive_monthly=True,retrieve_annual=True,
                 retrieve_slim_ts=True,retrieve_single_ts=True,
                 print_all = False, print_urls = True):
        self.destination = destination
        self.port = port
        self.host = host
        self.protocol = protocol
        self.retrieve_daily = retrieve_daily
        self.retreive_monthly = retreive_monthly
        self.retrieve_annual = retrieve_annual
        self.retrieve_slim_ts = retrieve_slim_ts
        self.retrieve_single_ts = retrieve_single_ts
        self.print_all = print_all
        self.print_urls = print_urls
        self.base_url = "%s://%s:%d" % (protocol,host,port)
        self._veneer = Veneer(host=self.host,port=self.port,protocol=self.protocol)

    def mkdirs(self,directory):
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save_data(self,base_name,data,ext,mode="b"):
        import os
        base_name = os.path.join(self.destination,base_name + "."+ext)
        directory = os.path.dirname(base_name)
        self.mkdirs(directory)
        f = open(base_name,"w"+mode)
        f.write(data)
        f.close()
    
    def retrieve_json(self,url,**kwargs):
        if self.print_urls:
            print("*** %s ***" % (url))
    
        text = urlopen(self.base_url + quote(url)).read().decode('utf-8')
        self.save_data(url[1:],bytes(text,'utf-8'),"json")
    
        if self.print_all:
            print(json.loads(text))
            print("")
        return json.loads(text)
    
    def retrieve_resource(self,url,ext):
        if self.print_urls:
            print("*** %s ***" % (url))
    
        self.save_data(url[1:],urlopen(self.base_url+quote(url)).read(),ext,mode="b")
    
    
    # Process Run list and results
    def retrieve_runs(self):
        run_list = self.retrieve_json("/runs")
        for run in run_list:
            run_results = self.retrieve_json(run['RunUrl'])
            ts_results = run_results['Results']
            if self.retrieve_single_ts:
                for result in ts_results:
                    self.retrieve_ts(result['TimeSeriesUrl'])

            if self.retrieve_slim_ts:
                self.retrieve_multi_ts(ts_results)

        if self.retrieve_slim_ts and len(run_list):
            self.retrieve_across_runs(ts_results)

    def retrieve_multi_ts(self,ts_results):
        recorders = list(set([(r['RecordingElement'],r['RecordingVariable']) for r in ts_results]))
        for r in recorders:
            for option in ts_results:
                if option['RecordingElement'] == r[0] and option['RecordingVariable'] == r[1]:
                    url = option['TimeSeriesUrl'].split('/')
                    url[4] = '__all__'
                    url = '/'.join(url)
                    self.retrieve_ts(url)
                    break

    def retrieve_across_runs(self,results_set):
        for option in results_set:
            url = option['TimeSeriesUrl'].split('/')
            url[2] = '__all__'
            url = '/'.join(url)
            self.retrieve_ts(url)
            break

    def retrieve_ts(self,ts_url):
        if self.retrieve_daily:
            self.retrieve_json(ts_url)
        if self.retreive_monthly:
            self.retrieve_json(ts_url + "/aggregated/monthly")
        if self.retrieve_annual:
            self.retrieve_json(ts_url + "/aggregated/annual")
    
    def retrieve_variables(self):
        variables = self.retrieve_json("/variables")
        for var in variables:
            if var['TimeSeries']: self.retrieve_json(var['TimeSeries'])
            if var['PiecewiseFunction']: self.retrieve_json(var['PiecewiseFunction'])

    def retrieve_all(self,destination,**kwargs):
        self.mkdirs(self.destination)
        self.retrieve_runs()
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

if __name__ == '__main__':
    # Output
    destination = sys.argv[1] if len(sys.argv)>1 else "C:\\temp\\veneer_download\\"
    print("Downloading all Veneer data to %s"%destination)
    retriever = VeneerRetriever(destination)
    retriever.retrieve_all(destination)
