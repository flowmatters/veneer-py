try:
    from urllib2 import urlopen, quote
except:
    from urllib.request import urlopen, quote

import json
import http.client as hc

# Source

PRINT_URLS=True
PRINT_ALL=False

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
        conn = hc.HTTPConnection(self.host,port=self.port)
        payload = json.dumps(data)
        conn.request(method,url,payload,headers={'Content-type':'application/json','Accept':'application/json'})
        if async:
            return conn
        resp = conn.getresponse()
        code = resp.getcode()
        if code==302:
            return code,resp.getheader('Location')
        elif code==200:
            return code,json.loads(resp.read().decode('utf-8'))
        else:
            return code,None

        return conn

    def post_json(self,url,data,async=False):
        return self.send_json(url,data,'POST',async)

    def run_server_side_script(self,script,async=False):
        #print(script)
        result = self.post_json('/ironpython',{'Script':script},async=async)
        if async:
            return result
        code,data = result
        if code == 403:
            raise Exception('Script disabled. Enable scripting in Veneer')
        return data

    def run_model(self,params={},async=False):

        conn = hc.HTTPConnection(self.host,port=self.port)
    #   conn.request('POST','/runs',json.dumps({'parameters':params}),headers={'Content-type':'application/json','Accept':'application/json'})
        conn.request('POST','/runs',json.dumps(params),headers={'Content-type':'application/json','Accept':'application/json'})
        if async:
            return conn

        resp = conn.getresponse()
        if resp.getcode()==302:
            return resp.getcode(),resp.getheader('Location')
        else:
            return resp.getcode(),None

    def retrieve_run(self,run='latest'):
        if run=='latest' and not self.live_source:
            all_runs = self.retrieve_json('/runs')
            return self.retrieve_json(all_runs[-1]['RunUrl'])

        return self.retrieve_json('/runs/%s'%str(run))

    def network(self):
        result = self.retrieve_json('/network')
        result['features'] = SearchableList(result['features'])
        return result

    def functions(self):
        return SearchableList(self.retrieve_json('/functions'))

    def variables(self):       
        return SearchableList(self.retrieve_json('/variables'))

    def result_matches_criteria(self,result,criteria):
        import re
        for key,pattern in criteria.items():
            if not re.match(pattern,result[key]):
                return False
        return True
        
    def name_time_series(self,result):
        return result['TimeSeriesName']

    def name_element_variable(self,result):
        element = result['NetworkElement']
        variable = result['RecordingVariable'].split(' - ')[-1]
        return '%s:%s'%(element,variable)

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
            index = [event['Date'] for event in retrieved.values()[0]]
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

    def _initScript(self,namespace=None):
        script = "# Generated Script\n"
        if not namespace is None:
            script += "import %s\n"%namespace
        script += "import clr\n"
        script += "import System\n"
        script += "clr.ImportExtensions(System.Linq)\n"
        return script

    def runScript(self,script,async=False):
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
        data = self.runScript(script)
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
            script += indentText + "except: pass\n"
            if first:
                script += indentText + "if have_succeeded: break\n"
            indent -= 1     

        return script

    def _stringToList(self,string_or_list):
        if isinstance(string_or_list,str):
            return [string_or_list]
        return string_or_list

    def _find_parameters_in_type(self,model_type):
        script = self._initScript(model_type)
        script += 'from TIME.Core.Metadata import ParameterAttribute\n'
        script += 'result = []\n'
        script += 'tmp = %s()\n'%model_type
        script += 'typeObject = tmp.GetType()\n'
        script += 'for member in dir(tmp):\n'
        script += '  try:\n'
        script += '    if typeObject.GetMember(member)[0].IsDefined(ParameterAttribute,True):\n'
        script += '      result.append(member)\n'
        script += '  except: pass'
#        print(script)
        res = self._safe_run(script)
        return [v['Value'] for v in res['Response']['Value']]

    def find_parameters(self,model_types):
        """
        Find the parameter names for a given model type or list of model types

        Returns:
         * A list of parameters (if a single model type is provided)
         * A dictionary model type name -> list of parameters (if more than model type)
        """
        model_types = list(set(self._stringToList(model_types)))
        result = {}
        for t in model_types:
            result[t] = self._find_parameters_in_type(t)
        if len(model_types)==1:
            return result[model_types[0]]
        return result

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
        resp = self.runScript(script)
        if not resp['Exception'] is None:
            raise Exception(resp['Exception'])
        data = resp['Response']['Value']
        if listQuery:
            return [d['Value'] for d in data]
        return data

    def set(self,theThing,theValue,namespace=None,literal=False,fromList=False):
        if literal and isinstance(theValue,str):
            theValue = "'"+theValue+"'"
        script = self._initScript(namespace)
        script += 'origNewVal = %s\n'%theValue
        if fromList:
            script += 'origNewVal.reverse()\n'
            script += 'newVal = origNewVal[:]\n'
        else:
            script += 'newVal = origNewVal\n'

        innerLoop = 'checkValueExists = %s%s\n'
        innerLoop += "%s%s = newVal"
        if fromList:
            innerLoop += '.pop()\n'
            innerLoop += 'if len(newVal)==0: newVal = origNewVal[:]\n'
        script += self._generateLoop(theThing,innerLoop)

#       return script
#        print(script)
#        return None
        result = self.runScript(script)
        if not result['Exception'] is None:
            raise Exception(result['Exception'])
        return None

    def sourceScenarioOptions(self,optionType,option=None,newVal = None):
        script = self._initScript('RiverSystem.ScenarioConfiguration.%s as %s'%(optionType,optionType))
        retrieve = "scenario.GetScenarioConfiguration[%s]()"%optionType
        if option is None:
            script += "result = dir(%s)\n"%retrieve
        else:
            if not newVal is None:
                script += "%s.%s = %s\n"%(retrieve,option,newVal)
            script += "result = %s.%s\n"%(retrieve,option)
        return self.runScript(script)

    def _safe_run(self,script):
        result = self.runScript(script)
        if not result['Exception'] is None:
            raise Exception(result['Exception'])
        return result

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

class VeneerCatchmentActions(object):
    def __init__(self,ironpython):
        self._ironpy = ironpython
        self.generation = VeneerCatchmentGenerationActions(self)

class VeneerCatchmentGenerationActions(object):
    def __init__(self,catchment):
        self._catchment = catchment
        self._ironpy = catchment._ironpy
        self._ns = 'RiverSystem.Constituents.CatchmentElementConstituentData as CatchmentElementConstituentData'

    def _build_accessor(self,parameter,catchments=None,fus=None,constituents=None):
        accessor = 'scenario.Network.ConstituentsManagement.Elements' + \
                    '.OfType[CatchmentElementConstituentData]()' + \
                    '.*FunctionalUnitData'

        if not fus is None:
            fus = self._ironpy._stringToList(fus)
            accessor += '.Where(lambda fuData: fuData.DisplayName in %s)'%fus

        accessor += '.*ConstituentModels'

        if not constituents is None:
            constituents = self._ironpy._stringToList(constituents)
            accessor +=  '.Where(lambda x: x.Constituent.Name in %s)'%constituents

        accessor += '.*ConstituentSources.*GenerationModel.%s'%parameter
        return accessor

    def get_models(self,catchments=None,fus=None,constituents=None):
        return self.get_param_values('GetType().FullName',catchments,fus,constituents)

    def get_param_values(self,parameter,catchments=None,fus=None,constituents=None):
        accessor = self._build_accessor(parameter,catchments,fus,constituents)
        return self._ironpy.get(accessor,self._ns)

    def set_param_values(self,parameter,values,catchments=None,fus=None,constituents=None,literal=False,fromList=False):
        accessor = self._build_accessor(parameter,catchments,fus,constituents)

        return self._ironpy.set(accessor,values,self._ns,literal,fromList)


class SearchableList(object):
    def __init__(self,the_list):
        self._list = the_list

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

    def __getattr__(self,name):
        PREFIX='find_by_'
        if name.startswith(PREFIX):
            field_name = name[len(PREFIX):]
            return lambda x: list(filter(lambda y: y[field_name]==x,self._list))

        raise AttributeError(attr + ' not allowed')

