
from .utils import _stringToList

class VeneerIronPython(object):
    """
    Helper functions for manipulating the internals of the Source model itself.

    These features rely on 'Allow Scripting' being enabled in Veneer in order to
    post custom IronPython scripts to Source.

    Specific helpers for querying and modifying the catchment components and instream components exist under
    .catchment and .link, respectively.

    eg

    v = Veneer()
    v.model.catchment?
    v.model.link?
    """
    def __init__(self,veneer):
        self._veneer = veneer
        self.ui = VeneerSourceUIHelpers(self)
        self._generator = VeneerScriptGenerators(self)
        self.catchment = VeneerCatchmentActions(self)
        self.link = VeneerLinkActions(self)
        self.node = VeneerNodeActions(self)

    def _initScript(self,namespace=None):
        script = "# Generated Script\n"
        if not namespace is None:
            namespace = _stringToList(namespace)

            script += '\n'.join(["import %s\n"%ns for ns in namespace])
        script += "import clr\n"
        script += "import System\n"
        script += "import FlowMatters.Source.Veneer.RemoteScripting.ScriptHelpers as H\n"
        script += "clr.ImportExtensions(System.Linq)\n"
        return script

    def clean_script(self,script):
        indent = len(script)-len(script.lstrip())
        if indent>0:
            lines = [l[(indent-1):] for l in script.splitlines()]
            return '\n'.join(lines)
        return script

    def runScript(self,script,async=False):
        return self.run_script(script,async)

    def run_script(self,script,async=False):
        script = self.clean_script(script)
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

    def find_model_type(self,model_type,must_be_model=True):
        '''
        Search for model types matching a given string pattern

        eg

        v.model.find_model_type('emc')
        '''
        script = self._initScript('TIME.Management.Finder as Finder')
        if must_be_model:
            script += 'import TIME.Core.Model as Model\n'
            script += 'f = Finder(Model)\n'
        else:
            script += 'import System.Object as Object\n'
            script += 'f = Finder(Object)\n'
        script += 'types = f.types()\n'
        script += 's = "%s"\n'%model_type
        script += 'result = types.Where(lambda t:t.Name.ToLower().Contains(s.ToLower()))'
        script += '.Select(lambda tt:tt.FullName)\n'
        res = self._safe_run(script)
        return [v['Value'] for v in res['Response']['Value']]

    def _find_members_with_attribute_in_type(self,model_type,attribute):
        script = self._initScript(model_type)
        if attribute:
            script += 'from TIME.Core.Metadata import %s\n'%attribute
        script += 'result = []\n'
        script += 'tmp = %s()\n'%model_type
        script += 'typeObject = tmp.GetType()\n'
        script += 'for member in dir(tmp):\n'
        script += '  try:\n'
        if attribute:
            script += '    if typeObject.GetMember(member)[0].IsDefined(%s,True):\n'%attribute
            script += '      result.append(member)\n'
        else:
            script += '    result.append(member)\n'
        script += '  except: pass'
        res = self._safe_run(script)
        return [v['Value'] for v in res['Response']['Value']]

    def _find_members_with_attribute_for_types(self,model_types,attribute=None):
        model_types = list(set(_stringToList(model_types)))
        result = {}
        for t in model_types:
            result[t] = self._find_members_with_attribute_in_type(t,attribute)
        if len(model_types)==1:
            return result[model_types[0]]
        return result

    def _find_fields_and_properties_for_type(self,model_type):
        script = self._initScript(model_type)
        script += 'from System.Reflection import PropertyInfo,FieldInfo\n'
        script += 'result = []\n'
        script += 'tmp = %s()\n'%model_type
        script += 'typeObject = tmp.GetType()\n'
        script += 'for member in dir(tmp):\n'
        script += '  try:\n'
        script += '    member_info = typeObject.GetMember(member)[0]\n'
        script += '    if isinstance(member_info,PropertyInfo):\n'
        script += '      result.append("%s %s"%(member,member_info.PropertyType.Name))\n'
        script += '    if isinstance(member_info,FieldInfo):\n'
        script += '      result.append("%s %s"%(member,member_info.FieldType.Name))\n'
        script += '  except: pass'
        res = self._safe_run(script)
        return dict([v['Value'].split(' ') for v in res['Response']['Value']])

    def find_properties(self,model_types):
        """
        Find all fields and properties for a given model type or list of model types

        Returns:
         * A list of fields and properties (if a single model type is provided)
         * A dictionary model type name -> list of fields and properties (if more than model type)
        """
        model_types = list(set(_stringToList(model_types)))
        result = {}
        for t in model_types:
            result[t] = self._find_fields_and_properties_for_type(t)
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
        data = resp['Response']['Value'] if resp['Response'] else resp['Response']
        if listQuery:
            return [d['Value'] if d else d for d in data]
        return data

    def get_data_sources(self,theThing,namespace=None):
        '''
        Get references (Veneer URLs) to the 
        '''
        script = self._initScript(namespace)
        script += ''
        listQuery = theThing.find(".*") != -1
        if listQuery:
            script += 'result = []\n'
            innerLoop = "ignoreExceptions=False\n"
            innerLoop += 'result.append(H.FindDataSource(scenario,%s__init__.__self__,"%s"))'
            script += self._generateLoop(theThing,innerLoop)
        else:
            obj = '.'.join(theThing.split('.')[0:-1])
            prop = theThing.split('.')[-1]
            script += "result = H.FindDataSourcer(%s,%s)\n"%(obj,prop)
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
                           data_group=None,namespace=None):
        assignment = "H.AssignTimeSeries(scenario,%s__init__.__self__,'%s','"+data_group+"',newVal"
        post_assignment = ",%d)"%column
        theValue = [fn.replace('\\','/') for fn in _stringToList(theValue)]
        if len(theValue)==1:
            theValue=theValue[0]
        return self._assignment(theThing,theValue,namespace,literal,from_list,False,assignment,post_assignment)

    def call(self,theThing,namespace=None):
        return self.get(theThing,namespace)

    def sourceScenarioOptions(self,optionType,option=None,newVal = None):
        return self.source_scenario_options(optionType,option,newVal)

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
        s += 'H.EnsureElementsHaveConstituentProviders(scenario)\n'
#        s += 'nw = scenario.Network\n'
#        s += 'nw.ConstituentsManagement.Reset(scenario.CurrentConfiguration.StartDate)\n'
        return self._safe_run(s)

    def running_configuration(self,new_value=None,return_all=False):
        '''
        Set or get the current running configuration for the Source model (eg Single Simulation), or
        get a list of all available running configurations

        eg:

        current = v.model.running_configuration()

        v.model.running_configuration('Single Simulation')

        all_available = v.model.running_configuration(return_all=True)
        '''
        collection = 'scenario.RunManager.Configurations'
        if new_value is None:
            if return_all:
                script = 'result = [conf.Name for conf in %s]'%collection
            else:
                script = 'result = scenario.RunManager.CurrentConfiguration.Name'
        else:
            script = 'scenario.RunManager.CurrentConfiguration = [conf for conf in %s if conf.Name.lower()=="%s".lower()][0]'
            script = script%(collection,new_value)
        return self._safe_run(script)

    def save(self,fn=None):
        if fn:
            fn = "'%s'"%fn
        else:
            fn = 'ph.ProjectMetaStructure.OutputFile'
        script = '''
                 from RiverSystem.ApplicationLayer.Consumers import DefaultCallback
                 from RiverSystem.ApplicationLayer.Creation import ProjectHandlerFactory
                 from RiverSystem import RiverSystemProject

                 ph = project_handler
                 cb = DefaultCallback()
                 cb.OutputFileName=%s
                 saved_cb = ph.CallBackHandler
                 try:
                     ph.CallBackHandler = cb
                     ph.ProjectMetaStructure.Project = scenario.Project
                     ph.ProjectMetaStructure.OutputFile = ''
                     ph.ProjectMetaStructure.SaveProjectToFile = True
                     ph.SaveProject()
                 finally:
                     ph.CallBackHandler = saved_cb
                 '''%fn
        return self._safe_run(script)

class VeneerScriptGenerators(object):
    def __init__(self,ironpython):
        self._ironpy = ironpython

    def find_feature_by_name(self):
        script = self._ironpy._initScript()
        script += "def find_feature_by_name(searchTerm,exact=False):\n"
        script += "  for n in scenario.Network.Nodes:\n"
        script += "    if n.Name == searchTerm:\n"
        script += "      return n\n"
        script += "    if not exact and n.Name.startswith(searchTerm):\n"
        script += "      return n\n"
        script += "  for l in scenario.Network.Links:\n"
        script += "    if l.Name == searchTerm:\n"
        script += "      return l\n"
        script += "    if not exact and l.Name.startswith(searchTerm):\n"
        script += "      return l\n"
        script += "  return None\n\n"
        return script

class VeneerSourceUIHelpers(object):
    def __init__(self,ironpython):
        self._ironpy = ironpython

    def open_editor(self,name_of_element):
        script = self._ironpy._initScript(namespace="RiverSystem.Controls.Controllers.FeatureEditorController as FeatureEditorController")
        script += self._ironpy._generator.find_feature_by_name()
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
            ns += ','+(','.join(_stringToList(self._ns)))
        return ns

    def help(self,param=None,**kwargs):
        if not param:
            param = '__init__.__self__'
        return self._ironpy.sourceHelp(self._build_accessor(param,**kwargs))

    def get_models(self,by_name=False,**kwargs):
        '''
        Return the models used in a particular context
        '''
        resp = self.get_param_values('GetType().FullName',**kwargs)
        if by_name:
            return dict(zip(self.names(**kwargs),resp))
        return resp

    def get_param_values(self,parameter,by_name=False,**kwargs):
        '''
        Return the values of a particular parameter used in a particular context
        '''
        accessor = self._build_accessor(parameter,**kwargs)
        resp = self._ironpy.get(accessor,kwargs.get('namespace',self._ns))
        if by_name:
            return dict(zip(self.names(**kwargs),resp))
        return resp

    def set_models(self,models,fromList=False,**kwargs):
        '''
        Assign computation models.
        '''
        return self.set_param_values(None,models,fromList=fromList,instantiate=True,**kwargs)

    def set_param_values(self,parameter,values,literal=False,fromList=False,instantiate=False,**kwargs):
        '''
        Set the values of a particular parameter used in a particular context
        '''
        accessor = self._build_accessor(parameter,**kwargs)
        ns = self._ns
        if instantiate:
            values = _stringToList(values)
            ns = self._instantiation_namespace(values)
            fromList = True
        return self._ironpy.set(accessor,values,ns,literal=literal,fromList=fromList,instantiate=instantiate)

    def get_data_sources(self,parameter,by_name=False,**kwargs):
        '''
        Return pointers (veneer URLs) to the data sources used as input to a particular parameter
        '''
        accessor = self._build_accessor(parameter,**kwargs)
        resp = self._ironpy.get_data_sources(accessor,kwargs.get('namespace',self._ns))
        if by_name:
            return dict(zip(self.names(**kwargs),resp))
        return resp

    def names(self,**kwargs):
        '''
        Return the names of the network elements
        '''
        return self.get_param_values(self._name_accessor,**kwargs)

    def assign_time_series(self,parameter,values,data_group,column=0,
                           literal=True,fromList=False,**kwargs):
        '''
        Assign an input time series to a model input input
        '''
        accessor = self._build_accessor(parameter,**kwargs)
        return self._ironpy.assign_time_series(accessor,values,from_list=fromList,
                                               literal=literal,column=column,
                                               data_group=data_group)

    def _call(self,accessor,namespace=None):
        return self._ironpy.call(accessor,namespace)

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
            accessor += '.Where(lambda fu: fu.definition.Name in %s)'%fus

        if not parameter is None:
            accessor += '.*%s'%parameter

        return accessor

class VeneerCatchmentActions(VeneerFunctionalUnitActions):
    '''
    Helpers for querying/modifying the catchment model setup

    Specific helpers exist under:

    * .runoff       (rainfall runoff in functional units)
    * .generation   (constituent generation in functional units)
    * .subcatchment (subcatchment level models)
    '''
    def __init__(self,ironpython):
        self._ironpy = ironpython
        self.runoff = VeneerRunoffActions(self)
        self.generation = VeneerCatchmentGenerationActions(self)
        self.subcatchment = VeneerSubcatchmentActions(self)
        self._ns = None

    def _build_accessor(self,parameter=None,catchments=None):
        accessor = 'scenario.Network.Catchments'

        if not catchments is None:
            catchments = _stringToList(catchments)
            accessor += '.Where(lambda c: c.DisplayName in %s)'%catchments

        if not parameter is None:
            accessor += '.*%s'%parameter

        return accessor

    def get_areas(self,catchments=None):
        '''
        Return catchment area in square metres
        '''
        return self.get_param_values('characteristics.areaInSquareMeters',catchments=catchments)
#        return self._ironpy.get('scenario.Network.Catchments.*characteristics.areaInSquareMeters')

    def names(self,catchments=None):
        return self.get_param_values('Name',catchments=catchments)

    def get_functional_unit_areas(self,catchments=None,fus=None):
        '''
        Return the area of each functional unit in each catchment:

        Parameters:

        catchments: Restrict to particular catchments by passing a list of catchment names

        fus: Restrict to particular functional unit types by passing a list of FU names
        '''
        return self.get_param_values('areaInSquareMeters',catchments=catchments,fus=fus)

    def set_functional_unit_areas(self,values,catchments=None,fus=None):
        '''
        Set the area of each functional unit in each catchment:

        Parameters:

        values: List of functional unit areas to apply.

        catchments: Restrict to particular catchments by passing a list of catchment names

        fus: Restrict to particular functional unit types by passing a list of FU names
        '''
        return self.set_param_values('areaInSquareMeters',values,fromList=True,catchments=catchments,fus=fus)

    def get_functional_unit_types(self,catchments=None,fus=None):
        '''
        Return a list of all functional unit types in the model
        '''
        return self.get_param_values('definition.Name')

class VeneerRunoffActions(VeneerFunctionalUnitActions):
    '''
    Helpers for querying/modifying the rainfall runoff model setup

    Query options:

    * catchments - the name(s) of catchments to match when querying/configuring.

    * fus - the type(s) of functional units to match when querying/configuring
    '''
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
        '''
        Assign an input time series to a rainfall runoff input
        '''
        accessor = self._build_accessor(parameter,catchments,fus)
        return self._ironpy.assign_time_series(accessor,values,from_list=fromList,
                                               literal=literal,column=column,
                                               data_group=data_group)

class VeneerCatchmentGenerationActions(VeneerFunctionalUnitActions):
    '''
    Helpers for querying/modifying the constituent generation model setup

    Query options:

    * catchments - the name(s) of catchments to match when querying/configuring.

    * fus - the type(s) of functional units to match when querying/configuring
    '''
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
    '''
    Helpers for querying/modifying the subcatchment-level models

    Query options:

    * catchments - the name(s) of catchments to match when querying/configuring.
    '''
    def __init__(self,catchment):
        self._catchment = catchment
        self._name_accessor = 'Catchment.DisplayName'
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
        self._name_accessor = '.DisplayName'

    def create(self,from_node,to_node,name=None):
        script = self._ironpy._initScript()
        script += self._ironpy._generator.find_feature_by_name()
        script += 'n1 = find_feature_by_name("%s",exact=True)\n'%from_node
        script += 'n2 = find_feature_by_name("%s",exact=True)\n'%to_node
        script += 'result = scenario.Network.Connect(n1,n2)\n'
        if name:
            script += 'result.Name = "%s"'%name
        return self._ironpy._safe_run(script)


class VeneerNetworkElementConstituentActions(VeneerNetworkElementActions):
    def __init__(self,ironpy):
        super(VeneerNetworkElementConstituentActions,self).__init__(ironpy)
        self._aspect_pre_modifer = {
            '':'',
            'played':'Data.ConstituentPlayedValues',
            'model':'Data.ProcessingModels'
        }
        self._aspect_post_modifer = {
            'model':'.*Model',
            'played':'.*',
            '':''
        }

    def assign_time_series(self,parameter,values,data_group,column=0,
                           literal=True,fromList=False,aspect='played',**kwargs):
        '''
        Assign an input time series to a rainfall runoff input
        '''
        accessor = self._build_accessor(parameter,aspect=aspect,**kwargs)
        return self._ironpy.assign_time_series(accessor,values,from_list=fromList,
                                               literal=literal,column=column,
                                               data_group=data_group,namespace=self._ns)

    def _build_accessor(self,parameter=None,constituents=None,aspect=None,**kwargs):
        aspect = self._default_aspect if aspect is None else aspect
        accessor = 'scenario.Network.ConstituentsManagement.Elements'
        accessor += self._filter_constituent_data_types()                   
        accessor += self._filter_by_query(**kwargs)
        accessor += '.*%s'%self._aspect_pre_modifer[aspect]

        if not constituents is None:
            constituents = _stringToList(constituents)
            if accessor[-1]!='*':
                accessor +='.'
            accessor += 'Where(lambda c: c.Constituent.Name in %s)'%constituents

        accessor += self._aspect_post_modifer[aspect]

        if not parameter is None:
            if accessor[-1]!='*':
                accessor+='.'
            accessor += parameter
        return accessor

    def initialise_played_constituents(self,**kwargs):
        accessor = self._build_accessor(aspect='',**kwargs)
        script = self._ironpy._initScript(self._ns)
        script += 'from RiverSystem.Constituents.ConstituentPlayedValue import ConstituentPlayedType as ConstituentPlayedType\n'
        script += "from RiverSystem.Constituents import ConstituentPlayedValue as ConstituentPlayedValue\n"
        script += 'playType = ConstituentPlayedType.varConcentration\n'
        script += 'constituents = scenario.Network.ConstituentsManagement.Config.Constituents\n'
        script += '\n'
        script += '\n'
        script += '\n'
        innerLoop = [
            "ignoreExceptions=False",
            "the_mod = %s%sData",
            "for constituent in constituents:",
            "  if not the_mod.ConstituentPlayedValues.Any(lambda cpv: (cpv.Constituent==constituent) and (cpv.PlayedType==playType)):",
            "    new_cpv = ConstituentPlayedValue(constituent)",
            "    new_cpv.PlayedType = playType",
            "    the_mod.ConstituentPlayedValues.Add(new_cpv)"
        ]
        innerLoop = '\n'.join(innerLoop)
        script += self._ironpy._generateLoop(accessor,innerLoop,first=False)

        return self._ironpy._safe_run(script)

class VeneerLinkConstituentActions(VeneerNetworkElementConstituentActions):
    def __init__(self,link):
        self._link = link
        self._name_accessor = 'Link.DisplayName'
        super(VeneerLinkConstituentActions,self).__init__(link._ironpy)
        self._ns = 'RiverSystem.Constituents.LinkElementConstituentData as LinkElementConstituentData'
        self._default_aspect = 'model'

    def _filter_constituent_data_types(self):
        return '.OfType[LinkElementConstituentData]()'

    def _filter_by_query(self,links=None):
        if links is None:
            return ''

        links = _stringToList(links)
        return '.Where(lambda lecd: lecd.Element.DisplayName in %s)'%links

class VeneerLinkRoutingActions(VeneerNetworkElementActions):
    '''
    Queries and actions relating to streamflow routing models.

    Query options:

    * links - the name(s) of links to match when querying/configuring.

    For example:

    v.model.links.routing.get_models(links=['Link #1','Link #2'])
    '''
    def __init__(self,link):
        self._link = link
        self._name_accessor = 'link.DisplayName'
        super(VeneerLinkRoutingActions,self).__init__(link._ironpy)

    def _build_accessor(self,parameter=None,links=None):
        accessor = 'scenario.Network.Links'
        if not links is None:
            links = _stringToList(links)
            accessor += '.Where(lambda l:l.DisplayName in %s)'%links
        accessor += '.*FlowRouting'

        if not parameter is None:
            accessor += '.%s'%parameter

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

class VeneerNodeActions(VeneerNetworkElementActions):
    '''
    Queries and actions relating to nodes (incuding node models).

    Query options:

    * nodes - the name(s) of nodes to match when querying/configuring.

    * node_types - the type(s) of nodes to match when querying/configuring

    For example:

    v.model.nodes.get_models(nodes='Fish River')
    '''
    def __init__(self,ironpython):
        super(VeneerNodeActions,self).__init__(ironpython)
        self._name_accessor = 'Node.Name'
        self.constituents = VeneerNodeConstituentActions(self)

    def _refine_accessor(self,node_access='',nodes=None,node_types=None):
        accessor = ""
        if not nodes is None:
            nodes = _stringToList(nodes)
            accessor += '.Where(lambda n:n%s.Name in %s)'%(node_access,nodes)
        if not node_types is None:
            node_types = _stringToList(node_types)
            accessor += '.Where(lambda n:n%s.NodeModel and n%s.NodeModel.GetType().Name.Split(".").Last() in %s)'%(node_access,node_access,node_types)
        return accessor

    def _build_accessor(self,parameter=None,nodes=None,node_types=None):
        accessor = 'scenario.Network.Nodes'
        accessor += self._refine_accessor(nodes=nodes,node_types=node_types)

        accessor += '.*NodeModel'
        if not parameter is None:
            accessor += '.%s'%parameter

        return accessor

    def create(self,name,node_type,location=None,schematic_location=None):
        script = self._ironpy._initScript('.'.join(node_type.split('.')[:-1]))
        script += 'import RiverSystem.E2ShapeProperties as E2ShapeProperties\n'
        script += 'network = scenario.Network\n'
        script += 'new_node = RiverSystem.Node()\n'
        if location:
            script += 'new_node.location.E = %f\n'%location[0]
            script += 'new_node.location.N = %f\n'%location[1]

        script += 'new_node.Name = "%s"\n'%name
        script += 'new_node.NodeModel = %s()\n'%node_type
        script += 'network.Add.Overloads[RiverSystem.INetworkElement](new_node)\n'

        if(schematic_location):
            script += 'schematic = H.GetSchematic(scenario)\n'
            script += 'if schematic:\n'
            script += '  shp = E2ShapeProperties()\n'
            script += '  shp.Feature = new_node\n'
            script += '  shp.Location.X = %f\n'%schematic_location[0]
            script += '  shp.Location.Y = %f\n'%schematic_location[1]
            script += '  schematic.ExistingFeatureShapeProperties.Add(shp)\n'
            script += '  print("Set location")\n'
        script += 'result = new_node\n'
        return self._ironpy._safe_run(script)
        # schematic_location???

    def remove(self,name):
        script = self._ironpy._initScript('RiverSystem')
        script += self._ironpy._generator.find_feature_by_name()
        script += 'network = scenario.Network\n'
        script += 'node = find_feature_by_name("%s",exact=True)\n'%name
        script += 'if node:\n'
        script += '    network.Delete.Overloads[RiverSystem.Node](node)\n'
        return self._ironpy._safe_run(script)

class VeneerNodeConstituentActions(VeneerNetworkElementConstituentActions):
    '''
    Queries and actions relating to nodes (incuding node models).

    Query options:

    * nodes - the name(s) of nodes to match when querying/configuring.

    * node_types - the type(s) of nodes to match when querying/configuring

    For example:

    v.model.nodes.constituents.get_models(nodes='Fish River')
    '''
    def __init__(self,node):
        self._node = node
        self._name_accessor = 'Element.Name'
        super(VeneerNodeConstituentActions,self).__init__(self._node._ironpy)
        self._ns = ['RiverSystem.Node as Node',
                    'RiverSystem.Constituents.NetworkElementConstituentData as NetworkElementConstituentData']
        self._default_aspect = '' #'Data.ConstituentPlayedValues'

    def _filter_constituent_data_types(self):
        return '.OfType[NetworkElementConstituentData]().Where(lambda d: isinstance(d.Element,Node))'

    def _filter_by_query(self,**kwargs):
        return self._node._refine_accessor(node_access='.Element',**kwargs)

#    def _build_accessor(self,parameter,nodes=None,node_types=None)
#        accessor = 'scenario.Network.ConstituentsManagement.Elements' + \
#        accessor += self._node._refine_accessor(nodes,node_types)



