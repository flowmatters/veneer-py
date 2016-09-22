
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

    def _initScript(self,namespace=None):
        script = "# Generated Script\n"
        if not namespace is None:
            script += "import %s\n"%namespace
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

    def find_model_type(self,model_type):
        '''
        Search for model types matching a given string pattern

        eg

        v.model.find_model_type('emc')
        '''
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
                           data_group=None):
        ns = None
        assignment = "H.AssignTimeSeries(scenario,%s__init__.__self__,'%s','"+data_group+"',newVal"
        post_assignment = ",%d)"%column
        return self._assignment(theThing,theValue,ns,literal,from_list,False,assignment,post_assignment)

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

    def help(self,param=None,**kwargs):
        if not param:
            param = '__init__.__self__'
        return self._ironpy.sourceHelp(self._build_accessor(param,**kwargs))

    def get_models(self,**kwargs):
        '''
        Return the models used in a particular context
        '''
        return self.get_param_values('GetType().FullName',**kwargs)

    def get_param_values(self,parameter,**kwargs):
        '''
        Return the values of a particular parameter used in a particular context
        '''
        accessor = self._build_accessor(parameter,**kwargs)
        return self._ironpy.get(accessor,kwargs.get('namespace',self._ns))

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

    def get_data_sources(self,parameter,**kwargs):
        '''
        Return pointers (veneer URLs) to the data sources used as input to a particular parameter
        '''
        accessor = self._build_accessor(parameter,**kwargs)
        return self._ironpy.get_data_sources(accessor,kwargs.get('namespace',self._ns))

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

    def get_areas(self,catchments=None):
        return self._ironpy.get('scenario.Network.Catchments.*characteristics.areaInSquareMeters')

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
