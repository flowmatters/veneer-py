
from .utils import (_stringToList, _variable_safe_name,
                    _safe_filename, deprecate_async, simplify_response)
from .templates import *
from .component import VeneerComponentModelActions

import os
import pandas as pd
import json

NODE_TYPES = {
    'inflow': 'RiverSystem.Nodes.Inflow.InjectedFlow',
    'gauge': 'RiverSystem.Nodes.Gauge.GaugeNodeModel',
    'confluence': 'RiverSystem.Nodes.Confluence.ConfluenceNodeModel',
    'loss': 'RiverSystem.Nodes.Loss.LossNodeModel',
    'water_user': 'RiverSystem.Nodes.WaterUser.WaterUserNodeModel',
    'storage': 'RiverSystem.Nodes.StorageNodeModel',
    'scenario_transfer': 'RiverSystem.Nodes.ScenarioTransfer.ScenarioTransferNodeModel',
    'transfer_ownership': 'RiverSystem.Nodes.TransferOwnership.TransferOwnershipNodeModel',
    'off_allocation': 'RiverSystem.Nodes.OffAllocation.OffAllocationNodeModel',
    'environmental_demand': 'RiverSystem.Nodes.EnvironmentalDemand.EnvironmentalDemandNodeModel',
    'farm_dam': 'RiverSystem.Nodes.FarmDam.FarmDamNodeModel'
}

ADDITIONAL_INPUTS={
    'RiverSystem.Flow.StorageRouting':[
        'link.RainFall',
        'link.Evaporation',
        'link.TimeSeriesFlux'
    ],
    'RiverSystem.Nodes.StorageNodeModel':[
        'StorageInternal.EvaporationInMetresPerSecond',
        'StorageInternal.RainfallInMetresPerSecond'
    ]
}

COMMON_STORAGE_PARAMETERS=[
    'RequiredStorageLevel',
    'PassOrdersThru',
    'MinimumOperatingTargetVolume',
    'MinimumOperatingTargetLevel',
    'MaximumOperatingTargetLevelWeir',
    'MaximumOperatingTargetVolumeWeir',
    'MaximumOperatingTargetLevel',
    'MaximumOperatingTargetVolume',
    'MaximumUnregulatedOperatingTargetLevel',
    'MaximumUnregulatedOperatingTargetVolume'
]

OTHER_STORAGE_PARAMETERS=[
    'GaugedLevelSpecification',
    'StorageDetailsSpecification',
    'UseMinMaxLevels',
    'OperatingConstraintsSpecification',
    'OperatingTargetsSpecification',
]

GENERAL_LINK_PARAMETERS=[
    'link.Elevation',
    'link.Length',
    'link.ReceiveCatchmentFluxAtTop'
]

ADDITIONAL_PARAMETERS={
    'RiverSystem.Flow.StorageRouting':[
        'InitStorage',
        'InitFlow',
        'IsInitFlow',
        'AverageRegFlow',
        'Divisions',
        'TriangularWeir'
    ]+GENERAL_LINK_PARAMETERS,
    'RiverSystem.Nodes.StorageNodeModel':OTHER_STORAGE_PARAMETERS+COMMON_STORAGE_PARAMETERS,
    'RiverSystem.Nodes.Loss.LossNodeModel':[
        'LossVolume',
        'LossChoiceEnum'
    ],
    'RiverSystem.Nodes.Splitter.RegulatedEffluentPartitioner':[
        'EffluentLink.Name',
        'RegulatorOpening',
        'TreatAsRegulatedSupplyNode'
    ],
    'RiverSystem.Nodes.Gauge.GaugeNodeModel':[
        'UnaccountedDifferenceMode'
    ],
    'RiverSystem.Nodes.WeirNodeModel':COMMON_STORAGE_PARAMETERS[:]+[f'StorageInternal.{p}' for p in OTHER_STORAGE_PARAMETERS]+['ApplyUnaccountedDifference'],
    'RiverSystem.Nodes.SupplyPoint.ExtractionNodeModel':[
        'IsExtractive'
    ]
}

def _transform_node_type_name(n):
    n = n[0].upper() + n[1:]
    splits = n.split('_')
    if len(splits) > 1:
        n = ''.join([comp.capitalize() for comp in splits])
    if n.endswith('NodeModel'):
        return n
    if n.endswith('Node'):
        return n + 'Model'
    if n in ['InjectedFlow','RegulatedEffluentPartitioner']:
        return n
    return n + 'NodeModel'


#@deprecate_async
class VeneerIronPython(object):
    """
    Helper functions for manipulating the internals of the Source model itself.

    These features rely on 'Allow Scripting' being enabled in Veneer in order
    to post custom IronPython scripts to Source.

    Specific helpers for querying and modifying the catchment components and
    instream components exist under .catchment and .link, respectively.

    eg

    v = Veneer()
    v.model.catchment?
    v.model.link?
    """

    def __init__(self, veneer):
        self._veneer = veneer
        self.ui = VeneerSourceUIHelpers(self)
        self._generator = VeneerScriptGenerators(self)
        self.catchment = VeneerCatchmentActions(self)
        self.link = VeneerLinkActions(self)
        self.node = VeneerNodeActions(self)
        self.functions = VeneerFunctionActions(self)
        self.simulation = VeneerSimulationActions(self)
        self.component = VeneerComponentModelActions(self)
        self.spatial = VeneerGeographicDataActions(self)

        from .operations import VeneerOperationsActions
        self.operations = VeneerOperationsActions(self)
        self.deferred_scripts = []
        self.deferring = False

    def defer(self):
        '''
        Start deferring script execution.
        '''
        self.deferring = True

    def flush(self):
        '''
        Run all deferred scripts, and stop deferring script execution
        '''
        self.deferring = False
        mega_script = '\n'.join(self.deferred_scripts)
        self.deferred_scripts = []
        if not len(mega_script):
            return
        return self._safe_run(mega_script)

    def _init_script(self, namespace=None):
        script = "# Generated Script\n"
        if not namespace is None:
            namespace = _stringToList(namespace)

            script += '\n'.join(["import %s\n" % ns for ns in namespace])
        script += "import clr\n"
        script += "clr.AddReference('System.Core')\n"
        script += "import System\n"
        script += "import FlowMatters.Source.Veneer.RemoteScripting.ScriptHelpers as H\n"
        script += "clr.ImportExtensions(System.Linq)\n"
        return script

    def clean_script(self, script):
        indent = len(script) - len(script.lstrip())
        if indent > 0:
            lines = [l[(indent - 1):] for l in script.splitlines()]
            return '\n'.join(lines)
        return script

    def runScript(self, script, run_async=False):
        return self.run_script(script, run_async)

    def run_script(self, script, run_async=False, init=False):
        if init:
            script = self._init_script() + '\n' + script
        script = self.clean_script(script)
        if self.deferring:
            self.deferred_scripts.append(script)
            return None

        return self._veneer.run_server_side_script(script, run_async)

    def sourceHelp(self, theThing='scenario', namespace=None):
        """
        Get some help on what you can do with theThing,
        where theThing is something you can access from a scenario, or the
        scenario itself.

        If theThing is a method, returns details on how to call the method
        If theThing is an object, returns a list of public methods and
        properties

        eg
        v.model.sourceHelp() # returns help on the scenario object
        v.model.sourceHelp('scenario.CurrentConfiguration')
        """
        # theThing = theThing.replace('.*','.First().')
        script = self._init_script(namespace)
        innerLoop = "theThing = %s%s\n"
        innerLoop += "if hasattr(theThing, '__call__'):\n"
        innerLoop += "    result = 'function'\n"
        innerLoop += "    help(theThing)\n"
        innerLoop += "else:\n"
        innerLoop += "    result = dir(theThing)"
        script += self._generateLoop(theThing, innerLoop, first=True)
        data = self.run_script(script)
        if not data['Exception'] is None:
            raise Exception(data['Exception'])
        if data['Response'] is None:
            raise Exception('Could not find anything matching %s' % theThing)
        if data['Response']['Value'] == 'function':
            print(data['StandardOut'])
        else:
            return [d['Value'] for d in data['Response']['Value']]

    def _generateLoop(self, theThing, innerLoop, first=False, names=None,inner_exception=''):
        script = ''
        script += "have_succeeded = False\n"
        script += "ignoreExceptions = True\n"
        indent = 0
        indentText = ''
        levels = theThing.split('.*')
        prevLoop = ''
        for level in levels[0:-1]:
            loopVar = "i_%d" % indent
            script += indentText + \
                'for %s in %s%s:\n' % (loopVar, prevLoop, level)
            indent += 1
            indentText = ' ' * (indent * 4)
            if names is not None:
                script += indentText + \
                    '%s = %s\n' % (names[indent // 2], loopVar)
            script += indentText + 'try:\n'
            indent += 1
            indentText = ' ' * (indent * 4)
            prevLoop = loopVar + '.'
        script += indentText
        substCount = innerLoop.count('%s') // 2
        script += innerLoop.replace('\n', '\n' +
                                    indentText) % tuple([prevLoop, levels[-1]]
                                                        * substCount)
        script += '\n'
        script += indentText + "have_succeeded = True\n"
        inner_most = True
        while indent > 0:
            indent -= 1
            indentText = ' ' * (indent * 4)
            script += indentText + "except:\n"
            if inner_most:
                script += indentText + '  %s\n'%inner_exception
                inner_most = False
            script += indentText + '  if not ignoreExceptions: raise\n'
            if first:
                script += indentText + "if have_succeeded: break\n"
            indent -= 1

        return script

    def nav(self):
        from .navigate import Queryable
        return Queryable(self._veneer)

    def find_model_type(self, model_type, must_be_model=True):
        '''
        Search for model types matching a given string pattern

        eg

        v.model.find_model_type('emc')
        '''
        script = self._init_script()
        script += 'try:\n'
        script += '  import TIME.Management.Finder as Finder\n'
        script += 'except:\n'
        script += '  import TIME.Management.AssemblyManager as AssemblyManager\n'
        script += '  def Finder(t):\n'
        script += '    class _result: pass\n'
        script += '    __result = _result()\n'
        script += '    def tmpF(): return AssemblyManager.FindTypes(t).ToList()\n'
        script += '    __result.types = tmpF\n'
        script += '    return __result\n'

        if must_be_model:
            script += 'import TIME.Core.Model as Model\n'
            script += 'f = Finder(Model)\n'
        else:
            script += 'import System.Object as Object\n'
            script += 'f = Finder(Object)\n'
        script += 'types = f.types()\n'
        script += 's = "%s"\n' % model_type
        script += 'result = types.Where(lambda t:t.Name.ToLower().Contains(s.ToLower()))'
        script += '.Select(lambda tt:tt.FullName)\n'
        res = self._safe_run(script)
        return [v['Value'] for v in res['Response']['Value']]

    def expand_model(self, model_type):
        if '.' in model_type:
            return model_type

        results = self.find_model_type(model_type)
        if len(results) == 0:
            raise Exception('No model matching: %s' % model_type)
        if len(results) > 1:
            raise Exception('Multiple models matching:%s - %s' %
                            (model_type, str(results)))
        return results[0]

    def _find_members_with_attribute_in_type(self, model_type, attribute):
        script = self._init_script(model_type)
        if attribute:
            script += 'from TIME.Core.Metadata import %s\n' % attribute
        script += 'result = []\n'
        script += 'tmp = %s()\n' % model_type
        script += 'typeObject = tmp.GetType()\n'
        script += 'for member in dir(tmp):\n'
        script += '  try:\n'
        if attribute:
            script += '    if typeObject.GetMember(member)[0].IsDefined(%s,True):\n' % attribute
            script += '      result.append(member)\n'
        else:
            script += '    result.append(member)\n'
        script += '  except: pass'
        res = self._safe_run(script)
        return [v['Value'] for v in res['Response']['Value']]

    def _find_members_with_attribute_for_types(self, model_types,
                                               attribute=None,
                                               additional={}):
        model_types = list(set(_stringToList(model_types)))
        result = {}
        for t in model_types:
            result[t] = self._find_members_with_attribute_in_type(t, attribute)
            if t in additional:
                result[t] = list(set(result[t]).union(additional[t]))
        if len(model_types) == 1:
            return result[model_types[0]]
        return result

    def _find_fields_and_properties_for_type(self, model_type):
        script = self._init_script(model_type)
        script += 'from System.Reflection import PropertyInfo,FieldInfo\n'
        script += 'result = []\n'
        script += 'tmp = %s()\n' % model_type
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

    def find_properties(self, model_types,additional={}):
        """
        Find all fields and properties for a given model type or list of
        model types

        Returns:
         * A list of fields and properties (if a single model type is provided)
         * A dictionary model type name -> list of fields and properties
           (if more than model type)
        """
        model_types = list(set(_stringToList(model_types)))
        result = {}
        for t in model_types:
            result[t] = self._find_fields_and_properties_for_type(t)
            if t in additional:
                result[t] = result[t].union(additional[t])
        if len(model_types) == 1:
            return result[model_types[0]]
        return result

    def find_parameters(self, model_types):
        """
        Find the parameter names for a given model type or list of model types

        Returns:
         * A list of parameters (if a single model type is provided)
         * A dictionary model type name -> list of parameters
           (if more than model type)
        """
        return self._find_members_with_attribute_for_types(model_types,
                                                           'ParameterAttribute',
                                                           ADDITIONAL_PARAMETERS)

    def find_inputs(self, model_types):
        """
        Find the input names for a given model type or list of model types

        Returns:
         * A list of inputs (if a single model type is provided)
         * A dictionary model type name -> list of inputs
           (if more than model type)
        """
        return self._find_members_with_attribute_for_types(model_types,
                                                            'InputAttribute',
                                                            ADDITIONAL_INPUTS)

    def find_states(self, model_types):
        """
        Find the state names for a given model type or list of model types

        Returns:
         * A list of states (if a single model type is provided)
         * A dictionary model type name -> list of states
           (if more than model type)
        """
        return self._find_members_with_attribute_for_types(model_types,
                                                           'StateAttribute')

    def find_outputs(self, model_types):
        """
        Find the output names for a given model type or list of model types

        Returns:
         * A list of outputs (if a single model type is provided)
         * A dictionary model type name -> list of outputs
           (if more than model type)
        """
        return self._find_members_with_attribute_for_types(model_types,
                                                           'OutputAttribute')

    def find_default_parameters(self, model_type):
        """
        Find the default value of all parameters for a given model type.

        Returns:
         * A list of parameters (if a single model type is provided)
         * A dictionary model type name -> list of parameters
           (if more than model type)
        """
        params = self.find_parameters(model_type)
        script = """
        import %s
        model = %s()
        result = {}
        """ % (model_type, model_type)
        script = self.clean_script(script)
        for p in params:
            script += '\ntry: result["%s"]=model.%s\nexcept: pass' % (p, p)
        return self.simplify_response(self._safe_run(script)['Response'])

    def simplify_response(self, response):
        return simplify_response(response)

    def get(self, theThing, namespace=None, names=None, alt_expression=None,skip_nulls=True):
        """
        Retrieve a value, or list of values from Source using theThing as a query string.

        Query should either start with `scenario` OR from a class imported using `namespace`
        """
        script = self._init_script(namespace)
        listQuery = theThing.find(".*") != -1
        if listQuery:
            script += 'result = []\n'
            if alt_expression is None:
                innerLoop = 'result.append(%s%s)'
            else:
                innerLoop = 'result.append(%s)' % alt_expression

            inner_exception = ''
            if not skip_nulls:
                inner_exception = 'result.append(None)'

            script += self._generateLoop(theThing, innerLoop, inner_exception=inner_exception, names=names)
        else:
            script += "result = %s\n" % theThing
#       return script

        resp = self.run_script(script)
        if not resp['Exception'] is None:
            raise Exception(resp['Exception'])
        data = resp['Response']['Value'] if resp['Response'] else resp['Response']
        if listQuery:
            return [self.simplify_response(d) for d in data]
        return data

    def get_data_sources(self, theThing, namespace=None,input_set=None):
        '''
        Get references (Veneer URLs) to the
        '''
        script = self._init_script(namespace)
        script += ''
        listQuery = theThing.find(".*") != -1
        if input_set is None:
            input_set = 'None'
        else:
            input_set = "'%s'" % input_set

        if listQuery:
            script += 'result = []\n'
            innerLoop = f'result.append(H.FindDataSource(scenario,%s__init__.__self__,"%s",{input_set}))'
            script += self._generateLoop(theThing, innerLoop)
        else:
            obj = '.'.join(theThing.split('.')[0:-1])
            prop = theThing.split('.')[-1]
            script += "result = H.FindDataSource(%s,%s,%s)\n" % (obj, prop,input_set)
#       return script

        resp = self.run_script(script)
        if not resp['Exception'] is None:
            raise Exception(resp['Exception'])
        data = resp['Response']['Value']
        if listQuery:
            return [d['Value'] for d in data]
        return data

    def get_functions(self, theThing, namespace=None):
        '''
        Get function names related to theThing
        '''
        script = self._init_script(namespace)
        script += ''
        listQuery = theThing.find(".*") != -1
        if listQuery:
            script += 'result = []\n'
            innerLoop = 'result.append(H.FindFunction(scenario,%s__init__.__self__,"%s"))'
            script += self._generateLoop(theThing, innerLoop)
        else:
            obj = '.'.join(theThing.split('.')[0:-1])
            prop = theThing.split('.')[-1]
            script += "result = H.FindFunction(%s,%s)\n" % (obj, prop)
#       return script

        resp = self.run_script(script)
        if not resp['Exception'] is None:
            raise Exception(resp['Exception'])
        data = resp['Response']['Value']
        if listQuery:
            return [None if d is None else d['Value'] for d in data]
        return data

    def _assignment(self, theThing, theValue, namespace=None, literal=False,
                    fromList=False, instantiate=False,
                    assignment="", post_assignment="",
                    print_script=False, dry_run=False):
        val_transform = '()' if instantiate else ''
        if literal and isinstance(theValue, str):
            theValue = "'" + theValue + "'"
        if fromList:
            if literal:
                theValue = [("'" + v + "'") if isinstance(v, str)
                            else v for v in theValue]
            theValue = '[' + (','.join([str(s) for s in theValue])) + ']'
        elif type(theValue) == list:
            theValue = 'tuple(%s)' % theValue

        script = self._init_script(namespace)
        script += 'origNewVal = %s\n' % theValue
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
            innerLoop += '.pop()' + val_transform + post_assignment + '\n'
            innerLoop += 'if len(newVal)==0: newVal = origNewVal[:]\n'
        else:
            innerLoop += val_transform + post_assignment
        script += self._generateLoop(theThing, innerLoop)
        script += 'result = have_succeeded\n'
#       return script
#        return None
        if dry_run:
            return script

        result = self.run_script(script)
        if result is None:
            return
        if not result['Exception'] is None:
            raise Exception(result['Exception'])
        return result['Response']['Value']

    def set(self, theThing, theValue, namespace=None, literal=False,
            fromList=False, instantiate=False,dry_run=False):
        return self._assignment(theThing, theValue, namespace, literal, fromList, instantiate, "%s%s = newVal", "",
                                dry_run=dry_run)

    def add_to_list(self, theThing, theValue, namespace=None, literal=False,
                    fromList=False, instantiate=False, allow_duplicates=False, n=1):
        if allow_duplicates:
            assignment = "for addCounter in range(%d):" % n
            assignment += "%s%s.Add(newVal"
        else:
            assignment = "theList=%s%s\nif not H.ListContainsInstance(theList,newVal"
            if instantiate:
                assignment += "()"
            assignment += "): theList.Add(newVal"

        return self._assignment(theThing, theValue, namespace, literal, fromList, instantiate, assignment, ")")

    def assign_time_series(self, theThing, theValue, column=0, from_list=False, literal=True,
                           data_group=None, namespace=None,dry_run=False):
        assignment = "H.AssignTimeSeries(scenario,%s__init__.__self__,'%s','" + \
            data_group + "',newVal"
        post_assignment = ",%d)" % column
        theValue = [_safe_filename(fn) for fn in _stringToList(theValue)]
        if len(theValue) == 1:
            theValue = theValue[0]
        return self._assignment(theThing, theValue, namespace, literal, from_list,
                                False, assignment, post_assignment,dry_run=dry_run)

    def clear_time_series(self,theThing,namespace=None):
        return self._assignment(theThing,None,namespace,False,False,False,CLEAR_TIMESERIES_LOOP)

    def call(self, theThing, parameter_tuple=None, literal=False, from_list=False, namespace=None):
        return self.get(theThing, namespace)

    def apply(self, accessor, code, name, init, namespace):
        script = self._init_script(namespace)
        if init:
            script += 'result = %s\n' % str(init)

        inner_loop = name + '= %s%s\n' + code
        script += self._generateLoop(accessor, inner_loop)
        if not init:
            script += 'result = have_succeeded\n'

        result = self.run_script(script)
        if not result['Exception'] is None:
            raise Exception(result['Exception'])
#        data = result['Response']['Value'] if result['Response'] else result['Response']
        return self.simplify_response(result['Response'])

    def sourceScenarioOptions(self, optionType, option=None, newVal=None):
        return self.source_scenario_options(optionType, option, newVal)

    def source_scenario_options(self, optionType, option=None, newVal=None):
        script = self._init_script(
            'RiverSystem.ScenarioConfiguration.%s as %s' % (optionType, optionType))
        retrieve = "scenario.GetScenarioConfiguration[%s]()" % optionType
        if option is None:
            script += "result = dir(%s)\n" % retrieve
        else:
            if not newVal is None:
                script += "%s.%s = %s\n" % (retrieve, option, newVal)
            script += "result = %s.%s\n" % (retrieve, option)
        res = self._safe_run(script)
        if newVal is None:
            return res

    def _safe_run(self, script):
        result = self.run_script(script)
        if not result['Exception'] is None:
            raise Exception(result['Exception'])
        return result

    def get_constituents(self):
        s = self._init_script()
        s += 'result = scenario.SystemConfiguration.Constituents.Select(lambda c: c.Name)\n'
        return self.simplify_response(self._safe_run(s)['Response'])

    def add_constituent(self, new_constituent):
        s = self._init_script(
            namespace='RiverSystem.Constituents.Constituent as Constituent')
        s += 'scenario.Network.ConstituentsManagement.Config.ProcessConstituents = True\n'
        s += 'theList = scenario.SystemConfiguration.Constituents\n'
        s += 'if not theList.Any(lambda c: c.Name=="%s"):\n' % new_constituent
        s += '  c=Constituent("%s")\n' % new_constituent
        s += '  theList.Add(c)\n'
        s += '  H.InitialiseModelsForConstituent(scenario,c)\n'
        s += 'H.EnsureElementsHaveConstituentProviders(scenario)\n'
        return self._safe_run(s)

    def get_constituent_sources(self):
        s = self._init_script()
        s += 'result = scenario.SystemConfiguration.ConstituentSources.Select(lambda c: c.Name)\n'
        return self.simplify_response(self._safe_run(s)['Response'])

    def add_constituent_source(self, new_source):
        s = self._init_script(
            namespace='RiverSystem.Catchments.Constituents.ConstituentSource as ConstituentSource')
        s += 'scenario.Network.ConstituentsManagement.Config.ProcessConstituents = True\n'
        s += 'theList = scenario.SystemConfiguration.ConstituentSources\n'
        s += 'if not theList.Any(lambda cs: cs.Name=="%s"):\n' % new_source
        s += '  cs=ConstituentSource("%s")\n' % new_source
        s += '  theList.Add(cs)\n'
        s += '  H.InitialiseModelsForConstituentSource(scenario)\n'
        s += 'H.EnsureElementsHaveConstituentProviders(scenario)\n'
        return self._safe_run(s)

    def running_configuration(self, new_value=None, return_all=False):
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
                script = 'result = [conf.Name for conf in %s]' % collection
            else:
                script = 'result = scenario.RunManager.CurrentConfiguration.Name'
        else:
            script = 'scenario.RunManager.CurrentConfiguration = [conf for conf in %s if conf.Name.lower()=="%s".lower()][0]'
            script = script % (collection, new_value)
        return self._safe_run(script)

    def save(self, fn=None):
        '''
        Save the current *project* to disk.

        fn - filename to save to. Should include .rsproj extension. If None, save using current filename
        '''
        if fn:
            fn = "'%s'" % os.path.abspath(fn).replace('\\', '\\\\')
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
                 ''' % fn
        return self._safe_run(script)


class VeneerScriptGenerators(object):
    def __init__(self, ironpython):
        self._ironpy = ironpython

    def find_feature_by_name(self):
        script = self._ironpy._init_script()
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
    def __init__(self, ironpython):
        self._ironpy = ironpython

    def open_editor(self, name_of_element):
        script = self._ironpy._init_script(
            namespace="RiverSystem.Controls.Controllers.FeatureEditorController as FeatureEditorController")
        script += self._ironpy._generator.find_feature_by_name()
        script += "f = find_feature_by_name('%s')\n" % name_of_element
        script += "if not f is None:\n"
        script += "  ctrl = FeatureEditorController(f)\n"
        script += "  ctrl.Initialise(scenario)\n"
        script += "  ctrl.Show(None)\n"

        self._ironpy._safe_run(script)

    def get_application_logs(self):
        script='''
        import TIME.Management.Log as Log
        messages = Log.GetAllMessages()
        result = [m.Message for m in messages]
        '''
        logs = [val['Value'] for val in self.runScript(script)['Response']['Value']]
        return logs

class VeneerNetworkElementActions(object):
    def __init__(self, ironpython):
        self._ironpy = ironpython
        self._ns = None
        self._pvr_element_name = None
        self._build_pvr_accessor = self._build_accessor
        self._pvr_attribute_prefix = ''
        self._aliases = {}
        self.name_columns = ['NetworkElement']
        self.query_params = []

    def _instantiation_namespace(self, types, enum=False):
        if enum:
            types = [t.split('+')[0] if '+' in t else '.'.join(t.split('.')[:-1]) for t in types]
        ns = ','.join(set(types))
        if not self._ns is None:
            ns += ',' + (','.join(_stringToList(self._ns)))
        return ns

    def _translate_property(self,prop):
        return self._aliases.get(prop,prop)

    def help(self, param=None, **kwargs):
        if not param:
            param = '__init__.__self__'
        return self._ironpy.sourceHelp(self._build_accessor(param, **kwargs))

    def nav_first(self, **kwargs):
        '''
        Get a Queryable object for the first match of the query parameters.

        Queryable object can be used for traditional object navigation.

        See https://github.com/flowmatters/veneer-py/blob/master/doc/examples/navigation/0-Introduction.ipynb
        '''
        from .navigate import Queryable
        accessor = self._build_accessor(
            None, **kwargs).replace('.*', '.First().')
        return Queryable(self._ironpy._veneer, path=accessor, namespace=self._ns)

    def get_models(self, by_name=False, skip_nulls=True, **kwargs):
        '''
        Return the models used in a particular context
        '''
        resp = self.get_param_values('GetType().FullName', skip_nulls=skip_nulls, **kwargs)
        if by_name:
            return dict(zip(self.names(**kwargs), resp))
        return resp

    def get_param_values(self, parameter, by_name=False, skip_nulls=True, **kwargs):
        '''
        Return the values of a particular parameter used in a particular context
        '''
        parameter = self._translate_property(parameter)
        accessor = self._build_accessor(parameter, **kwargs)
        resp = self._ironpy.get(accessor, kwargs.get('namespace', self._ns),skip_nulls=skip_nulls)
        if by_name:
            return dict(zip(self.names(**kwargs), resp))
        return resp

    def set_models(self, models, fromList=False, **kwargs):
        '''
        Assign computation models.
        '''
        return self.set_param_values(None, models, fromList=fromList, instantiate=True, **kwargs)

    def set_param_values(self, parameter, values,
                         literal=False, fromList=False, instantiate=False,
                         enum=False, dry_run=False,**kwargs):
        '''
        Set the values of a particular parameter used in a particular context
        '''
        parameter = self._translate_property(parameter)
        accessor = self._build_accessor(parameter, **kwargs)
        ns = self._ns
        if instantiate or enum:
            values = _stringToList(values)
            ns = self._instantiation_namespace(values, enum)
            fromList = True
            values = [v.replace('+','.') for v in values]
        return self._ironpy.set(accessor, values, ns, literal=literal, fromList=fromList,
                                instantiate=instantiate,dry_run=dry_run)

    def add_to_list(self, parameter, value, instantiate=False, n=1, **kwargs):
        parameter = self._translate_property(parameter)
        accessor = self._build_accessor(parameter, **kwargs)
        ns = value if instantiate else None
        return self._ironpy.add_to_list(accessor, value, namespace=ns, n=n, instantiate=True, allow_duplicates=True)

    def get_data_sources(self, parameter, by_name=False, input_set=None, **kwargs):
        '''
        Return pointers (veneer URLs) to the data sources used as input to a particular parameter
        '''
        parameter = self._translate_property(parameter)
        accessor = self._build_accessor(parameter, **kwargs)
        resp = self._ironpy.get_data_sources(
            accessor, kwargs.get('namespace', self._ns),input_set)
        if by_name:
            return dict(zip(self.names(**kwargs), resp))
        return resp

    def get_functions(self, parameter, by_name=False, **kwargs):
        '''
        Returns names of functions applied to a particular parameter
        '''
        parameter = self._translate_property(parameter)
        accessor = self._build_accessor(parameter, **kwargs)
        resp = self._ironpy.get_functions(
            accessor, kwargs.get('namespace', self._ns))
        if by_name:
            return dict(zip(self.names(**kwargs), resp))
        return resp

    def names(self, **kwargs):
        '''
        Return the names of the network elements
        '''
        return self.get_param_values(self._name_accessor, **kwargs)

    def enumerate_names(self, **kwargs):
        '''
        Enumerate the names of the matching network elements as tuples of name components
        '''
        return [(n,) for n in self.names(**kwargs)]

    def assign_time_series(self, parameter, values, data_group, column=0,
                           literal=True, fromList=False, **kwargs):
        '''
        Assign an input time series to a model input input
        '''
        parameter = self._translate_property(parameter)
        accessor = self._build_accessor(parameter, **kwargs)
        return self._ironpy.assign_time_series(accessor, values, from_list=fromList,
                                               literal=literal, column=column,
                                               data_group=data_group,
                                               namespace=self._ns)

    def clear_time_series(self, parameter, **kwargs):
        '''
        Assign an input time series to a model input input
        '''
        parameter = self._translate_property(parameter)
        accessor = self._build_accessor(parameter, **kwargs)
        return self._ironpy.clear_time_series(accessor,namespace=self._ns)

    def create_modelled_variable(self, parameter, element_name=None, variable_name='', **kwargs):
        '''
        Create a modelled variable for accessing a model's properties from a function.

        **NOTE:** DIFFERENT BEHAVIOUR. In contrast to many other functions, the parameter here should be named as they appear in the Source user interface.
        (Where in most other functions here, parameter is the name as it appears in code). For example, 'Rainfall' is used HERE to indicate the Rainfall input of
        a rainfall runoff model, but, 'rainfall' is used in assign_time_series.

        See potential_modelled_variables for a list of parameter names applicable to your query (kwargs)
        '''
        accessor = self._build_accessor(parameter, **kwargs)
        names = ['$' + _variable_safe_name(parameter + '_' + '_'.join(name_tuple))
                 for name_tuple in self.enumerate_names(**kwargs)]
        ns = 'RiverSystem.Functions.Variables.ModelledVariable as ModelledVariable'
        init = '{"created":[],"failed":[]}\n'
        init += BUILD_PVR_LOOKUP
        init += VALID_IDENTIFIER_FN
        init += 'orig_names=%s\n' % names
        init += 'names=orig_names[::-1]\n'
        if len(variable_name) and not variable_name.startswith('$'):
            variable_name = '$'+variable_name
        accessor = self._build_pvr_accessor('__init__.__self__', **kwargs)
        if not element_name:
            element_name = self._pvr_element_name

        if element_name is None:
            element_name = parameter

        code = CREATED_MODELLED_VARIABLE % (
            element_name, self._pvr_attribute_prefix + parameter, variable_name)
        return self._ironpy.apply(accessor, code, 'target', init, ns)
        # create accessor
        # loop through, generate name, create model variable
        # create modelled variable:
        # * object
        # * name
        # * projectviewrow
        # * add...

    def potential_modelled_variables(self, **kwargs):
        '''
        Find a list of potential properties that can be used to as modelled variables given the current query (kwargs)

        Returns a list of 2-element tuples (element_name, parameter_name).

        These values can be used with create_modelled_variable for the same query (kwargs)
        '''
        accessor = self._build_pvr_accessor('__init__.__self__', **kwargs)
        init = "{}\n"
        init += BUILD_PVR_LOOKUP

        code = FIND_MODELLED_VARIABLE_TARGETS
        lookup = self._ironpy.apply(accessor, code, 'target', init, None)
        result = []
        for k, vals in lookup.items():
            vals = list(set(vals))
            for val in vals:
                result.append((k, val))
        return result
#        return {k:list(set(v)) for k,v in result.items()}

    def enum_pvrs(self, **kwargs):
        '''
        List of all recordable attributes matching a given query.

        Information can be used with v.configure_recording and when setting up modelled variables.
        '''
        accessor = self._build_pvr_accessor('__init__.__self__ ', **kwargs)
        init = '[]\n'
        init += BUILD_PVR_LOOKUP

        code = ENUM_PVRS % tuple([self._pvr_element_name] * 2)
        return self._ironpy.apply(accessor[:-1], code, 'target', init, None)

    def apply_function(self, parameter, functions, **kwargs):
        '''
        Apply a function, from the function manager to a given model parameter or input
        '''
        parameter = self._translate_property(parameter)
        functions = _stringToList(functions)
        init = '{"success":0,"fail":0}\n'
        init += APPLY_FUNCTION_INIT % functions
        param_path = parameter.split('.')
        parameter = param_path[-1]
        param_path = '.'.join(param_path[:-1])
        if len(param_path):
            param_path = '.'+param_path
        code = APPLY_FUNCTION_LOOP % (param_path,parameter)

        accessor = self._build_accessor('__init__.__self__', **kwargs)
        return self._ironpy.apply(accessor, code, 'target', init, self._ns)

    def clear_function_usage(self,parameter,**kwargs):
        '''
        Remove any function usages applies to `parameter` on elements matching the query (kwargs)
        '''
        parameter = self._translate_property(parameter)
        init = '{"success":0,"fail":0}\n'
        param_path = parameter.split('.')
        parameter = param_path[-1]
        param_path = '.'.join(param_path[:-1])
        if len(param_path):
            param_path = '.'+param_path
        code = CLEAR_FUNCTION_LOOP%(param_path,parameter)
        accessor = self._build_accessor('__init__.__self__', **kwargs)
        return self._ironpy.apply(accessor, code, 'target', init, self._ns)

    def model_table(self, **kwargs):
        '''
        Build a dataframe of models in use
        '''
        names = self.enumerate_names(**kwargs)
        models = self.get_models(skip_nulls=False,**kwargs)
        self.name_columns
        rows = [dict(list(zip(self.name_columns, n)) + [('model', m)])
                for n, m in zip(names, models)]
        return pd.DataFrame(rows)

    def tabulate_parameters(self, model_type=None, **kwargs):
        '''
        Build DataFrame of model parameters.

        model_type - model type of interest.

        If None (default), do for ALL models used and return a dictionary of model types => parameter dataframes.
        '''
        def properties(m):
            return self._ironpy.find_parameters(m)

        def values(p, **kwargs):
            return self.get_param_values(p, **kwargs)

        return self._tabulate_properties(properties, values, model_type, **kwargs)

    def tabulate_inputs(self, model_type=None, input_set=None,**kwargs):
        def properties(m):
            return self._ironpy.find_inputs(m)

        def values(p, **kwargs):
            return self.get_data_sources(p, input_set=input_set, **kwargs)

        return self._tabulate_properties(properties, values, model_type, **kwargs)

    def tabulate_functions(self, model_type=None, **kwargs):
        def properties(m):
            return self._ironpy.find_inputs(m)

        def values(p, **kwargs):
            return self.get_functions(p, **kwargs)

        return self._tabulate_properties(properties, values, model_type, **kwargs)

    def _tabulate_properties(self, property_getter, value_getter, model_type=None, _property_lookup=None, _all_properties=None, _names=None, **kwargs):
        all_models = self.get_models(skip_nulls=False,**kwargs)
        if _property_lookup is None:
            _property_lookup = {m: [] if m is None else property_getter(m) for m in set(all_models)}
        if _all_properties is None:
            _all_properties = {m: [] if m is None else self._ironpy.find_properties(m) for m in set(all_models)}

        if _names is None:
            _names = list(self.enumerate_names(**kwargs))

        if model_type is None:
            models = set(all_models)
            return {m: None if m is None else self._tabulate_properties(property_getter, value_getter, m, _property_lookup, _all_properties, _names, **kwargs) for m in set(models)}

        model_type = self._ironpy.expand_model(model_type)
        table = {}
        for i, col_name in enumerate(self.name_columns):
            table[col_name] = [name_row[i] for j, name_row in enumerate(_names) if all_models[j] == model_type]

        for p in _property_lookup.get(model_type,[]):
            values = value_getter(p, **kwargs)
            if not len(values):
                continue
            table[p] = []

            for m in all_models:
                p_to_check = p.split('.')[0]
                if not p_to_check in _all_properties[m]:
                    continue

                if m == model_type:
                    table[p].append(values[0])

                values = values[1:]

        table = {k: v for k, v in table.items() if len(v) > 0}
        names = self.name_columns + [nm for nm in _property_lookup.get(model_type,[]) if nm in table]
        return pd.DataFrame(table, columns=names)

    def tabulate_list_values(self,property_name,columns,properties=None,**kwargs):
        if properties is None:
            properties = columns[:]
        code = TABULATION_SCRIPTLET.format(property_name=property_name,values=','.join(['row.'+p for p in properties]))
        vals = self.apply(code,init='[]',**kwargs)
        return pd.DataFrame(vals,columns=columns)

    def call(self, method, parameter_tuple=None, literal=False, fromList=False, **kwargs):
        accessor = self._build_accessor(method, **kwargs)
        return self._ironpy.call(accessor, parameter_tuple, literal=literal, from_list=fromList)

    def apply(self, code, name='target', init=None, namespace=None, **kwargs):
        accessor = self._build_accessor('__init__.__self__ ', **kwargs)
        return self._ironpy.apply(accessor[:-1], code, name, init, namespace)

    def _call(self, accessor, namespace=None):
        return self._ironpy.call(accessor, namespace)


class VeneerFunctionalUnitActions(VeneerNetworkElementActions):
    def __init__(self, catchment):
        self._catchment = catchment
        self._name_accessor = "definition.Name"
        super(VeneerFunctionalUnitActions, self).__init__(catchment._ironpy)
        self._build_pvr_accessor = self._build_fu_accessor
        self.name_columns = ['Catchment', 'Functional Unit']
        self.query_params = ['catchments','fus']

    def _build_accessor(self, parameter=None, catchments=None, fus=None):
        return self._build_fu_accessor(parameter, catchments, fus)

    def _build_fu_accessor(self, parameter=None, catchments=None, fus=None):
        accessor = 'scenario.Network.Catchments'

        if not catchments is None:
            catchments = _stringToList(catchments)
            accessor += '.Where(lambda c: c.DisplayName in %s)' % catchments

        accessor += '.*FunctionalUnits'

        if not fus is None:
            fus = _stringToList(fus)
            accessor += '.Where(lambda fu: fu.definition.Name in %s)' % fus

        if not parameter is None:
            accessor += '.*%s' % parameter

        return accessor

    def names(self, **kwargs):
        accessor = self._build_fu_accessor(self._name_accessor, **kwargs)
        return self._catchment._ironpy.get(accessor)

    def enumerate_names(self, **kwargs):
        fu_names = self.names(**kwargs)
        cname_accessor = self._build_fu_accessor(
            'catchment.DisplayName', **kwargs)
        catchment_names = self._catchment._ironpy.get(cname_accessor)
        return zip(catchment_names, fu_names)

    def rename(self,*args,**kwargs):
        all_pairs = list(args) + list(kwargs.items())

        script = 'config=scenario.SystemFunctionalUnitConfiguration\n'
        script += 'pairs = %s\n' % all_pairs
        script += 'for existing_name,new_name in pairs:\n'
        script += '  config.fuDefinitions.RenameItemInList(existing_name,new_name)'
        return self._ironpy.run_script(script)

class VeneerCatchmentActions(VeneerNetworkElementActions):
    '''
    Helpers for querying/modifying the catchment model setup

    Specific helpers exist under:

    * .runoff       (rainfall runoff in functional units)
    * .generation   (constituent generation in functional units)
    * .subcatchment (subcatchment level models)
    '''

    def __init__(self, ironpython):
        super(VeneerCatchmentActions, self).__init__(ironpython)
        self._ironpy = ironpython
        self.functional_units = VeneerFunctionalUnitActions(self)
        self.function_units = self.functional_units
        self.runoff = VeneerRunoffActions(self)
        self.generation = VeneerCatchmentGenerationActions(self)
        self.subcatchment = VeneerSubcatchmentActions(self)
        self._ns = None
        self._name_accessor = "Name"

    def _build_accessor(self, parameter=None, catchments=None):
        accessor = 'scenario.Network.Catchments'

        if not catchments is None:
            catchments = _stringToList(catchments)
            accessor += '.Where(lambda c: c.DisplayName in %s)' % catchments

        if not parameter is None:
            accessor += '.*%s' % parameter

        return accessor

    def get_areas(self, by_name=False, catchments=None):
        '''
        Return catchment area in square metres
        '''
        return self.get_param_values('characteristics.areaInSquareMeters', by_name=by_name, catchments=catchments)
#        return self._ironpy.get('scenario.Network.Catchments.*characteristics.areaInSquareMeters')

    def tabulate_areas(self,catchments=None):
        tbl = self.functional_units.tabulate_parameters(model_type='RiverSystem.Catchments.StandardFunctionalUnit',catchments=catchments)
        return tbl[['Catchment','Functional Unit','areaInSquareMeters']]

    def get_functional_unit_areas(self, catchments=None, fus=None):
        '''
        Return the area of each functional unit in each catchment:

        Parameters:

        catchments: Restrict to particular catchments by passing a list of catchment names

        fus: Restrict to particular functional unit types by passing a list of FU names
        '''
        return self.functional_units.get_param_values('areaInSquareMeters', catchments=catchments, fus=fus)
#        accessor = self._build_fu_accessor('areaInSquareMeters',catchments,fus)
#        return self._ironpy.get(accessor)

    def set_functional_unit_areas(self, values, catchments=None, fus=None):
        '''
        Set the area of each functional unit in each catchment:

        Parameters:

        values: List of functional unit areas to apply.

        catchments: Restrict to particular catchments by passing a list of catchment names

        fus: Restrict to particular functional unit types by passing a list of FU names
        '''
        return self.functional_units.set_param_values('areaInSquareMeters', values, fromList=True, catchments=catchments, fus=fus)
#        accessor = self._build_fu_accessor('areaInSquareMeters',catchments,fus)
#        return self._ironpy.set(accessor,values,fromList=True)

    def get_functional_unit_types(self, catchments=None, fus=None):
        '''
        Return a list of all functional unit types in the model
        '''
        return self.get_param_values('FunctionalUnits.*definition.Name')

    def add_functional_unit_type(self,fu_type):
        '''
        Add a new functional unit type, if it doesn't already exist in the model.
        '''
        script = self._ironpy._init_script()
        script += CREATE_FUNCTIONAL_UNIT%fu_type
        return self._ironpy._safe_run(script)

    def remove(self, name):
        '''
        Remove named catchment from the network
        '''
        script = self._ironpy._init_script('RiverSystem')
        script += self._ironpy._generator.find_feature_by_name()
        script += 'network = scenario.Network\n'
        script += 'catchment = network.CatchmentWithName("%s")\n' % name
        script += 'if catchment:\n'
        script += '    network.Remove.Overloads[RiverSystem.ICatchment](catchment)\n'
        return self._ironpy._safe_run(script)


class VeneerRunoffActions(VeneerFunctionalUnitActions):
    '''
    Helpers for querying/modifying the rainfall runoff model setup

    Query options:

    * catchments - the name(s) of catchments to match when querying/configuring.

    * fus - the type(s) of functional units to match when querying/configuring
    '''

    def __init__(self, catchment):
        super(VeneerRunoffActions, self).__init__(catchment)
        self._pvr_element_name = "Rainfall Runoff Model"
        self._pvr_attribute_prefix = self._pvr_element_name + '@'

    def _build_accessor(self, parameter=None, catchments=None, fus=None):
        accessor = super(VeneerRunoffActions, self)._build_accessor(
            'rainfallRunoffModel', catchments, fus)

        if not parameter is None:
            accessor += '.%s' % parameter

        return accessor

#    def get_models(self,catchments=None,fus=None):
#        return self.get_param_values('GetType().FullName',catchments,fus)

#    def get_param_values(self,parameter,catchments=None,fus=None):
#        accessor = self._build_accessor(parameter,catchments,fus)
#        return self._ironpy.get(accessor)

    def assign_time_series(self, parameter, values, data_group, column=0,
                           catchments=None, fus=None, literal=True, fromList=False):
        '''
        Assign an input time series to a rainfall runoff input
        '''
        accessor = self._build_accessor(parameter, catchments, fus)
        return self._ironpy.assign_time_series(accessor, values, from_list=fromList,
                                               literal=literal, column=column,
                                               data_group=data_group,
                                               namespace=self._ns)

class VeneerCatchmentGenerationActions(VeneerFunctionalUnitActions):
    '''
    Helpers for querying/modifying the constituent generation model setup

    Query options:

    * catchments - the name(s) of catchments to match when querying/configuring.

    * fus - the type(s) of functional units to match when querying/configuring.

    * constituents - the name(s) of the constituents to match when querying/configuring.

    * sources - the name(s) of specific constituent sources to match when querying/configuring.
    '''

    def __init__(self, catchment):
        super(VeneerCatchmentGenerationActions, self).__init__(catchment)
        self._ns = 'RiverSystem.Constituents.CatchmentElementConstituentData as CatchmentElementConstituentData'
        self.name_columns = ['Catchment', 'Functional Unit',
                             'Constituent', 'ConstituentSource']
        self.query_params = self.query_params + ['constituents','sources']

    def _build_accessor(self, parameter, catchments=None, fus=None, constituents=None, sources=None):
        accessor = 'scenario.Network.ConstituentsManagement.Elements' + \
            '.OfType[CatchmentElementConstituentData]()'

        if not catchments is None:
            catchments = _stringToList(catchments)
            accessor += '.Where(lambda cData: cData.Catchment.Name in %s)' % catchments

        accessor += '.*FunctionalUnitData'

        if not fus is None:
            fus = _stringToList(fus)
            accessor += '.Where(lambda fuData: fuData.DisplayName in %s)' % fus

        accessor += '.*ConstituentModels'

        if not constituents is None:
            constituents = _stringToList(constituents)
            accessor += '.Where(lambda x: x.Constituent.Name in %s)' % constituents

        accessor += '.*ConstituentSources'

        if not sources is None:
            sources = _stringToList(sources)
            accessor += '.Where(lambda cs: cs.Name in %s)' % sources

        accessor += '.*GenerationModel'
        if not parameter is None:
            accessor += '.%s' % parameter

        return accessor

    def enumerate_names(self, fu_only=False, **kwargs):
        if fu_only:
            fu_names = self.names(**kwargs)
            cname_accessor = self._build_fu_accessor(
                'catchment.DisplayName', **kwargs)
            catchment_names = self._catchment._ironpy.get(cname_accessor)

            return zip(fu_names, catchment_names)

        accessor = self._build_accessor(None, **kwargs)
        names = self._ironpy.get(accessor,
                                 self._ns,
                                 names=['cat', 'fu', 'con', 'src'],
                                 alt_expression='(cat.DisplayName,fu.DisplayName,con.DisplayName,src.DisplayName)')
        return [tuple(n) for n in names]


class VeneerSubcatchmentActions(VeneerNetworkElementActions):
    '''
    Helpers for querying/modifying the subcatchment-level models

    Query options:

    * catchments - the name(s) of catchments to match when querying/configuring.
    '''

    def __init__(self, catchment):
        self._catchment = catchment
        self._name_accessor = 'Catchment.DisplayName'
        super(VeneerSubcatchmentActions, self).__init__(catchment._ironpy)

    def _build_accessor(self, parameter, models=True, **kwargs):
        accessor = 'scenario.Network.Catchments'

        if not kwargs.get('catchments') is None:
            catchments = _stringToList(kwargs['catchments'])
            accessor += '.Where(lambda sc: sc.DisplayName in %s)' % catchments

        if models:
            accessor += '.*CatchmentModels'

        if not parameter is None:
            accessor += '.*%s' % parameter
        return accessor

#    def get_param_values(self,parameter,**kwargs):
#        accessor = self._build_accessor(parameter,**kwargs)
#        return self._ironpy.get(accessor)

#    def get_models(self,**kwargs):
#        return self.get_param_values('GetType().FullName',**kwargs)

    def add_model(self, model_type, add_if_existing=False, catchments=None, allow_duplicates=False):
        accessor = self._build_accessor(parameter=None, catchments=catchments)
        return self._ironpy.add_to_list(accessor, model_type, model_type,
                                        instantiate=True, allow_duplicates=allow_duplicates)

    def reset(self, namespace=None, **kwargs):
        self._call(self._build_accessor(
            'reset()', models=False, **kwargs), namespace)


class VeneerLinkActions(object):
    def __init__(self, ironpython):
        self._ironpy = ironpython
        self.constituents = VeneerLinkConstituentActions(self)
        self.routing = VeneerLinkRoutingActions(self)
        self._name_accessor = '.DisplayName'

    def create(self, from_node, to_node, name=None, routing=None,allow_duplicates=False):
        script = self._ironpy._init_script()
        if routing is None:
          routing = 'RiverSystem.Flow.StraightThroughRouting'
        script += f'import {routing}\n'
        script += self._ironpy._generator.find_feature_by_name()
        script += 'n1 = find_feature_by_name("%s",exact=True)\n' % from_node
        script += 'n2 = find_feature_by_name("%s",exact=True)\n' % to_node
        indent = ''
        if name and not allow_duplicates:
            script += 'if not find_feature_by_name("%s",exact=True):\n' % name
            indent = '  '
        script += indent + f'result = scenario.Network.Connect(n1,n2,{routing}())\n'
        if name:
            script += indent + 'result.Name = "%s"' % name
        return self._ironpy._safe_run(script)


class VeneerNetworkElementConstituentActions(VeneerNetworkElementActions):
    def __init__(self, ironpy):
        super(VeneerNetworkElementConstituentActions, self).__init__(ironpy)
        self._aspect_pre_modifer = {
            '': '',
            'played': 'Data.ConstituentPlayedValues',
            'model': 'Data.ProcessingModels',
            'initial': 'InitialConcentrations.Concentrations'
        }
        self._aspect_post_modifer = {
            'model': '.*Model',
            'played': '.*',
            '': '',
            'initial': '.*'
        }
        self.name_columns = ['NetworkElement', 'Constituent']
        self.query_params = ['constituents']

    def assign_time_series(self, parameter, values, data_group, column=0,
                           literal=True, fromList=False, aspect='played', **kwargs):
        '''
        Assign an input time series to a model input/parameter

        '''
        accessor = self._build_accessor(parameter, aspect=aspect, **kwargs)
        return self._ironpy.assign_time_series(accessor, values, from_list=fromList,
                                               literal=literal, column=column,
                                               data_group=data_group, namespace=self._ns)

    def _build_accessor(self, parameter=None, constituents=None, aspect=None, played_type=None, **kwargs):
        aspect = self._default_aspect if aspect is None else aspect
        accessor = 'scenario.Network.ConstituentsManagement.Elements'
        accessor += self._filter_constituent_data_types()
        accessor += self._filter_by_query(**kwargs)
        accessor += '.*%s' % self._aspect_pre_modifer[aspect]

        if aspect == 'played' and played_type:
            accessor += '.Where(lambda p: p.PlayedType==ConstituentPlayedType.%s)' % played_type

        if not constituents is None:
            constituents = _stringToList(constituents)
            if accessor[-1] != '*':
                accessor += '.'
            accessor += 'Where(lambda c: c.Constituent.Name in %s)' % constituents

        accessor += self._aspect_post_modifer[aspect]

        if not parameter is None:
            if accessor[-1] != '*':
                accessor += '.'
            accessor += parameter
        return accessor

    def initialise_played_constituents(self, played_type='varConcentration', **kwargs):
        accessor = self._build_accessor(aspect='', **kwargs)
        script = self._ironpy._init_script(self._ns)
        script += 'from RiverSystem.Constituents.ConstituentPlayedValue import ConstituentPlayedType as ConstituentPlayedType\n'
        script += "from RiverSystem.Constituents import ConstituentPlayedValue as ConstituentPlayedValue\n"
        script += 'playType = ConstituentPlayedType.%s\n' % played_type
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
        script += self._ironpy._generateLoop(accessor, innerLoop, first=False)

        return self._ironpy._safe_run(script)

    def enumerate_names(self, fu_only=False, **kwargs):
        accessor = self._build_accessor(None, **kwargs)
        names = self._ironpy.get(accessor,
                                 self._ns,
                                 names=['ne', 'con'],
                                 alt_expression='(ne.DisplayName,con.Constituent.Name)')
        return [tuple(n) for n in names]


class VeneerLinkConstituentActions(VeneerNetworkElementConstituentActions):
    def __init__(self, link):
        self._link = link
        self._name_accessor = 'Link.DisplayName'
        super(VeneerLinkConstituentActions, self).__init__(link._ironpy)
        self.name_columns =['Link','Constituent']
        self.query_params=['links','constituents']
        self._ns = 'RiverSystem.Constituents.LinkElementConstituentData as LinkElementConstituentData'
        self._ns += '\nfrom RiverSystem.Constituents.ConstituentPlayedValue import ConstituentPlayedType as ConstituentPlayedType\n'
        self._default_aspect = 'model'

    def _filter_constituent_data_types(self):
        return '.OfType[LinkElementConstituentData]()'

    def _filter_by_query(self, links=None):
        if links is None:
            return ''

        links = _stringToList(links)
        return '.Where(lambda lecd: lecd.Element.DisplayName in %s)' % links


class VeneerLinkRoutingActions(VeneerNetworkElementActions):
    '''
    Queries and actions relating to streamflow routing models.

    Query options:

    * links - the name(s) of links to match when querying/configuring.

    For example:

    v.model.links.routing.get_models(links=['Link #1','Link #2'])
    '''

    def __init__(self, link):
        self._link = link
        self._name_accessor = 'link.DisplayName'
        super(VeneerLinkRoutingActions, self).__init__(link._ironpy)
        self._pvr_element_name = None

    def _build_accessor(self, parameter=None, links=None):
        accessor = 'scenario.Network.Links'
        if not links is None:
            links = _stringToList(links)
            accessor += '.Where(lambda l:l.DisplayName in %s)' % links
        accessor += '.*FlowRouting'

        if not parameter is None:
            accessor += '.%s' % parameter

        return accessor

#    def set_model(self,theThing,theValue,namespace=None,literal=False,fromList=False,instantiate=False):
    def set_models(self, models, fromList=False, **kwargs):

        models = _stringToList(models)
        assignment = "theLink=%s%s\n"
        assignment += "val = newVal"

        post_assignment = "\n"
        post_assignment += "is_sr = 'StorageRouting' in val.__name__\n"
        post_assignment += "theLink.FlowRouting = val(theLink) if is_sr else val()"

        accessor = self._build_accessor(**kwargs)[:-13] + '.*__init__.__self__'
        namespace = self._instantiation_namespace(models)

        return self._ironpy._assignment(accessor, models, namespace, literal=False, fromList=True,
                                        instantiate=False,
                                        assignment=assignment,
                                        post_assignment=post_assignment)

    def piecewise_routing_table(self,link):
        return self.tabulate_list_values('Piecewises',
                                         ['IndexFlow','TravelTime'],
                                         links=[link])

    def load_piecewise_routing_table(self,table,links):
        piecewise_txt = '\n'.join(['pw_table.Add(Piecewise(IndexFlow=%f,TravelTime=%f))'%(r['IndexFlow'],r['TravelTime']) for _,r  in table.iterrows()])
        code = LOAD_PIECEWISE_ROUTING_TABLE_SCRIPTLET%piecewise_txt
        return self.apply(code,init='0',links=links)

    def rating_table_meta(self,link):
        return self.tabulate_list_values('link.RatingCurveLibrary.Curves',
                                        ['Name','StartDate','OverbankFlowlevel'],
                                        links=link)

    def rating_tables(self,link,idx=0):
        return self.tabulate_list_values(f'link.RatingCurveLibrary.Curves[{idx}].Points',
                                        ['Level','Discharge','Width','DeadStorage'],
                                        links=link)

    def load_rating_table(self,table,start,overbank,links):
        rating_txt = '\n'.join(['curve.Points.Add(LinkRatingCurvePoint(Level=%f,Discharge=%f,Width=%f,DeadStorage=%s))'% \
                                (r['Level'],r['Discharge'],r['Width'],r['DeadStorage']) for _,r  in table.iterrows()])
        code = LOAD_RATING_TABLE_SCRIPTLET%(rating_txt,start.year,start.month,start.day,overbank)
        return self.apply(code,init='0',links=links)


class VeneerNodeActions(VeneerNetworkElementActions):
    '''
    Queries and actions relating to nodes (incuding node models).

    Query options:

    * nodes - the name(s) of nodes to match when querying/configuring.

    * node_types - the type(s) of nodes to match when querying/configuring

    * splitter - whether to target splitter nodes

    * raw - whether to target the node itself

    For example:

    v.model.nodes.get_models(nodes='Fish River')
    '''

    def __init__(self, ironpython):
        super(VeneerNodeActions, self).__init__(ironpython)
        self._name_accessor = 'Node.Name'
        self.constituents = VeneerNodeConstituentActions(self)
        from .nodes.storages import VeneerStorageActions
        self.storages = VeneerStorageActions(self)

        from .nodes.water_users import VeneerWaterUserActions
        self.water_users = VeneerWaterUserActions(self)

        from .nodes.losses import VeneerLossNodeActions
        self.losses = VeneerLossNodeActions(self)

        from .nodes.partitioning import VeneerFlowPartitioningActions
        self.partitioning = VeneerFlowPartitioningActions(self)

        from .nodes.gauges import VeneerGaugeActions
        self.gauges = VeneerGaugeActions(self)

        from .nodes.farm_dams import VeneerFarmDamActions
        self.farm_dams = VeneerFarmDamActions(self)

        from .nodes.inflows import VeneerInflowActions
        self.inflows = VeneerInflowActions(self)

    def _refine_accessor(self, node_access='', nodes=None, node_types=None, splitter=False):
        accessor = ""
        if not nodes is None:
            nodes = _stringToList(nodes)
            accessor += '.Where(lambda n:n%s.Name in %s)' % (node_access, nodes)
        if not node_types is None:
            node_types = _stringToList(node_types)
            node_types = [_transform_node_type_name(n) for n in node_types]
            accessor += '.Where(lambda n:n%s.%s and n%s.%s.GetType().Name.Split(".").Last() in %s)' % (
                node_access, self._model_property(splitter), node_access, self._model_property(splitter), node_types)
        return accessor

    def _model_property(self, splitter):
        return 'FlowPartitioning' if splitter else 'NodeModel'

    def _build_accessor(self, parameter=None, nodes=None, node_types=None, raw=False, splitter=False):
        accessor = 'scenario.Network.Nodes'
        accessor += self._refine_accessor(nodes=nodes, node_types=node_types,splitter=splitter)

        next_wildcard='.'
        if raw:
            next_wildcard='.*'
        else:
            accessor += '.*%s' % self._model_property(splitter)

        if not parameter is None:
            accessor += f'{next_wildcard}{parameter}'

        return accessor

    def names(self, raw=True,**kwargs):
        '''
        Return the names of the network elements
        '''
        if raw:
            return self.get_param_values('Name', raw=True,**kwargs)
        return self.get_param_values(self._name_accessor, **kwargs)

    def model_table(self, **kwargs):
        '''
        Build a dataframe of models in use
        '''
        kwargs['raw'] = True
        names = self.enumerate_names(**kwargs)

        kwargs['raw'] = False
        kwargs['splitter'] = False
        models = self.get_models(skip_nulls=False,**kwargs)

        kwargs['splitter'] = True
        splitters = self.get_models(skip_nulls=False, **kwargs)

        self.name_columns
        rows = [dict(list(zip(self.name_columns, n)) + [('model', m), ('splitter', s)])
                for n, m, s in zip(names, models, splitters)]
        return pd.DataFrame(rows)

    def create(self, name, node_type, location=None, schematic_location=None, splitter=False,**kwargs):
        is_supply_point = 'ExtractionNodeModel' in node_type

        script = self._ironpy._init_script('.'.join(node_type.split('.')[:-1]))
        script += 'import RiverSystem.E2ShapeProperties as E2ShapeProperties\n'
        script += 'clr.AddReference("System.Drawing")\n'
        script += 'import System.Drawing.PointF as PointF\n'
        script += 'import RiverSystem.Utils.RiverSystemUtils as rsutils\n'
        if is_supply_point:
            script += 'from RiverSystem.Flow import ExtractionPartitioner\n'
        script += 'network = scenario.Network\n'
        script += 'new_node = RiverSystem.Node()\n'
        if location:
            script += 'new_node.location.E = %f\n' % location[0]
            script += 'new_node.location.N = %f\n' % location[1]

        script += 'new_node.Name = "%s"\n' % name
        for k, v in kwargs.items():
            script += 'new_node.%s = %s\n' % (k, v)

        script += 'new_node.%s = %s()\n' % (self._model_property(splitter), node_type)
        if is_supply_point:
            script += 'new_node.FlowPartitioning = ExtractionPartitioner()\n'

        script += 'rsutils.SeedEntityTarget(new_node.%s,scenario)\n' % self._model_property(
            splitter)
        script += 'network.Add.Overloads[RiverSystem.INetworkElement](new_node)\n'

        if(schematic_location):
            script += 'schematic = H.GetSchematic(scenario)\n'
            # script += 'if schematic:\n'
            script += 'shp = E2ShapeProperties()\n'
            script += 'shp.Feature = new_node\n'
            script += 'shp.Location = PointF(%f,%f)\n'%(schematic_location[0],schematic_location[1])
            # script += 'shp.Location.X = %f\n' % schematic_location[0]
            # script += 'shp.Location.Y = %f\n' % schematic_location[1]
            script += 'schematic.ExistingFeatureShapeProperties.Add(shp)\n'
            script += 'print("Set location")\n'
        script += 'result = new_node\n'
        return self._ironpy._safe_run(script)
        # schematic_location???

    def remove(self, name):
        script = self._ironpy._init_script('RiverSystem')
        script += self._ironpy._generator.find_feature_by_name()
        script += 'network = scenario.Network\n'
        script += 'node = find_feature_by_name("%s",exact=True)\n' % name
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

    def __init__(self, node):
        self._node = node
        self._name_accessor = 'Element.Name'
        super(VeneerNodeConstituentActions, self).__init__(self._node._ironpy)
        self._ns = ['RiverSystem.Node as Node',
                    'RiverSystem.Constituents.NetworkElementConstituentData as NetworkElementConstituentData']
        self._default_aspect = ''  # 'Data.ConstituentPlayedValues'

    def _filter_constituent_data_types(self):
        return '.OfType[NetworkElementConstituentData]().Where(lambda d: isinstance(d.Element,Node))'

    def _filter_by_query(self, **kwargs):
        return self._node._refine_accessor(node_access='.Element', **kwargs)

#    def _build_accessor(self,parameter,nodes=None,node_types=None)
#        accessor = 'scenario.Network.ConstituentsManagement.Elements' + \
#        accessor += self._node._refine_accessor(nodes,node_types)


class VeneerFunctionActions():
    '''
    Routines for managing Source functions.
    '''

    def __init__(self, ironpython):
        self._ironpy = ironpython

    def _accessor(self, option, functions=None):
        accessor = 'scenario.Network.FunctionManager.Functions'
        if functions is not None:
            functions = _stringToList(functions)
            accessor += '.Where(lambda v: v.Name in %s)' % functions
        accessor += '.*' + option

        return accessor

    def create_functions(self, names, general_equation, params=[[]], name_params=None, use_format=False,function_path=None):
        '''
        Create one function, or multiple functions based on a pattern

        names: Either a list of function names OR a generic name to be filled in using the values of name_params

        general_equation: A general equation with %%s, %%f, etc substitution characters where params will be included

        params: A list of parameters to the general_equation.
                These can be anything, but will often be the names of modelled variables, and/or scalars

        name_params: A list of tuples, containing the name parameters to substitute into the names (if a template name is provided)

        use_format: If True, use str.format() to substitute parameters into the general_equation and names.
                    If False, use the old-style % operator.

        function_path: If provided, the path within the function library to store the function(s) in.
                          If the path does not exist, it will be created.

        Returns a dictionary with keys created and failed, each a list of function names
        '''
        names = _stringToList(names)
        if len(names) == 1:
            if name_params:
                names = ['$' + _variable_safe_name(names[0] % name_param_set)
                         for name_param_set in name_params]
            elif len(params)>1:
                names = ['%s_%d' % (names[0], d) for d in range(len(params))]

        if use_format:
            functions = list(
                zip(names, [general_equation.format(**param_set) for param_set in params]))
        else:
            functions = list(
                zip(names, [general_equation % param_set for param_set in params]))

        if function_path is not None:
            if function_path.endswith('.'):
                function_path = function_path[:-1]
            if function_path.startswith('$'):
                function_path = function_path[1:]
            function_path = f"'{function_path}'"
        script = self._ironpy._init_script() + CREATE_FUNCTIONS%(function_path,functions)
        result = self._ironpy.run_script(script)
        if not result['Exception'] is None:
            raise Exception(result['Exception'])
#        data = result['Response']['Value'] if result['Response'] else result['Response']
        return self._ironpy.simplify_response(result['Response'])

    def create_time_series_variables(self,variable_names,data_source,columns):
        script=self._ironpy._init_script() + CREATE_TS_VARIABLE_SCRIPT%(data_source,variable_names,columns)
        result = self._ironpy.run_script(script)
        if not result['Exception'] is None:
            raise Exception(result['Exception'])
        return self._ironpy.simplify_response(result['Response'])

    def create_piecewise_variable(self,variable_name,table=None):
        x_name = 'Lookup'
        y_name = 'Result'
        if not variable_name.startswith('$'):
            variable_name = '$' + variable_name
        if table is not None:
            x_name = table.columns[0]
            y_name = table.columns[1]
        script=self._ironpy._init_script() + CREATE_PIECEWISE_VARIABLE_SCRIPT%(variable_name,x_name,y_name)
        result = self._ironpy.run_script(script)
        if not result['Exception'] is None:
            raise Exception(result['Exception'])
        if table is not None:
            self._ironpy._veneer.update_variable_piecewise(variable_name,table)

    def delete_variables(self, names):
        script = self._ironpy._init_script()
        script += 'names = %s\n' % names
        script += 'to_remove = scenario.Network.FunctionManager.Variables.Where(lambda v: v.Name in names).ToList()\n'
        script += 'result = [v.Name for v in to_remove]\n'
        script += 'for v in to_remove: scenario.Network.FunctionManager.Variables.Remove(v)\n'
        result = self._ironpy.run_script(script)
        if not result['Exception'] is None:
            raise Exception(result['Exception'])
#        data = result['Response']['Value'] if result['Response'] else result['Response']
        return self._ironpy.simplify_response(result['Response'])

    def delete_functions(self, names):
        script = self._ironpy._init_script()
        script += 'names = %s\n' % names
        script += 'to_remove = scenario.Network.FunctionManager.Functions.Where(lambda v: v.Name in names).ToList()\n'
        script += 'result = [v.Name for v in to_remove]\n'
        script += 'for v in to_remove: scenario.Network.FunctionManager.Functions.Remove(v)\n'
        result = self._ironpy.run_script(script)
        if not result['Exception'] is None:
            raise Exception(result['Exception'])
#        data = result['Response']['Value'] if result['Response'] else result['Response']
        return self._ironpy.simplify_response(result['Response'])

    def get_options(self, option, functions=None):
        '''
        Return the current value of `option` for one or more functions.
        '''
        accessor = self._accessor(option, functions)
        resp = self._ironpy.get(accessor)
        return resp

    def get_options(self, option, functions=None):
        '''
        Return the current value of `option` for one or more functions.

        option - one of
            'EvaluationTimes'
            'ForceEvaluate'
            'InitialValue'
            'ResultUnit'
            'IsClone'
            'IsConstant' (readonly)
            'Expression'
            'Transient'
            'HasContextVariables' (readonly)
            'CanBeRecorded' (readonly)
            'Name'
            'FullName' (readonly)

        '''
        accessor = self._accessor(option, functions)
        resp = self._ironpy.get(accessor)
        return resp

    def set_options(self, option, values, fromList=False, functions=None, literal=False):
        '''
        Set an option on one or more functions.

        option - can be any of the known option types (see get_options) OR 'SIUnit'.

        When option=='SIUnit', the values are parsed using the standard SI unit parsing within Source.
        '''
        if option=='ResultUnit':
            values = _stringToList(values)
            fromList=True
            values = [f'Unit.PredefinedUnit(CommonUnits.{u})' for u in values]
        if option=='SIUnit':
            values = _stringToList(values)
            fromList=True
            option = 'ResultUnit'
            values = [f'Unit.parse("{u}")' for u in values]

        accessor = self._accessor(option, functions)
        ns = 'RiverSystem.Management.ExpressionBuilder.TimeOfEvaluation as TimeOfEvaluation'
        ns += '\nimport RiverSystem.Utils.UnitLibrary as UnitLibrary\n'
        ns += '\nfrom TIME.Core import Unit, CommonUnits\n'
        return self._ironpy.set(accessor, values, ns, literal=literal, fromList=fromList)

    def set_time_of_evaluation(self, toe, fromList=False, functions=None):
        '''
        Set the Time of Evaluation for one or more functions.

        Currently only supports setting a *single* time of evaluation.abs

        Should be a string and one of:
            'None'
            'EndOfTimeStep'
            'DuringResourceAssessmentTriggers'
            'DuringOrderingPhase'
            'DuringFlowPhase'
            'StartOfTimeStep'
            'PostTimeStep'
            'PostFlowPhase'
            'DuringResourceAssessmentEntry'
            'ElementPostConstraints'
            'ElementPostOrdering'
            'ElementPostFlow'
            'InitialiseRun'
            'StartOfRun'
        '''

        toe = _stringToList(toe)
        toe = ['TimeOfEvaluation.%s' % s for s in toe]
        return self.set_options('EvaluationTimes', toe, fromList=True, functions=functions)

    def get_known_time_periods(self):
        script = self._ironpy._init_script() + GET_TIME_PERIODS
        return self._ironpy.simplify_response(self._ironpy._safe_run(script)['Response'])

    def set_modelled_variable_time_period(self,tp,variables=None):
        template = self._ironpy._init_script() + SET_TIME_PERIODS
        script = template%(tp,variables)
        self._ironpy._safe_run(script)

    def set_modelled_variable_units(self,units,variables):
        template = self._ironpy._init_script() + SET_MODEL_VARIABLE_UNITS
        script = template%(variables,units)
        self._ironpy._safe_run(script)

class VeneerSimulationActions():
    def __init__(self, ironpython):
        self._ironpy = ironpython

    def get_configuration(self):
        script = self._ironpy._init_script()

        script += 'result = scenario.RunManager.CurrentConfiguration.GetType().FullName'
        result = self._ironpy.run_script(script)
        if not result['Exception'] is None:
            raise Exception(result['Exception'])
#        data = result['Response']['Value'] if result['Response'] else result['Response']
        return self._ironpy.simplify_response(result['Response'])

    def select_input_set(self,name):
        '''
        Set the current input set (for current scenario and configuration) to the one with the given name.

        name - name of the input set to select. If it does not exist, an exception will be raised.

        New input sets can be created using v.create_input_set(name, configuration)
        '''
        script = self._ironpy._init_script()+SET_INPUT_SET_SCRIPT%name
        return self._ironpy._safe_run(script)

    def current_input_set(self):
        '''
        Get the name of the currently selected input set for the current scenario and configuration.

        Returns:
            * The name of the currently selected input set, or None if no input set is selected.

        Details of the input set can be retrieved using v.input_sets()
        '''
        return self._ironpy.get('scenario.CurrentConfiguration.SelectedInputSet.Name')

    def timestep_in_seconds(self):
        return self._ironpy.get('scenario.CurrentConfiguration.TimeStep.span.TotalSeconds')

    def get_assurance_rules(self):
        columns = ['Category', 'Name', 'LogLevel']

        prefix = 'scenario.Network.AssuranceManager.DefaultLogLevels'
        default_values = {col: self._ironpy.get(
            '%s.*%s' % (prefix, col)) for col in columns}

        ns = 'RiverSystem.Assurance.AssuranceConfiguration as AssuranceConfiguration'
        prefix = 'scenario.GetScenarioConfiguration[AssuranceConfiguration]().Entries'
        overwritten_values = {col: self._ironpy.get(
            '%s.*%s' % (prefix, col), namespace=ns) for col in columns}

        combined = pd.concat(
            [pd.DataFrame(default_values), pd.DataFrame(overwritten_values)])
        return combined.drop_duplicates(subset=['Category', 'Name'], keep='last')

    def configure_assurance_rule(self, level='Off', rule=None, category=None):
        level = '"%s"' % level
        if rule is not None:
            rule = '"%s"' % rule
        if category is not None:
            category = '"%s"' % category
        script = self._ironpy._init_script(
        ) + 'H.ConfigureAssuranceRule(scenario,%s,%s,%s)' % (level, rule, category)
        return self._ironpy.run_script(script)

class VeneerGeographicDataActions(object):
    def __init__(self,ironpython):
        self._ironpy = ironpython

    def has_raster(self,r):
        try:
            return self._ironpy.get(f'scenario.GeographicData.{r}.Columns') > 0
        except:
            return False

    def raster_categories(self,r):
        kvps = [kvp['Value'][1:-1].split(', ',1) for kvp in self._ironpy.get(f'scenario.GeographicData.{r}.Categories')]
        kvps = [(int(c),v) for c,v in kvps]
        return dict(kvps)

    def save_raster(self,r,fn,categories=False):
        script = SAVE_RASTER_SCRIPT%(os.path.abspath(fn),r)
        res = self._ironpy.run_script(script)

        if categories:
            raster_categories = self.raster_categories(r)
            with open(f'{fn}.json','w') as fp:
                json.dump(raster_categories,fp,indent=2)
        return res

# all_names=v.model.catchment.enumerate_names()
# var_names=v.model.name_subst("tss_load_%s_%s",all_names)
# v.model.catchment.generation.create_modelled_variable('load','tss_load_$catchment_$fu',constituents="tss")
# function_codes=v.model.name_subst("exp(%s,1.2)",var_names)
# function_names=v.model.name_subst("metals_%s_%s",all_names)
# v.model.functions.create_function(function_names,function_codes)
# v.model.catchment.generation.apply_function("ObservedLoad",function_names,constituents="metals")


def build_dynamic_methods():
    def add_node_creator(name, klass):
        def creator(self, node_name, location=None, schematic_location=None):
            return self.create(node_name, klass, location, schematic_location)
        creator.__name__ = "new_%s" % name
        setattr(VeneerNodeActions, creator.__name__, creator)

    for name, klass in NODE_TYPES.items():
        add_node_creator(name, klass)


build_dynamic_methods()
