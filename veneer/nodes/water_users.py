from veneer.server_side import VeneerNetworkElementActions
from types import MethodType
from veneer.utils import _quote_string

DEMAND_TYPES={
    'timeseries':'RiverSystem.DemandModels.TimeSeries.TimeSeriesDemandNodeModel',
    'irrigator':'RiverSystem.DemandModels.Irrigator.IrrigatorDemand',
    'monthly_pattern':'RiverSystem.DemandModels.MonthlyPattern.MonthlyDemandNodeModel'
}

ADD_DEMAND_SCRIPTLET='''
ignoreExceptions=False
from %s import %s

new_demand = %s()
new_demand.Name = %s

existing_demand = target.AvailableConfiguredDemand.FirstOrDefault(lambda d:d.Name==new_demand.Name)
if existing_demand is not None:
    target.AvailableConfiguredDemand.Remove(existing_demand)

target.AvailableConfiguredDemand.Add(new_demand)

if %s:
    target.DemandModel = new_demand
result += 1
'''

ACTIVATE_DEMAND_SCRIPTLET='''
ignoreExceptions=False

existing_demand = target.AvailableConfiguredDemand.FirstOrDefault(lambda d:d.Name==%s)
if existing_demand is not None:
    target.DemandModel = existing_demand
    result += 1
'''

class VeneerWaterUserActions(VeneerNetworkElementActions):
    def __init__(self,node_actions):
        self.node_actions = node_actions
        self._name_accessor = self.node_actions._name_accessor
        super(VeneerWaterUserActions, self).__init__(node_actions._ironpy)
        for demand in DEMAND_TYPES.keys():
            def method(r):
                def add_demand_of_type(self,name=None,activate=False,nodes=None):
                    return self.add_demand(r,name,activate,nodes)
                add_demand_of_type.__doc__ = self.add_demand.__doc__
                return add_demand_of_type
            setattr(self,'add_%s'%demand,MethodType(method(demand), self))


    def _build_accessor(self, parameter=None, nodes=None):
        return self.node_actions._build_accessor(parameter,nodes=nodes,node_types='WaterUserNodeModel')

    def add_demand(self,demand_type,name=None,activate=False,nodes=None):
        '''
        Create a new demand on one or more water user nodes

        name: name for new demand. If not provided, will be '<demandtype> from script'.
              IF the name already exists on another demand on a particular water user, the existing demand will be replaced!

        activate: if True, set the new demand model to be the active demand model on the water user
        '''
        if not demand_type in DEMAND_TYPES:
            raise Exception("Unknown demand type: %s"%demand_type)
        if name is None:
            name = '"%s from script"'%demand_type
        name = _quote_string(name)

        klass = DEMAND_TYPES[demand_type]
        namespace = '.'.join(klass.split('.')[:-1])
        klass = klass.split('.')[-1]

        code = ADD_DEMAND_SCRIPTLET%(namespace,klass,klass,name,activate)

        return self.node_actions.apply(code,init='0',node_types='WaterUserNodeModel',nodes=nodes)

    def set_active_demand(self,name,nodes=None):
        '''
        Set the active demand model for the water user node
        '''
        code = ACTIVATE_DEMAND_SCRIPTLET%_quote_string(name)
        return self.node_actions.apply(code,init='0',node_types='WaterUserNodeModel',nodes=nodes)

    def demands(self,nodes=None):
        return self.get_param_values('AvailableConfiguredDemand.*Name',nodes=nodes)

