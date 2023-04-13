import pandas as pd
from veneer.server_side import VeneerNetworkElementActions

class VeneerFlowPartitioningActions(VeneerNetworkElementActions):
    def __init__(self,node_actions):
        self.node_actions = node_actions
        self._name_accessor = self.node_actions._name_accessor
        super(VeneerFlowPartitioningActions, self).__init__(node_actions._ironpy)

    def _build_accessor(self, parameter=None, nodes=None):
        return self.node_actions._build_accessor(parameter,nodes=nodes,
                                                splitter=True,node_types='RegulatedEffluentPartitioner')

    def effluent_relationship(self,node):
        '''
        Retrieve the effluent relationship table for a given node
        '''
        return self.tabulate_list_values('EffluentRelationship',['UpstreamFlow','MinimumEffluent','MaximumEffluent'],nodes=[node])

