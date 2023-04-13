import pandas as pd
from veneer.server_side import VeneerNetworkElementActions

GET_EFFLUENT_RELATIONSHIP_SCRIPTLET='''
ignoreExceptions=False
tbl = target.EffluentRelationship
for ix in range(tbl.Count):
    row = tbl[ix]
    result.append((row.UpstreamFlow,row.MinimumEffluent,row.MaximumEffluent))
'''

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
        code = GET_EFFLUENT_RELATIONSHIP_SCRIPTLET
        vals = self.apply(code,init='[]',nodes=[node])
        return pd.DataFrame(vals,columns=['upstream_flow','minimum_effluent','maximum_effluent'])

